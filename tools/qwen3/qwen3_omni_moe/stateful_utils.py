import numpy as np
import openvino as ov
from openvino import opset13

from .constants import BEAM_IDX_NAME, INPUTS_EMBEDS, PKV_INPUT_PREFIX, PKV_OUTPUT_PREFIX


def model_has_input_output_name(ov_model: ov.Model, name: str) -> bool:
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])


def fuse_cache_reorder(
    ov_model: ov.Model,
    not_kv_inputs: list[str],
    key_value_input_names: list[str],
    gather_dim: int,
) -> None:
    if model_has_input_output_name(ov_model, BEAM_IDX_NAME):
        raise ValueError("Model already has fused cache")
    input_batch = ov_model.input(INPUTS_EMBEDS).get_partial_shape()[0]
    beam_idx = opset13.parameter(name=BEAM_IDX_NAME, dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({BEAM_IDX_NAME})
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])
    for input_name in key_value_input_names:
        parameter_output_port = ov_model.input(input_name)
        consumers = parameter_output_port.get_target_inputs()
        gather = opset13.gather(parameter_output_port, beam_idx, opset13.constant(gather_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()


def build_state_initializer(ov_model: ov.Model, batch_dim: int) -> None:
    input_ids = ov_model.input(INPUTS_EMBEDS)
    batch = opset13.gather(
        opset13.shape_of(input_ids, output_type="i64"),
        opset13.constant([0]),
        opset13.constant(0),
    )
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [dim.min_length for dim in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [
                (opset13.constant(np.array([dim], dtype=np.int64)) if isinstance(dim, int) else dim) for dim in dims
            ]
            shape = opset13.concat(dims, axis=0)
            broadcast = opset13.broadcast(opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape)
            op.set_arguments([broadcast])
    ov_model.validate_nodes_and_infer_types()


def make_stateful(
    ov_model: ov.Model,
    key_value_input_names: list[str],
    key_value_output_names: list[str],
    batch_dim: int,
) -> None:
    from openvino._offline_transformations import apply_make_stateful_transformation

    input_output_map: dict[str, str] = {}
    for kv_name_pair in zip(key_value_input_names, key_value_output_names):
        input_output_map[kv_name_pair[0]] = kv_name_pair[1]

    apply_make_stateful_transformation(ov_model, input_output_map)
    build_state_initializer(ov_model, batch_dim)


def _names_match(port, prefix: str) -> bool:
    return any(prefix in name for name in port.get_names())


def patch_stateful(ov_model: ov.Model, num_logit_outputs: int) -> None:
    key_value_input_names = [inp.get_any_name() for inp in ov_model.inputs if _names_match(inp, PKV_INPUT_PREFIX)]
    key_value_output_names = [out.get_any_name() for out in ov_model.outputs if _names_match(out, PKV_OUTPUT_PREFIX)]
    not_kv_inputs = [inp for inp in ov_model.inputs if not _names_match(inp, PKV_INPUT_PREFIX)]
    if not key_value_input_names or not key_value_output_names:
        return

    fuse_cache_reorder(ov_model, not_kv_inputs, key_value_input_names, 0)
    make_stateful(
        ov_model,
        key_value_input_names,
        key_value_output_names,
        0,
    )
