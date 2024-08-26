
from pathlib import Path

OPENPOSE_OV_PATH = Path("../model/openpose.xml")

def export_ov_model():
    import torch
    import openvino as ov
    from controlnet_aux import OpenposeDetector

    def cleanup_torchscript_cache():
        """
        Helper for removing cached model representation
        """
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
        torch.jit._state._clear_class_state()


    if not OPENPOSE_OV_PATH.exists():
        pose_estimator = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        with torch.no_grad():
            ov_model = ov.convert_model(
                pose_estimator.body_estimation.model,
                example_input=torch.zeros([1, 3, 184, 128]),
                input=[[1, 3, 184, 128]],
            )
            ov.save_model(ov_model, OPENPOSE_OV_PATH)
            del ov_model
            cleanup_torchscript_cache()
        print("OpenPose successfully converted to IR")
    else:
        print(f"OpenPose will be loaded from {OPENPOSE_OV_PATH}")


def validate_ov_pose_model():
    from collections import namedtuple
    import torch
    import openvino as ov
    from controlnet_aux import OpenposeDetector

    class OpenPoseOVModel:
        """Helper wrapper for OpenPose model inference"""

        def __init__(self, core, model_path, device="AUTO"):
            self.core = core
            self.model = core.read_model(model_path)
            self.compiled_model = core.compile_model(self.model, device)

        def __call__(self, input_tensor: torch.Tensor):
            """
            inference step

            Parameters:
            input_tensor (torch.Tensor): tensor with prerpcessed input image
            Returns:
            predicted keypoints heatmaps
            """
            h, w = input_tensor.shape[2:]
            input_shape = self.model.input(0).shape
            if h != input_shape[2] or w != input_shape[3]:
                self.reshape_model(h, w)
            results = self.compiled_model(input_tensor)
            return torch.from_numpy(results[self.compiled_model.output(0)]), torch.from_numpy(results[self.compiled_model.output(1)])

        def reshape_model(self, height: int, width: int):
            """
            helper method for reshaping model to fit input data

            Parameters:
            height (int): input tensor height
            width (int): input tensor width
            Returns:
            None
            """
            self.model.reshape({0: [1, 3, height, width]})
            self.compiled_model = self.core.compile_model(self.model)

        def parameters(self):
            Device = namedtuple("Device", ["device"])
            return [Device(torch.device("cpu"))]

    from PIL import Image
    import time

    img = Image.open("pose.png")

    core = ov.Core()
    ov_openpose = OpenPoseOVModel(core, OPENPOSE_OV_PATH, device="CPU")
    pose_estimator = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    pose_estimator.body_estimation.model = ov_openpose
    start_time = time.time()
    pose = pose_estimator(img)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"result is saved as pose.detect.png, it tooks: {execution_time} seconds to inference on CPU.")
    pose.save("pose.detect.png")
    

if __name__ == "__main__":
    export_ov_model()
    validate_ov_pose_model()