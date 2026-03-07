import torch
from typing import Any


class FlatCache:
    def __init__(self, kv_tensors: list[torch.Tensor]) -> None:
        num_layers = len(kv_tensors) // 2
        self._keys = [kv_tensors[i * 2] for i in range(num_layers)]
        self._values = [kv_tensors[i * 2 + 1] for i in range(num_layers)]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = torch.cat([self._keys[layer_idx], key_states], dim=-2)
        v = torch.cat([self._values[layer_idx], value_states], dim=-2)
        self._keys[layer_idx] = k
        self._values[layer_idx] = v
        return k, v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._keys[layer_idx].shape[-2]

    def get_mask_sizes(
        self,
        cache_position: torch.Tensor,
        layer_idx: int,
    ) -> tuple[int, int]:
        kv_length = self._keys[layer_idx].shape[-2] + cache_position.shape[0]
        return kv_length, 0

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self):
        for k, v in zip(self._keys, self._values):
            yield k, v
