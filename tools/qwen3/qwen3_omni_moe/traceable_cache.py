import torch
from typing import Any


class TraceableCache:
    def __init__(
        self,
        past_key_values: list[list[torch.Tensor]],
    ) -> None:
        self._keys = [kv[0] for kv in past_key_values]
        self._values = [kv[1] for kv in past_key_values]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = torch.cat([self._keys[layer_idx], key_states], dim=2)
        v = torch.cat([self._values[layer_idx], value_states], dim=2)
        self._keys[layer_idx] = k
        self._values[layer_idx] = v
        return k, v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._keys[layer_idx].shape[2]

    def get_mask_sizes(
        self,
        cache_position: torch.Tensor,
        layer_idx: int,
    ) -> tuple[int, int]:
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length(layer_idx) + query_length
        return kv_length, 0

    def __iter__(self):
        for k, v in zip(self._keys, self._values):
            yield k, v

    def __len__(self) -> int:
        return len(self._keys)
