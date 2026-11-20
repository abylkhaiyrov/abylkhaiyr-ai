from __future__ import annotations

from torch import Tensor, empty


class LayerKVCache:
    def __init__(self, capacity: int = 0) -> None:
        self.capacity = capacity
        self.keys: Tensor | None = None
        self.values: Tensor | None = None
        self._length = 0

    @property
    def length(self) -> int:
        return self._length

    def _reserve(self, template: Tensor, needed: int) -> None:
        batch, heads, _, dim = template.shape
        if self.keys is not None and self.keys.shape[2] >= needed:
            return
        capacity = max(
            self.capacity, needed, 2 * self.keys.shape[2] if self.keys is not None else 0
        )
        keys = empty(batch, heads, capacity, dim, dtype=template.dtype, device=template.device)
        values = empty(keys.shape, dtype=keys.dtype, device=keys.device)
        if self.keys is not None and self.values is not None:
            keys[:, :, : self._length] = self.keys[:, :, : self._length]
            values[:, :, : self._length] = self.values[:, :, : self._length]
        self.keys, self.values = keys, values

    def update(self, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        end = self._length + keys.shape[2]
        self._reserve(keys, end)
        assert self.keys is not None and self.values is not None
        self.keys[:, :, self._length : end] = keys
        self.values[:, :, self._length : end] = values
        self._length = end
        return self.keys[:, :, :end], self.values[:, :, :end]

    def truncate(self, length: int) -> None:
        self._length = min(self._length, length)

    def reorder(self, indices: Tensor) -> None:
        if self.keys is not None and self.values is not None:
            self.keys = self.keys.index_select(0, indices)
            self.values = self.values.index_select(0, indices)


class KVCache:
    def __init__(self, num_layers: int, capacity: int = 0) -> None:
        self.layers = [LayerKVCache(capacity) for _ in range(num_layers)]

    @property
    def length(self) -> int:
        return self.layers[0].length

    def truncate(self, length: int) -> None:
        for layer in self.layers:
            layer.truncate(length)

    def reorder(self, indices: Tensor) -> None:
        for layer in self.layers:
            layer.reorder(indices)
