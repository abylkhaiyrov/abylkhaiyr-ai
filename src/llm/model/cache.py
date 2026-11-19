from __future__ import annotations

import torch
from torch import Tensor


class LayerKVCache:
    def __init__(self) -> None:
        self.keys: Tensor | None = None
        self.values: Tensor | None = None

    @property
    def length(self) -> int:
        return 0 if self.keys is None else self.keys.shape[2]

    def update(self, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        if self.keys is None or self.values is None:
            self.keys, self.values = keys, values
        else:
            self.keys = torch.cat((self.keys, keys), dim=2)
            self.values = torch.cat((self.values, values), dim=2)
        return self.keys, self.values

    def truncate(self, length: int) -> None:
        if self.keys is not None and self.values is not None:
            self.keys = self.keys[:, :, :length]
            self.values = self.values[:, :, :length]

    def reorder(self, indices: Tensor) -> None:
        if self.keys is not None and self.values is not None:
            self.keys = self.keys.index_select(0, indices)
            self.values = self.values.index_select(0, indices)


class KVCache:
    def __init__(self, num_layers: int) -> None:
        self.layers = [LayerKVCache() for _ in range(num_layers)]

    @property
    def length(self) -> int:
        return self.layers[0].length

    def truncate(self, length: int) -> None:
        for layer in self.layers:
            layer.truncate(length)

    def reorder(self, indices: Tensor) -> None:
        for layer in self.layers:
            layer.reorder(indices)
