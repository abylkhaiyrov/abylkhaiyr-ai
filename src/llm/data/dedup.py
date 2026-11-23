from __future__ import annotations

import hashlib

import numpy as np

MERSENNE_PRIME = (1 << 61) - 1
HASH_MASK = (1 << 32) - 1
HASH_BYTES = 8


def stable_hash(text: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(text.encode("utf-8"), digest_size=HASH_BYTES).digest(), "big"
    )


class ExactDeduplicator:
    def __init__(self) -> None:
        self.seen: set[int] = set()

    def is_duplicate(self, text: str) -> bool:
        key = stable_hash(text)
        if key in self.seen:
            return True
        self.seen.add(key)
        return False


class MinHashDeduplicator:
    def __init__(self, permutations: int, threshold: float, shingle_size: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.multipliers = rng.integers(1, HASH_MASK, size=permutations, dtype=np.uint64)
        self.offsets = rng.integers(0, HASH_MASK, size=permutations, dtype=np.uint64)
        self.shingle_size = shingle_size
        self.bands, self.rows = self._band_layout(permutations, threshold)
        self.tables: list[dict[bytes, list[np.ndarray]]] = [{} for _ in range(self.bands)]
        self.threshold = threshold

    @staticmethod
    def _band_layout(permutations: int, threshold: float) -> tuple[int, int]:
        best, best_error = (1, permutations), float("inf")
        for rows in range(1, permutations + 1):
            if permutations % rows:
                continue
            bands = permutations // rows
            midpoint = (1.0 / bands) ** (1.0 / rows)
            if abs(midpoint - threshold) < best_error:
                best, best_error = (bands, rows), abs(midpoint - threshold)
        return best

    def signature(self, text: str) -> np.ndarray:
        words = text.lower().split()
        size = min(self.shingle_size, max(len(words), 1))
        shingles = {" ".join(words[i : i + size]) for i in range(max(len(words) - size + 1, 1))}
        hashes = np.fromiter((stable_hash(s) & HASH_MASK for s in shingles), dtype=np.uint64)
        mixed = (
            hashes[:, None] * self.multipliers[None, :] + self.offsets[None, :]
        ) % MERSENNE_PRIME
        return (mixed & HASH_MASK).min(axis=0)

    def _keys(self, signature: np.ndarray) -> list[bytes]:
        return [
            signature[band * self.rows : (band + 1) * self.rows].tobytes()
            for band in range(self.bands)
        ]

    def _matches(self, signature: np.ndarray, keys: list[bytes]) -> bool:
        return any(
            float(np.mean(candidate == signature)) >= self.threshold
            for table, key in zip(self.tables, keys, strict=True)
            for candidate in table.get(key, ())
        )

    def matches(self, text: str) -> bool:
        signature = self.signature(text)
        return self._matches(signature, self._keys(signature))

    def is_duplicate(self, text: str) -> bool:
        signature = self.signature(text)
        keys = self._keys(signature)
        if self._matches(signature, keys):
            return True
        for table, key in zip(self.tables, keys, strict=True):
            table.setdefault(key, []).append(signature)
        return False
