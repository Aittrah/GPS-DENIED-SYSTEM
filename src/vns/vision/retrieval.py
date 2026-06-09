"""Coarse retrieval: find top-K candidate reference images by descriptor voting."""

import logging
from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Tuple

import cv2
import numpy as np

from vns.database.reference_db import DatabaseEntry

logger = logging.getLogger("vns.vision.retrieval")

_FLANN_INDEX_LSH = 6


class RetrievalBackend(ABC):
    """Abstract interface for descriptor-based image retrieval.

    Subclass this to plug in BoW, VLAD, or other backends without
    changing callers.
    """

    @abstractmethod
    def build_index(self, entries: List[DatabaseEntry]) -> None:
        """Build the search index from database entries."""

    @abstractmethod
    def query(
        self, descriptors: np.ndarray, top_k: int,
    ) -> List[Tuple[str, int]]:
        """Return top-K (entry_id, vote_count) sorted by votes descending."""


class FlannLshBackend(RetrievalBackend):
    """FLANN with Locality-Sensitive Hashing for binary ORB descriptors."""

    def __init__(
        self,
        table_number: int = 6,
        key_size: int = 12,
        multi_probe_level: int = 1,
    ) -> None:
        self._table_number = table_number
        self._key_size = key_size
        self._multi_probe_level = multi_probe_level
        self._flann: cv2.FlannBasedMatcher = None  # type: ignore[assignment]
        self._descriptor_to_entry: List[str] = []
        self._index_built = False

    def build_index(self, entries: List[DatabaseEntry]) -> None:
        """Concatenate all entry descriptors and build one LSH index."""
        all_descriptors: List[np.ndarray] = []
        self._descriptor_to_entry = []

        for entry in entries:
            if entry.descriptors is not None and len(entry.descriptors) > 0:
                all_descriptors.append(entry.descriptors)
                self._descriptor_to_entry.extend(
                    [entry.id] * len(entry.descriptors)
                )

        if not all_descriptors:
            logger.warning("No descriptors found in database entries")
            self._index_built = False
            return

        stacked = np.vstack(all_descriptors).astype(np.uint8)

        index_params = dict(
            algorithm=_FLANN_INDEX_LSH,
            table_number=self._table_number,
            key_size=self._key_size,
            multi_probe_level=self._multi_probe_level,
        )
        search_params = dict(checks=50)

        self._flann = cv2.FlannBasedMatcher(index_params, search_params)
        self._flann.add([stacked])
        self._flann.train()
        self._index_built = True

        logger.info(
            "Built FLANN/LSH index: %d descriptors from %d entries",
            len(stacked),
            len(entries),
        )

    def query(
        self, descriptors: np.ndarray, top_k: int,
    ) -> List[Tuple[str, int]]:
        """Nearest-neighbor vote: each query descriptor votes for its closest entry."""
        if not self._index_built or self._flann is None:
            return []
        if descriptors is None or len(descriptors) == 0:
            return []

        query_des = descriptors.astype(np.uint8)
        try:
            matches = self._flann.knnMatch(query_des, k=1)
        except cv2.error as exc:
            logger.warning("FLANN query failed: %s", exc)
            return []

        votes: Counter = Counter()
        for m_list in matches:
            if m_list:
                idx = m_list[0].trainIdx
                if 0 <= idx < len(self._descriptor_to_entry):
                    votes[self._descriptor_to_entry[idx]] += 1

        return votes.most_common(top_k)


class RetrievalIndex:
    """Coarse retrieval from the reference database.

    Wraps a :class:`RetrievalBackend` (default: FLANN/LSH).  Swap the
    backend for BoW or VLAD without touching calling code.
    """

    def __init__(self, backend: RetrievalBackend = None) -> None:  # type: ignore[assignment]
        self._backend = backend or FlannLshBackend()

    def build_index(self, entries: List[DatabaseEntry]) -> None:
        self._backend.build_index(entries)

    def query(
        self, descriptors: np.ndarray, top_k: int = 5,
    ) -> List[Tuple[str, int]]:
        return self._backend.query(descriptors, top_k)
