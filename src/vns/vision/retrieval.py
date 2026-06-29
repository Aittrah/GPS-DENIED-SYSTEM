"""Coarse retrieval: find top-K candidate reference images by descriptor voting."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from collections import Counter
from typing import List, Tuple

import cv2
import numpy as np

from vns.database.reference_db import DatabaseEntry, ReferenceDatabase
from vns.vision.bovw_retrieval import BoVWIndex

logger = logging.getLogger("vns.vision.retrieval")

_FLANN_INDEX_LSH = 6
RetrievalScore = int | float
RetrievalResults = List[Tuple[str, RetrievalScore]]


def _normalize_retrieval_mode(value: object) -> str:
    if value is None:
        return "flann"

    normalized = str(value).strip().lower()
    if normalized in {"flann", "flann_lsh"}:
        return "flann"
    if normalized == "bovw":
        return "bovw"
    raise ValueError(
        "Unsupported retrieval mode. Expected one of: 'flann', 'flann_lsh', or 'bovw'."
    )


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
        self,
        descriptors: np.ndarray | None,
        top_k: int,
    ) -> RetrievalResults:
        """Return top-K candidates ordered best-first."""


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
        self,
        descriptors: np.ndarray | None,
        top_k: int,
    ) -> RetrievalResults:
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

        return [(entry_id, float(vote_count)) for entry_id, vote_count in votes.most_common(top_k)]


class BoVWRetrievalBackend(RetrievalBackend):
    """BoVW histogram retrieval reconstructed from an in-memory reference database."""

    def __init__(self, index: BoVWIndex) -> None:
        self._index = index

    @classmethod
    def from_database(cls, database: ReferenceDatabase) -> "BoVWRetrievalBackend":
        if database.vocabulary is None:
            raise ValueError(
                "BoVW retrieval selected but the reference database contains no "
                "BoVW vocabulary."
            )

        ids = list(database.entries.keys())
        if not ids:
            raise ValueError(
                "BoVW retrieval selected but the reference database contains no entries."
            )

        histograms = []
        for entry_id in ids:
            histogram = database.entries[entry_id].bovw_histogram
            if histogram is None:
                raise ValueError(
                    "BoVW retrieval selected but entry "
                    f"'{entry_id}' is missing its BoVW histogram."
                )
            histograms.append(np.asarray(histogram, dtype=np.float32))

        index = BoVWIndex(
            database.vocabulary,
            ids,
            np.vstack(histograms).astype(np.float32),
            metric=database.bovw_metric,
        )
        return cls(index)

    def build_index(self, entries: List[DatabaseEntry]) -> None:
        """BoVW indexes are reconstructed up front from the loaded database."""
        if len(entries) != len(self._index.ids):
            logger.warning(
                "BoVW index built for %d entries but received %d entries during "
                "initialization.",
                len(self._index.ids),
                len(entries),
            )

    def query(
        self,
        descriptors: np.ndarray | None,
        top_k: int,
    ) -> RetrievalResults:
        if descriptors is None or len(descriptors) == 0:
            return []
        return self._index.query(descriptors, k=top_k)


class RetrievalIndex:
    """Coarse retrieval from the reference database.

    Wraps a :class:`RetrievalBackend` (default: FLANN/LSH).  Swap the
    backend for BoW or VLAD without touching calling code.
    """

    def __init__(
        self,
        backend: RetrievalBackend = None,  # type: ignore[assignment]
        *,
        backend_name: str = "flann",
    ) -> None:
        self._backend = backend or FlannLshBackend()
        self._backend_name = backend_name

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object] | None,
        database: ReferenceDatabase,
    ) -> "RetrievalIndex":
        retrieval_cfg = dict(config or {})
        mode = _normalize_retrieval_mode(
            retrieval_cfg.get("mode", retrieval_cfg.get("backend"))
        )
        bovw_cfg = retrieval_cfg.get("bovw")
        normalized_bovw_cfg = bovw_cfg if isinstance(bovw_cfg, Mapping) else {}
        fallback_to_flann = bool(
            normalized_bovw_cfg.get("fallback_to_flann", False)
        )

        if mode == "bovw":
            try:
                if not bool(normalized_bovw_cfg.get("enabled", True)):
                    raise ValueError(
                        "BoVW retrieval selected but retrieval.bovw.enabled is false."
                    )
                index = cls(
                    backend=BoVWRetrievalBackend.from_database(database),
                    backend_name="bovw",
                )
            except ValueError:
                if not fallback_to_flann:
                    raise
                logger.warning(
                    "BoVW retrieval unavailable; falling back to FLANN/LSH.",
                    exc_info=True,
                )
                index = cls(backend=FlannLshBackend(), backend_name="flann")
        else:
            index = cls(backend=FlannLshBackend(), backend_name="flann")

        index.build_index(list(database.entries.values()))
        return index

    def build_index(self, entries: List[DatabaseEntry]) -> None:
        self._backend.build_index(entries)

    def query(
        self,
        descriptors: np.ndarray | None,
        top_k: int = 5,
    ) -> RetrievalResults:
        return self._backend.query(descriptors, top_k)
