#!/usr/bin/env python3
"""
Bag of Visual Words (BoVW) coarse-retrieval index for the VNS.

This is the *query-side* (online) counterpart to the offline vocabulary builder
in ``simulation/scripts/build_reference_database.py``. It reconstructs the
visual-word vocabulary and a NearestNeighbors index from a ``.vnsdb`` file so a
live query frame's ORB descriptors can be turned into a histogram and matched
against all reference images in roughly O(1) (NN lookup) instead of O(N) brute
force descriptor matching.

CRITICAL: the descriptor -> histogram logic lives in ONE place
(:func:`compute_bovw_histogram`) and is imported by both the offline builder and
this online index. They must never diverge: a different assignment or
normalization here would silently break retrieval (query histograms would live
in a different space than the stored reference histograms).
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from vns.database.reference_db import ReferenceDatabase

try:
    from sklearn.neighbors import NearestNeighbors
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "scikit-learn is required for BoVW retrieval. "
        "Install with: pip3 install scikit-learn"
    ) from exc

logger = logging.getLogger(__name__)


def compute_bovw_histogram(
    descriptors: Optional[np.ndarray],
    vocabulary: Optional[np.ndarray],
    k: int,
) -> np.ndarray:
    """Build an L2-normalized BoVW histogram for a set of ORB descriptors.

    This is the single shared assign+normalize routine used by BOTH the offline
    database builder and the online query path, so the two can never diverge.

    Args:
        descriptors: (M, 32) uint8 ORB descriptors, or empty / None.
        vocabulary: (K, 32) float32 k-means cluster centers (visual words).
        k: number of visual words (== ``vocabulary.shape[0]``).

    Returns:
        (k,) float32 histogram, L2-normalized. All zeros when there are no
        descriptors or no vocabulary.
    """
    hist = np.zeros(k, dtype=np.float32)
    if descriptors is None or len(descriptors) == 0:
        return hist
    if vocabulary is None or len(vocabulary) == 0:
        return hist

    # ORB descriptors are BINARY (uint8, Hamming space), but the vocabulary
    # centers were trained with (Euclidean) MiniBatchKMeans. We therefore assign
    # each descriptor to its nearest center by Euclidean distance, matching how
    # the vocabulary was built. This binary-on-Euclidean assignment is a known,
    # accepted approximation chosen for speed and offline/online consistency.
    desc = np.asarray(descriptors, dtype=np.float32)
    vocab = np.asarray(vocabulary, dtype=np.float32)

    # Nearest center by Euclidean distance, computed via a single BLAS gemm to
    # hit the sub-millisecond query budget. Since
    #   ||d - c||^2 = ||d||^2 + ||c||^2 - 2 d.c,
    # and ||d||^2 is constant across centers for a given descriptor, the nearest
    # center is argmax_c (2 d.c - ||c||^2). This is exactly equivalent to a
    # Euclidean argmin (verified identical to sklearn pairwise_distances_argmin).
    center_sq = np.einsum("ij,ij->i", vocab, vocab)
    scores = desc @ vocab.T
    scores *= 2.0
    scores -= center_sq
    words = np.argmax(scores, axis=1)

    counts = np.bincount(words, minlength=k).astype(np.float32)
    norm = np.linalg.norm(counts)
    if norm > 0:
        counts /= norm
    return counts


class BoVWIndex:
    """Query-time BoVW index over reference-image histograms.

    Reconstructed from a ``.vnsdb`` via :meth:`load`. ``k`` and ``metric`` are
    taken from the persisted database (they must match the trained vocabulary);
    only ``top_k`` is a caller/config concern.
    """

    def __init__(
        self,
        vocabulary: np.ndarray,
        ids: List[str],
        histograms: np.ndarray,
        metric: str = "cosine",
    ) -> None:
        self.vocabulary = np.asarray(vocabulary, dtype=np.float32)
        self.k = int(self.vocabulary.shape[0])
        self.ids = list(ids)
        self.histograms = np.asarray(histograms, dtype=np.float32)
        self.metric = metric
        # A sklearn NearestNeighbors index is fit here per the database spec (and
        # is rebuilt rather than pickled, so it is robust to scikit-learn version
        # differences). The hot query() path below uses an equivalent vectorized
        # numpy lookup instead, because kneighbors() carries ~1ms of fixed
        # per-call overhead that would blow the sub-millisecond query budget.
        # _nn remains available via query_sklearn() for parity / very large sets.
        self._nn = NearestNeighbors(metric=metric)
        self._nn.fit(self.histograms)

    @classmethod
    def load(cls, db_path: str) -> "BoVWIndex":
        """Reconstruct the vocabulary + NN index from a ``.vnsdb`` file."""
        db = ReferenceDatabase.load(db_path)
        if db.vocabulary is None:
            raise ValueError(
                f"Database '{db_path}' contains no BoVW index. Rebuild it with: "
                "build_reference_database.py --build-vocab"
            )

        ids = list(db.entries.keys())
        histograms = []
        for entry_id in ids:
            histogram = db.entries[entry_id].bovw_histogram
            if histogram is None:
                raise ValueError(
                    f"Database '{db_path}' has an incomplete BoVW index: "
                    f"entry '{entry_id}' is missing its histogram."
                )
            histograms.append(histogram)

        if not histograms:
            raise ValueError(
                f"Database '{db_path}' contains no BoVW histograms."
            )

        index = cls(
            vocabulary=db.vocabulary,
            ids=ids,
            histograms=np.vstack(histograms).astype(np.float32),
            metric=db.bovw_metric,
        )
        logger.info(
            "Loaded BoVW index: K=%d words, %d reference histograms, metric=%s",
            index.k,
            len(index.ids),
            index.metric,
        )
        return index

    def compute_histogram(self, descriptors: np.ndarray) -> np.ndarray:
        """Histogram for a query frame's descriptors (shared offline/online logic)."""
        return compute_bovw_histogram(descriptors, self.vocabulary, self.k)

    def _distances(self, hist: np.ndarray) -> np.ndarray:
        """Distance from ``hist`` to every reference histogram, matching ``self.metric``.

        Vectorized to keep query() sub-millisecond. Reference histograms and the
        query histogram are L2-normalized, so cosine distance == 1 - dot product;
        l2 is the Euclidean norm. These match scikit-learn's metric semantics, so
        the returned distances agree with the fitted NearestNeighbors index.
        """
        if self.metric == "cosine":
            return 1.0 - self.histograms @ hist
        if self.metric == "l2":
            diff = self.histograms - hist
            return np.sqrt(np.einsum("ij,ij->i", diff, diff))
        raise ValueError(f"Unsupported metric: {self.metric!r} (expected 'cosine' or 'l2')")

    def query(
        self,
        descriptors: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Return the top-``k`` candidate reference images for a query frame.

        Uses a vectorized numpy lookup (sub-millisecond); see :meth:`query_sklearn`
        for the equivalent scikit-learn NearestNeighbors path.

        Args:
            descriptors: (M, 32) ORB descriptors of the query frame.
            k: number of candidates to return.

        Returns:
            List of ``(ref_id, distance)`` sorted nearest-first.
        """
        hist = self.compute_histogram(descriptors)
        k = min(k, len(self.ids))
        distances = self._distances(hist)
        # argpartition for the k smallest, then sort just those k.
        top = np.argpartition(distances, k - 1)[:k] if k < len(self.ids) else np.arange(len(self.ids))
        top = top[np.argsort(distances[top])]
        return [(self.ids[idx], float(distances[idx])) for idx in top]

    def query_sklearn(
        self,
        descriptors: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Equivalent top-``k`` lookup via the fitted sklearn NearestNeighbors index."""
        hist = self.compute_histogram(descriptors)
        k = min(k, len(self.ids))
        distances, indices = self._nn.kneighbors(hist.reshape(1, -1), n_neighbors=k)
        return [(self.ids[idx], float(dist)) for dist, idx in zip(distances[0], indices[0])]
