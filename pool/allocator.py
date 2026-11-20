"""Byte-granular GPU buffer-pool exercise.

The pool is deliberately a plain mutable ``dict`` so the tests can build
fixtures without any helper code from this module::

    {
        "capacity": int,        # bytes owned; blocks cover [0, capacity)
        "max_capacity": int,    # hard limit for grow(), in bytes
        "blocks": [block, ...], # sorted by offset, contiguous, no overlap
        "next_id": int,         # monotonic counter used to mint handles
    }

A block is::

    {
        "offset": int,          # byte offset from the start of the pool
        "size": int,            # extent in bytes (0 only for a zero-size alloc)
        "handle": str | None,   # None means free; otherwise an opaque handle
        "tag": object,          # request tag for live blocks, None when free
    }

All sizes, offsets, alignments, and statistics are in bytes. Live handles
are opaque strings, unique for the lifetime of a pool, and stable across
defragmentation: moving a block changes its offset, never its handle.
"""

import bisect
import collections
from collections.abc import Iterable, Mapping, Sequence


class PoolError(Exception):
    """Raised for invalid requests, handles, or pool state.

    All public operations report contract violations with this exception;
    no partial mutation is allowed before it is raised.
    """


def align_size(size: int, alignment: int) -> int:
    """Return the smallest multiple of *alignment* that is >= *size*.

    ``size`` and ``alignment`` are byte counts. ``size`` must be a
    non-negative integer. ``alignment`` must be a positive power of two
    (1, 2, 4, 8, ...); zero, negative values, and non-powers of two raise
    ``PoolError`` instead of silently rounding or dividing by zero.

    The returned value is exact byte arithmetic and is never smaller than
    ``size``, so an allocation of that size cannot spill into its neighbour.
    ``align_size(0, alignment) == 0`` for every valid alignment.
    """
    raise NotImplementedError


def allocate(pool: dict, request: Mapping) -> dict:
    """Allocate one buffer from *pool* and return a handle record.

    ``request`` has exactly the contract fields ``{"size", "align", "tag"}``.
    ``size`` is the number of requested bytes (>= 0), ``align`` is a positive
    power-of-two byte alignment, and ``tag`` is opaque caller metadata copied
    onto the live block.

    Allocation uses best fit: among free extents large enough to hold the
    aligned size, pick the smallest; break ties by the lowest offset. The
    returned ``size`` is the number of bytes actually reserved, i.e.
    ``align_size(request["size"], request["align"])``. ``offset`` is aligned
    and the extent fits entirely inside the pool. The opaque ``handle`` is a
    new string unique within the pool; a second request never reuses it.

    A zero-byte request succeeds without consuming capacity and returns a
    size-0 live extent with a unique handle. If no free extent fits, raise
    ``PoolError``; this function does not grow the pool implicitly (call
    :func:`grow` first). Invalid size/align values raise ``PoolError``.
    """
    raise NotImplementedError


def release(pool: dict, handle: str) -> dict:
    """Release the live allocation identified by *handle*.

    ``handle`` must be a currently live handle of *pool*. An unknown,
    already-released, duplicated, or foreign handle raises ``PoolError`` and
    leaves the pool unchanged. Success converts the allocation to free space
    and immediately coalesces it with every touching free neighbour, so the
    pool never retains two adjacent free extents.

    Returns ``{"handle": handle, "offset": int, "size": int}`` where offset
    and size describe the allocation that was released, before coalescing.
    Released handles are retired and are never handed out again.
    """
    raise NotImplementedError


def defragment(pool: dict, keep_order: bool) -> dict:
    """Compact live allocations toward offset 0 and coalesce free space.

    Every live allocation keeps its handle, size, and tag; only offsets of
    movable blocks may change. A caller holding a handle can still release
    it afterwards, so no external reference becomes dangling.

    With ``keep_order=True`` the current relative order of live blocks is
    preserved. With ``keep_order=False`` the implementation may reorder them
    to pack more tightly. In both cases the free bytes form one trailing
    extent (zero-size allocations occupy no bytes).

    Returns::

        {
            "moved": [
                {"handle": str, "old_offset": int, "new_offset": int},
                ...
            ],
            "free": int,   # free bytes after compaction
        }

    Blocks that did not move are omitted from ``moved``.
    """
    raise NotImplementedError


def stats(pool: dict) -> dict:
    """Return byte-weighted pool statistics.

    The result is::

        {
            "used": int,           # sum of live block sizes, bytes
            "free": int,           # sum of free extent sizes, bytes
            "fragmentation": float # 0.0 .. 1.0
        }

    Fragmentation is size-aware, not a block count: with ``F`` free bytes and
    ``L`` the largest single free extent, it is ``1 - L / F`` (0.0 when there
    is no free space). Thus one large free extent scores 0.0, while the same
    total free bytes split into many small extents scores closer to 1.0.
    Zero-size allocations contribute nothing to ``used``.
    """
    raise NotImplementedError


def grow(pool: dict, size: int) -> dict:
    """Increase the pool capacity by at least *size* additional bytes.

    ``size`` must be positive. The new capacity is returned as
    ``{"capacity": int, "added": int}`` and is also written back to
    ``pool["capacity"]``. Existing allocations keep their handles and
    offsets, and the newly added bytes become (or join) free space.

    If the requested growth would make capacity exceed
    ``pool["max_capacity"]``, raise ``PoolError`` before changing anything.
    Growth must honour the requested byte count rather than a fixed 1 MiB
    step, so a large request can succeed whenever the hard limit allows it.
    """
    raise NotImplementedError


def validate(pool: dict, handles: Iterable[str]) -> list[str]:
    """Return a list of human-readable problems; an empty list means valid.

    Checks both the supplied *handles* and the pool's live blocks for:

    * duplicate handles (the same handle supplied twice, or a handle used by
      two live blocks);
    * unknown or out-of-range handles (not live in this pool);
    * live extents whose offset/size falls outside ``[0, capacity)``;
    * live extents that overlap each other or a free extent.

    The input order is preserved when reporting duplicates/unknown handles.
    This function must not mutate the pool and must not raise ``PoolError``
    for a merely inconsistent pool; report the issue instead.
    """
    raise NotImplementedError


def pool_batch(pools: Sequence[dict], requests: Sequence[Mapping]) -> dict:
    """Allocate many requests across several pools as one transaction.

    Requests are processed in order. Each request is allocated from the
    first pool (in the supplied pool order) that can satisfy it, using the
    same best-fit rule as :func:`allocate`.

    The result is::

        {
            "allocations": [
                {
                    "pool": int,       # index into pools
                    "handle": str,
                    "offset": int,
                    "size": int,
                    "tag": object,
                },
                ...
            ]
        }

    The transaction is all-or-nothing. If any request cannot be satisfied,
    every allocation made earlier in this call is released, every pool is
    restored to its pre-call layout, and ``PoolError`` is raised. No handle
    from a failed batch may remain live, and handles returned by a successful
    batch are unique. An empty request sequence returns an empty allocation
    list.
    """
    raise NotImplementedError
