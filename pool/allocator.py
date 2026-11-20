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

import copy
from collections.abc import Iterable, Mapping, Sequence


class PoolError(Exception):
    """Raised for invalid requests, handles, or pool state.

    All public operations report contract violations with this exception;
    no partial mutation is allowed before it is raised.
    """


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_alignment(alignment: int) -> int:
    """Return *alignment* if it is a positive power of two, else raise."""
    if not _is_int(alignment) or alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise PoolError(
            f"alignment must be a positive power of two, got {alignment!r}"
        )
    return alignment


def _mint_handle(pool: dict) -> str:
    handle = f"handle-{pool['next_id']}"
    pool["next_id"] += 1
    return handle


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
    if not _is_int(size) or size < 0:
        raise PoolError(f"size must be a non-negative integer, got {size!r}")
    _check_alignment(alignment)
    # Round UP: (size + alignment - 1) with the low bits cleared.
    return (size + alignment - 1) & ~(alignment - 1)


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
    size = request["size"]
    align = request["align"]
    tag = request["tag"]
    # Validates both size and alignment before any mutation.
    aligned = align_size(size, align)

    if aligned == 0:
        # Zero-byte request: consumes no capacity, always succeeds.
        handle = _mint_handle(pool)
        pool["blocks"].insert(
            0, {"offset": 0, "size": 0, "handle": handle, "tag": tag}
        )
        return {"handle": handle, "offset": 0, "size": 0, "tag": tag}

    # Best fit: smallest free extent that fits (including alignment
    # padding), ties broken by lowest offset.
    best = None  # (extent size, offset, index, padding)
    for index, block in enumerate(pool["blocks"]):
        if block["handle"] is not None:
            continue
        padding = (-block["offset"]) % align
        if padding + aligned > block["size"]:
            continue
        key = (block["size"], block["offset"])
        if best is None or key < best[0]:
            best = (key, index, padding)
    if best is None:
        raise PoolError(
            f"no free extent fits {aligned} bytes (alignment {align})"
        )

    _, index, padding = best
    block = pool["blocks"][index]
    offset = block["offset"] + padding
    remainder = block["size"] - padding - aligned
    handle = _mint_handle(pool)

    replacement = []
    if padding:
        replacement.append(
            {"offset": block["offset"], "size": padding, "handle": None, "tag": None}
        )
    replacement.append(
        {"offset": offset, "size": aligned, "handle": handle, "tag": tag}
    )
    if remainder:
        replacement.append(
            {
                "offset": offset + aligned,
                "size": remainder,
                "handle": None,
                "tag": None,
            }
        )
    pool["blocks"][index : index + 1] = replacement
    return {"handle": handle, "offset": offset, "size": aligned, "tag": tag}


def _coalesce(pool: dict) -> None:
    """Merge every run of touching free extents into one; drop empty ones."""
    merged = []
    for block in pool["blocks"]:
        if block["handle"] is None and block["size"] == 0:
            continue
        if (
            merged
            and merged[-1]["handle"] is None
            and block["handle"] is None
            and merged[-1]["offset"] + merged[-1]["size"] == block["offset"]
        ):
            merged[-1]["size"] += block["size"]
        else:
            merged.append(block)
    pool["blocks"] = merged


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
    if not isinstance(handle, str):
        raise PoolError(f"invalid handle {handle!r}")
    index = None
    for i, block in enumerate(pool["blocks"]):
        if block["handle"] is not None and block["handle"] == handle:
            index = i
            break
    if index is None:
        raise PoolError(f"unknown or already-released handle {handle!r}")

    block = pool["blocks"][index]
    released = {"handle": handle, "offset": block["offset"], "size": block["size"]}
    if block["size"] == 0:
        # Zero-size allocations occupy no bytes; just drop the record.
        del pool["blocks"][index]
    else:
        block["handle"] = None
        block["tag"] = None
    _coalesce(pool)
    return released


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
    live = [block for block in pool["blocks"] if block["handle"] is not None]
    if not keep_order:
        # Reordering is permitted; larger blocks first packs at least as
        # tightly and keeps the result deterministic.
        live.sort(key=lambda block: (-block["size"], block["offset"]))

    moved = []
    cursor = 0
    compacted = []
    for block in live:
        old_offset = block["offset"]
        block["offset"] = cursor
        if cursor != old_offset:
            moved.append(
                {
                    "handle": block["handle"],
                    "old_offset": old_offset,
                    "new_offset": cursor,
                }
            )
        cursor += block["size"]
        compacted.append(block)

    free = pool["capacity"] - cursor
    if free:
        compacted.append(
            {"offset": cursor, "size": free, "handle": None, "tag": None}
        )
    pool["blocks"] = compacted
    return {"moved": moved, "free": free}


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
    used = 0
    free_sizes = []
    for block in pool["blocks"]:
        if block["handle"] is None:
            if block["size"]:
                free_sizes.append(block["size"])
        else:
            used += block["size"]
    free = sum(free_sizes)
    fragmentation = 0.0 if free == 0 else 1.0 - max(free_sizes) / free
    return {"used": used, "free": free, "fragmentation": fragmentation}


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
    if not _is_int(size) or size <= 0:
        raise PoolError(f"grow size must be a positive integer, got {size!r}")
    new_capacity = pool["capacity"] + size
    if new_capacity > pool["max_capacity"]:
        raise PoolError(
            f"growing by {size} bytes would exceed max_capacity "
            f"{pool['max_capacity']}"
        )
    blocks = pool["blocks"]
    if blocks and blocks[-1]["handle"] is None:
        blocks[-1]["size"] += size
    else:
        blocks.append(
            {
                "offset": pool["capacity"],
                "size": size,
                "handle": None,
                "tag": None,
            }
        )
    pool["capacity"] = new_capacity
    return {"capacity": new_capacity, "added": size}


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
    issues = []
    capacity = pool["capacity"]
    blocks = pool["blocks"]
    live = [block for block in blocks if block["handle"] is not None]
    live_handles = {block["handle"] for block in live}

    # Handles used by more than one live block.
    seen_live = set()
    for block in live:
        handle = block["handle"]
        if handle in seen_live:
            issues.append(
                f"duplicate handle {handle!r} is used by more than one live block"
            )
        seen_live.add(handle)

    # Supplied handles, in input order.
    seen_supplied = set()
    for handle in handles:
        if handle in seen_supplied:
            issues.append(f"duplicate handle {handle!r} supplied more than once")
        elif handle not in live_handles:
            issues.append(f"unknown handle {handle!r} is not live in this pool")
        seen_supplied.add(handle)

    # Live extents outside [0, capacity).
    for block in live:
        end = block["offset"] + block["size"]
        if block["offset"] < 0 or end > capacity:
            issues.append(
                f"live extent {block['handle']!r} at "
                f"[{block['offset']}, {end}) falls outside [0, {capacity})"
            )

    # Overlaps between any extents (live/live or live/free).
    ordered = sorted(blocks, key=lambda block: block["offset"])
    previous_end = None
    previous_desc = None
    for block in ordered:
        if previous_end is not None and block["offset"] < previous_end:
            issues.append(
                f"extent at offset {block['offset']} overlaps extent "
                f"{previous_desc} ending at {previous_end}"
            )
        end = block["offset"] + block["size"]
        if previous_end is None or end > previous_end:
            previous_end = end
            previous_desc = (
                f"handle {block['handle']!r}"
                if block["handle"] is not None
                else "free extent"
            )
    return issues


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
    snapshots = [copy.deepcopy(pool) for pool in pools]
    allocations = []
    try:
        for request in requests:
            placed = None
            for index, pool in enumerate(pools):
                try:
                    record = allocate(pool, request)
                except PoolError:
                    continue
                placed = {
                    "pool": index,
                    "handle": record["handle"],
                    "offset": record["offset"],
                    "size": record["size"],
                    "tag": request["tag"],
                }
                break
            if placed is None:
                raise PoolError(
                    f"no pool can satisfy request {dict(request)!r}; "
                    "rolling back the whole batch"
                )
            allocations.append(placed)
    except PoolError:
        # All-or-nothing: restore every pool to its pre-call layout.
        for pool, snapshot in zip(pools, snapshots):
            pool.clear()
            pool.update(snapshot)
        raise
    return {"allocations": allocations}
