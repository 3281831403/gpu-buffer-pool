"""Initial red test suite for the GPU buffer-pool exercise.

Each test calls at least one public stub in ``pool.allocator`` and expresses
one user-visible contract. The suite must collect cleanly while all fifteen
tests fail with ``NotImplementedError``.
"""

import copy

import pytest

from pool.allocator import (
    PoolError,
    align_size,
    allocate,
    defragment,
    grow,
    pool_batch,
    release,
    stats,
    validate,
)


def make_pool(capacity=4096, max_capacity=1 << 20):
    """Build the documented plain-dict pool shape with one free extent."""
    blocks = []
    if capacity:
        blocks.append({"offset": 0, "size": capacity, "handle": None, "tag": None})
    return {
        "capacity": capacity,
        "max_capacity": max_capacity,
        "blocks": blocks,
        "next_id": 1,
    }


def make_layout(spans, max_capacity=1 << 20):
    """Build a contiguous pool from ``(size, handle, tag)`` spans."""
    blocks = []
    offset = 0
    for size, handle, tag in spans:
        blocks.append(
            {"offset": offset, "size": size, "handle": handle, "tag": tag}
        )
        offset += size
    return {
        "capacity": offset,
        "max_capacity": max_capacity,
        "blocks": blocks,
        "next_id": 100,
    }


def request(size, align=1, tag="buffer"):
    """Return a request using exactly the documented request fields."""
    return {"size": size, "align": align, "tag": tag}


# ---------------------------------------------------------------------------
# One regression test per documented defect.
# ---------------------------------------------------------------------------


def test_align_rounds_up_and_rejects_non_power_of_two():
    # Defect 1: rounding down lets a 5-byte request reserve only 4 bytes and
    # overwrite the next block.
    assert align_size(5, 4) == 8
    with pytest.raises(PoolError):
        align_size(16, 3)

    pool = make_pool(64)
    first = allocate(pool, request(5, 4, "first"))
    second = allocate(pool, request(3, 4, "second"))
    assert first["size"] == 8
    assert first["offset"] % 4 == 0
    assert second["offset"] >= first["offset"] + 8


def test_allocate_uses_best_fit_not_first_fit():
    # Defect 2: a 256-byte first free extent is chosen for an 80-byte request
    # even though a 96-byte extent fits better.
    pool = make_layout(
        [
            (64, None, None),
            (64, "sep-a", "a"),
            (256, None, None),
            (64, "sep-b", "b"),
            (96, None, None),
            (96, "sep-c", "c"),
        ]
    )
    result = allocate(pool, request(80, 8, "best-fit"))
    assert result["offset"] == 448
    assert result["size"] == 80
    assert result["handle"] not in {"sep-a", "sep-b", "sep-c"}


def test_release_coalesces_adjacent_free_blocks():
    # Defect 3: three touching free extents must become one whole-pool extent.
    pool = make_pool(1024)
    first = allocate(pool, request(64, 8, "a"))["handle"]
    middle = allocate(pool, request(64, 8, "b"))["handle"]
    last = allocate(pool, request(64, 8, "c"))["handle"]

    release(pool, first)
    release(pool, last)
    release(pool, middle)

    free = [block for block in pool["blocks"] if block["handle"] is None]
    assert len(free) == 1
    assert free[0]["offset"] == 0
    assert free[0]["size"] == 1024


def test_defragment_keeps_handles_valid():
    # Defect 4: compaction must not invalidate handles or strand references.
    pool = make_pool(512)
    first = allocate(pool, request(64, 8, "a"))["handle"]
    middle = allocate(pool, request(64, 8, "b"))["handle"]
    last = allocate(pool, request(64, 8, "c"))["handle"]
    release(pool, middle)

    live_before = {
        block["handle"] for block in pool["blocks"] if block["handle"] is not None
    }
    result = defragment(pool, keep_order=False)
    live_after = {
        block["handle"] for block in pool["blocks"] if block["handle"] is not None
    }

    assert live_after == live_before
    assert {move["handle"] for move in result["moved"]} <= live_after
    for handle in sorted(live_after):
        release(pool, handle)

    free = [block for block in pool["blocks"] if block["handle"] is None]
    assert len(free) == 1
    assert free[0]["size"] == 512
    assert first in live_after and last in live_after


def test_stats_fragmentation_weights_free_extent_sizes():
    # Defect 5: block-count fragmentation ignores that most free bytes are in
    # one 816-byte extent.
    pool = make_layout(
        [
            (64, None, None),
            (64, "used-a", "a"),
            (16, None, None),
            (64, "used-b", "b"),
            (816, None, None),
        ]
    )
    result = stats(pool)
    assert result["used"] == 128
    assert result["free"] == 896
    assert result["fragmentation"] == pytest.approx(5 / 56)
    assert 0.0 <= result["fragmentation"] <= 1.0


def test_grow_supports_large_request_within_limit():
    # Defect 6: a fixed 1 MiB growth step cannot satisfy an 8 MiB increase,
    # although the hard limit allows it.
    max_capacity = 16 * 1024 * 1024
    pool = make_pool(1024, max_capacity=max_capacity)
    existing = allocate(pool, request(128, 64, "before"))["handle"]

    result = grow(pool, 8 * 1024 * 1024)
    assert result["capacity"] == pool["capacity"]
    assert result["added"] >= 8 * 1024 * 1024
    assert 1024 + 8 * 1024 * 1024 <= pool["capacity"] <= max_capacity

    huge = allocate(pool, request(4 * 1024 * 1024, 4096, "huge"))
    assert huge["size"] == 4 * 1024 * 1024
    assert huge["offset"] % 4096 == 0

    with pytest.raises(PoolError):
        grow(pool, 64 * 1024 * 1024)
    release(pool, existing)


def test_allocate_returns_distinct_handles():
    # Defect 7: two identical requests must never receive the same handle.
    pool = make_pool(1024)
    first = allocate(pool, request(64, 8, "a"))
    second = allocate(pool, request(64, 8, "b"))
    assert first["handle"] != second["handle"]
    assert first["offset"] != second["offset"]

    release(pool, first["handle"])
    third = allocate(pool, request(64, 8, "c"))
    assert third["handle"] not in {first["handle"], second["handle"]}


def test_release_rejects_foreign_or_stale_handle():
    # Defect 8: an unknown/out-of-range handle must not free somebody's block.
    pool = make_pool(256)
    live = allocate(pool, request(64, 8, "live"))["handle"]

    issues = validate(pool, [live, live, "other-pool:never-issued"])
    assert issues
    with pytest.raises(PoolError):
        release(pool, "other-pool:never-issued")

    release(pool, live)
    free = [block for block in pool["blocks"] if block["handle"] is None]
    assert len(free) == 1
    assert free[0]["size"] == 256


def test_pool_batch_rolls_back_partial_allocations():
    # Defect 9: only two of three 64-byte requests fit; a failure must leave
    # no half-allocated state behind.
    pool = make_pool(128, max_capacity=128)
    requests = [request(64, 8, "a"), request(64, 8, "b"), request(64, 8, "c")]
    snapshot = copy.deepcopy(pool)

    with pytest.raises(PoolError):
        pool_batch([pool], requests)

    assert pool == snapshot
    assert all(block["handle"] is None for block in pool["blocks"])


def test_align_rejects_zero_alignment():
    # Defect 10: alignment=0 must be rejected instead of dividing by zero.
    for alignment in (0, -4, 3, 6):
        with pytest.raises(PoolError):
            align_size(16, alignment)
    assert align_size(5, 4) == 8


# ---------------------------------------------------------------------------
# Five additional contract tests.
# ---------------------------------------------------------------------------


def test_allocate_zero_size_returns_zero_size_block():
    pool = make_pool(256)
    result = allocate(pool, request(0, 8, "zero"))
    assert result["size"] == 0
    assert result["offset"] % 8 == 0

    report = stats(pool)
    assert report["used"] == 0
    assert report["free"] == 256

    release(pool, result["handle"])
    assert stats(pool)["free"] == 256


def test_allocate_rejects_zero_alignment():
    pool = make_pool(256)
    with pytest.raises(PoolError):
        allocate(pool, request(16, 0, "bad-align"))
    assert pool["blocks"][0]["handle"] is None
    assert stats(pool)["used"] == 0


def test_allocate_from_empty_pool_raises():
    pool = make_pool(0, max_capacity=1024)
    assert pool["blocks"] == []
    with pytest.raises(PoolError):
        allocate(pool, request(1, 1, "empty"))


def test_release_same_handle_twice_raises():
    pool = make_pool(256)
    handle = allocate(pool, request(64, 8, "once"))["handle"]
    release(pool, handle)
    with pytest.raises(PoolError):
        release(pool, handle)


def test_pool_batch_success_has_unique_handles():
    pool = make_pool(2048)
    requests = [
        request(64, 8, "a"),
        request(128, 16, "b"),
        request(32, 4, "c"),
    ]
    result = pool_batch([pool], requests)
    allocations = result["allocations"]

    assert len(allocations) == 3
    assert [item["pool"] for item in allocations] == [0, 0, 0]
    assert [item["tag"] for item in allocations] == ["a", "b", "c"]
    handles = [item["handle"] for item in allocations]
    assert len(set(handles)) == 3
    for item, alignment in zip(allocations, (8, 16, 4)):
        assert item["size"] >= 1
        assert item["offset"] % alignment == 0
