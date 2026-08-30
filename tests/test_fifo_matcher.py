"""Yon bazli FIFO eslestirici davranis testleri."""

from mcmot.fifo_matcher import FifoMatcher, TrackEvent


def event(t, camera, track, kind, direction="camA_to_camB"):
    return TrackEvent(t, camera, track, "car", direction, kind)


def test_fifo_matches_in_exit_order():
    matcher = FifoMatcher(delay_s=0.7, window_width_s=0.8)
    matcher.process(event(1.0, "camA", 10, "exit"))
    matcher.process(event(1.1, "camA", 11, "exit"))
    first = matcher.process(event(1.8, "camB", 20, "entry"))
    second = matcher.process(event(2.0, "camB", 21, "entry"))
    assert (first.source.track_id, first.target.track_id) == (10, 20)
    assert (second.source.track_id, second.target.track_id) == (11, 21)


def test_early_entry_does_not_consume_queue_head():
    matcher = FifoMatcher(delay_s=0.7, window_width_s=0.8)
    matcher.process(event(1.0, "camA", 10, "exit"))
    assert matcher.process(event(1.4, "camB", 20, "entry")) is None
    match = matcher.process(event(1.8, "camB", 21, "entry"))
    assert match.source.track_id == 10
    assert match.target.track_id == 21


def test_expired_head_is_removed_before_next_match():
    matcher = FifoMatcher(delay_s=0.7, window_width_s=0.8)
    matcher.process(event(1.0, "camA", 10, "exit"))
    matcher.process(event(2.0, "camA", 11, "exit"))
    match = matcher.process(event(2.8, "camB", 20, "entry"))
    assert [item.track_id for item in matcher.expired] == [10]
    assert match.source.track_id == 11


def test_directions_have_independent_queues():
    matcher = FifoMatcher(delay_s=0.7, window_width_s=0.8)
    matcher.process(event(1.0, "camA", 10, "exit"))
    matcher.process(event(1.0, "camB", 30, "exit", "camB_to_camA"))
    reverse = matcher.process(event(1.8, "camA", 40, "entry", "camB_to_camA"))
    assert reverse.source.track_id == 30
    assert matcher.queue_snapshot("camA_to_camB", 1.8)[0]["track_id"] == 10

