"""Additional tests for SegmentTracker behaviour beyond the existing suite.

The basic happy path is covered by tests/test_streaming_policy.py — these
tests exercise the merge / overlap / finalization edges that drive most
of the still-uncovered branches in streaming_policy.py.
"""

from __future__ import annotations

from app.streaming_policy import STABILITY_SEC, SegmentTracker

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_normalize_text_lowers_and_collapses_whitespace() -> None:
    assert SegmentTracker._normalize_text("  Hello   WORLD\n") == "hello world"


def test_is_substring_match_respects_min_ratio() -> None:
    # "hello" inside "hello world" — ratio 5/11 ≈ 0.45 — fails 0.8 threshold.
    assert SegmentTracker._is_substring_match("hello", "hello world", min_ratio=0.8) is False
    # Same prefix but at 0.4 ratio threshold this passes.
    assert SegmentTracker._is_substring_match("hello", "hello world", min_ratio=0.4) is True
    # Empty inputs short-circuit to False.
    assert SegmentTracker._is_substring_match("", "abc", min_ratio=0.5) is False


def test_calc_overlap_ratio_disjoint_returns_zero() -> None:
    t = SegmentTracker()
    assert t._calc_overlap_ratio(0.0, 1.0, 2.0, 3.0) == 0.0


def test_calc_overlap_ratio_full_containment_returns_one() -> None:
    t = SegmentTracker()
    assert t._calc_overlap_ratio(1.0, 2.0, 0.0, 5.0) == 1.0


def test_calc_overlap_ratio_zero_duration_first_range() -> None:
    t = SegmentTracker()
    assert t._calc_overlap_ratio(1.0, 1.0, 0.0, 5.0) == 0.0


def test_overlaps_majority_uses_either_direction() -> None:
    t = SegmentTracker()
    # New range fully covers existing -> reverse ratio is 1.0 -> majority True.
    assert t._overlaps_majority(0.0, 10.0, 4.0, 5.0) is True
    # Disjoint -> False.
    assert t._overlaps_majority(0.0, 1.0, 5.0, 6.0) is False


# ---------------------------------------------------------------------------
# update_from_hypothesis — branches for new / merge / update / finalize
# ---------------------------------------------------------------------------


def test_update_creates_new_segment_when_no_match() -> None:
    tracker = SegmentTracker()
    hyps = [{"start": 0.0, "end": 1.0, "text": "Hello world"}]
    all_segs, newly_final = tracker.update_from_hypothesis(hyps, window_start=0.0, now_sec=0.5)
    assert len(all_segs) == 1
    assert all_segs[0].src == "Hello world"
    assert all_segs[0].final is False
    assert newly_final == []


def test_update_skips_too_short_segments() -> None:
    tracker = SegmentTracker()
    hyps = [{"start": 0.0, "end": 0.5, "text": "x"}]  # 1-char text
    all_segs, _ = tracker.update_from_hypothesis(hyps, window_start=0.0, now_sec=0.1)
    assert all_segs == []


def test_update_merges_continuation_for_same_speaker() -> None:
    """A short gap between same-speaker segments triggers _find_mergeable_segment."""
    tracker = SegmentTracker()
    tracker.update_from_hypothesis(
        [{"start": 0.0, "end": 1.0, "text": "Hello", "speaker_id": "SPEAKER_00"}],
        window_start=0.0,
        now_sec=0.5,
    )
    # Second hypothesis starts 0.3s after the first ended — merge window is 0.8s.
    all_segs, _ = tracker.update_from_hypothesis(
        [{"start": 1.3, "end": 2.0, "text": "world", "speaker_id": "SPEAKER_00"}],
        window_start=0.0,
        now_sec=1.5,
    )
    assert len(all_segs) == 1
    assert "Hello" in all_segs[0].src and "world" in all_segs[0].src


def test_update_does_not_merge_when_gap_too_large() -> None:
    """A 1.5s gap is beyond MAX_MERGE_GAP_SEC — separate segments emerge."""
    tracker = SegmentTracker()
    tracker.update_from_hypothesis(
        [{"start": 0.0, "end": 1.0, "text": "Hello", "speaker_id": "SPEAKER_00"}],
        window_start=0.0,
        now_sec=0.5,
    )
    all_segs, _ = tracker.update_from_hypothesis(
        [{"start": 2.5, "end": 3.0, "text": "world", "speaker_id": "SPEAKER_00"}],
        window_start=0.0,
        now_sec=2.7,
    )
    assert len(all_segs) == 2


def test_update_refines_existing_segment_when_text_grows() -> None:
    """Whisper extending its hypothesis on the same time range should overwrite text."""
    tracker = SegmentTracker()
    tracker.update_from_hypothesis(
        [{"start": 0.0, "end": 1.0, "text": "Hello", "speaker_id": "S0"}],
        window_start=0.0,
        now_sec=0.5,
    )
    all_segs, _ = tracker.update_from_hypothesis(
        [{"start": 0.0, "end": 1.5, "text": "Hello world", "speaker_id": "S0"}],
        window_start=0.0,
        now_sec=1.6,
    )
    assert len(all_segs) == 1
    assert all_segs[0].src == "Hello world"


def test_update_keeps_long_existing_when_new_hypothesis_is_much_shorter() -> None:
    """If a new hypothesis is <70% of the existing length, keep the old text."""
    tracker = SegmentTracker()
    tracker.update_from_hypothesis(
        [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "This is a longer sentence that we already saved",
                "speaker_id": "S0",
            }
        ],
        window_start=0.0,
        now_sec=2.5,
    )
    # Refinement is only 4 chars — way under 70% of original length.
    all_segs, _ = tracker.update_from_hypothesis(
        [{"start": 0.0, "end": 5.0, "text": "This", "speaker_id": "S0"}],
        window_start=0.0,
        now_sec=3.0,
    )
    assert len(all_segs) == 1
    assert "longer sentence" in all_segs[0].src  # original text preserved


def test_update_finalizes_segment_after_stability_window() -> None:
    """A segment whose abs_end is older than STABILITY_SEC should finalize."""
    tracker = SegmentTracker()
    tracker.update_from_hypothesis(
        [{"start": 0.0, "end": 1.0, "text": "Hello world"}],
        window_start=0.0,
        now_sec=0.5,
    )
    # Tick again well past the stability window — segment finalizes.
    _, newly_final = tracker.update_from_hypothesis(
        [],
        window_start=0.0,
        now_sec=1.0 + STABILITY_SEC + 1.0,
    )
    assert len(newly_final) == 1
    assert newly_final[0].final is True
    assert newly_final[0] in tracker.finalized_segments


def test_strip_finalized_prefix_removes_repeated_prefix() -> None:
    """Whisper sometimes re-emits a finalized sentence as a prefix of new text."""
    tracker = SegmentTracker()
    # Force-finalize a segment so the strip logic engages.
    tracker.update_from_hypothesis(
        [{"start": 0.0, "end": 1.0, "text": "Do you work in Germany"}],
        window_start=0.0,
        now_sec=0.5,
    )
    tracker.force_finalize_all()
    # Now feed a hypothesis that contains the just-finalized prefix.
    all_segs, _ = tracker.update_from_hypothesis(
        [
            {
                "start": 1.0,
                "end": 2.5,
                "text": "Do you work in Germany how long would you like to work in Germany",
            }
        ],
        window_start=0.0,
        now_sec=1.5,
    )
    new_live = [s for s in all_segs if not s.final]
    assert new_live, "Expected a new live segment after stripping the finalized prefix"
    assert "Do you work in Germany" not in new_live[0].src


def test_force_finalize_all_drains_live_segments() -> None:
    tracker = SegmentTracker()
    tracker.update_from_hypothesis(
        [{"start": 0.0, "end": 1.0, "text": "Hello world"}],
        window_start=0.0,
        now_sec=0.5,
    )
    tracker.update_from_hypothesis(
        [{"start": 1.5, "end": 2.5, "text": "Another bit"}],
        window_start=0.0,
        now_sec=2.0,
    )
    assert len(tracker.cumulative_segments) >= 1
    assert len(tracker.finalized_segments) == 0

    drained = tracker.force_finalize_all()

    assert len(drained) >= 1
    assert all(s.final for s in drained)
    assert tracker.cumulative_segments == []
    assert all(s in tracker.finalized_segments for s in drained)


def test_force_finalize_all_idempotent_when_already_final() -> None:
    tracker = SegmentTracker()
    tracker.update_from_hypothesis(
        [{"start": 0.0, "end": 1.0, "text": "Hello world"}],
        window_start=0.0,
        now_sec=0.5,
    )
    tracker.force_finalize_all()
    second = tracker.force_finalize_all()
    assert second == []  # nothing left to finalize


def test_finalized_end_time_reflects_latest_finalized_segment() -> None:
    tracker = SegmentTracker()
    tracker.update_from_hypothesis(
        [{"start": 0.0, "end": 1.0, "text": "First sentence"}],
        window_start=0.0,
        now_sec=0.5,
    )
    tracker.update_from_hypothesis(
        [{"start": 2.0, "end": 4.0, "text": "Second sentence"}],
        window_start=0.0,
        now_sec=2.5,
    )
    tracker.force_finalize_all()
    assert tracker.finalized_end_time == 4.0


def test_finalized_end_time_zero_when_empty() -> None:
    assert SegmentTracker().finalized_end_time == 0.0


# ---------------------------------------------------------------------------
# _is_compatible_segment edges
# ---------------------------------------------------------------------------


def test_is_compatible_same_speaker_id_always_compatible() -> None:
    from app.streaming_policy import Segment

    tracker = SegmentTracker()
    existing = Segment(
        id=0, abs_start=0, abs_end=1, src="x", src_lang="de", final=False, speaker_id="S0"
    )
    # Same speaker_id even when languages differ — compatible.
    assert tracker._is_compatible_segment(existing, speaker_id="S0", src_lang="en") is True


def test_is_compatible_different_speakers_with_same_role_compatible() -> None:
    from app.streaming_policy import Segment

    tracker = SegmentTracker()
    existing = Segment(
        id=0,
        abs_start=0,
        abs_end=1,
        src="x",
        src_lang="de",
        final=False,
        speaker_id="S0",
        speaker_role="german",
    )
    assert (
        tracker._is_compatible_segment(
            existing, speaker_id="S1", src_lang="de", speaker_role="german"
        )
        is True
    )


def test_is_compatible_rejects_german_vs_foreign() -> None:
    from app.streaming_policy import Segment

    tracker = SegmentTracker()
    existing = Segment(id=0, abs_start=0, abs_end=1, src="x", src_lang="de", final=False)
    assert tracker._is_compatible_segment(existing, speaker_id=None, src_lang="en") is False


def test_is_compatible_unknown_lang_passes() -> None:
    from app.streaming_policy import Segment

    tracker = SegmentTracker()
    existing = Segment(id=0, abs_start=0, abs_end=1, src="x", src_lang="unknown", final=False)
    assert tracker._is_compatible_segment(existing, speaker_id=None, src_lang="de") is True
