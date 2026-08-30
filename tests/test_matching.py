"""Faz 3 gecikme kalibrasyonu ve bire-bir atama testleri."""

import importlib.util
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = PROJECT_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


candidates = load_script("03_build_match_candidates.py")
assignment = load_script("03_assign_matches.py")


def test_delay_shifts_time_score_center():
    assert math.isclose(candidates.time_similarity(1.1, delay_s=0.7, window_width_s=0.8), 1.0)
    assert abs(candidates.time_similarity(1.5, delay_s=0.7, window_width_s=0.8)) < 1e-9
    assert math.isclose(candidates.time_similarity(1.8, delay_s=1.4, window_width_s=0.8), 1.0)


def test_duration_and_delay_gate_candidates():
    row_a = {
        "movement_label": "camA_to_camB", "track_id": "1", "class": "car",
        "exit_timestamp": "10.0", "enter_timestamp": "9.0",
    }
    row_b = {
        "movement_label": "camA_to_camB", "track_id": "2", "class": "car",
        "exit_timestamp": "12.0", "enter_timestamp": "11.1",
    }
    result = candidates.build_candidates(
        [row_a], [row_b], delay_s=0.7, window_width_s=0.8,
        clock_offset=0.0, duration_s=30.0, feats={},
    )
    assert len(result) == 1
    assert result[0]["delta_t"] == 1.1
    assert result[0]["time_score"] == 1.0

    assert candidates.build_candidates(
        [row_a], [row_b], delay_s=1.2, window_width_s=0.8,
        clock_offset=0.0, duration_s=30.0, feats={},
    ) == []


def _candidate(src, dst, score):
    return {
        "movement_label": "camA_to_camB",
        "src_camera": "camA", "src_track": str(src), "src_class": "car",
        "dst_camera": "camB", "dst_track": str(dst), "dst_class": "car",
        "src_exit_t": float(src), "dst_enter_t": float(dst),
        "delta_t": 1.1, "time_score": score, "score": score,
        "class_match": 1, "color_sim": None,
    }


def test_optimal_assignment_beats_greedy_choice():
    rows = [
        _candidate(1, 1, 0.90),
        _candidate(1, 2, 0.80),
        _candidate(2, 1, 0.85),
        _candidate(2, 2, 0.10),
    ]
    chosen = assignment.optimal_pairs(rows, min_score=0.0)
    pairs = {(row["src_track"], row["dst_track"]) for row in chosen}
    assert pairs == {("1", "2"), ("2", "1")}


def test_confidence_rejects_basic_inconsistency():
    row = _candidate(1, 2, 0.90)
    row["class_match"] = 0
    assert assignment.confidence_level(row, margin=0.90) == "low"

    row["class_match"] = 1
    row["size_sim"] = 0.30
    assert assignment.confidence_level(row, margin=0.90) == "low"
