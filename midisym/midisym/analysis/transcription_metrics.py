import mir_eval
import numpy as np
from ..parser.container import Note


def notes_to_intervals_and_pitches(notes: list[Note]) -> tuple[np.ndarray, np.ndarray]:
    intervals = np.array([[note.start, note.end] for note in notes])
    pitches = np.array([note.pitch for note in notes])
    # remove negative intervals
    # 유효한 interval 필터링 (시작 시간이 종료 시간보다 작은 경우)
    assert intervals.shape == (
        pitches.shape[0],
        2,
    ), f"intervals.shape: {intervals.shape}, pitches.shape: {pitches.shape}"
    # 시작 시간이 종료 시간보다 작은 경우만 유지
    valid_intervals_indices = intervals[:, 0] < intervals[:, 1]
    # 음높이가 0보다 큰 경우만 유지
    valid_pitch_indices = pitches > 0
    # 두 조건을 모두 만족하는 인덱스
    valid_indices = valid_intervals_indices & valid_pitch_indices

    # 유효한 interval과 pitch 값만 유지
    intervals = intervals[valid_indices]
    pitches = pitches[valid_indices]

    return intervals, pitches


def calculate_transcription_measure(
    ref_notes: list[Note], est_notes: list[Note], offset_tolerance: float
) -> dict:
    reference_intervals, reference_pitches = notes_to_intervals_and_pitches(ref_notes)
    estimated_intervals, estimated_pitches = notes_to_intervals_and_pitches(est_notes)

    pitch_tolerance = 50.0  # 센트 단위

    metrics = mir_eval.transcription.evaluate(
        reference_intervals,
        reference_pitches,
        estimated_intervals,
        estimated_pitches,
        onset_tolerance=offset_tolerance,
        pitch_tolerance=pitch_tolerance,
    )

    # 메트릭 값을 소수점 이하 3자리로 반올림
    rounded_metrics = round_metrics(metrics, decimals=3)

    return rounded_metrics


def round_metrics(metrics: dict, decimals: int = 3) -> dict:
    """
    metrics 딕셔너리의 값을 소수점 아래 `decimals` 자리로 반올림합니다.
    """
    return {key: round(value, decimals) for key, value in metrics.items()}


def calculate_mir_eval_match_note_measure(
    ref_notes: list[Note], est_notes: list[Note], offset_tolerance: float
) -> dict:
    reference_intervals, reference_pitches = notes_to_intervals_and_pitches(ref_notes)
    estimated_intervals, estimated_pitches = notes_to_intervals_and_pitches(est_notes)

    pitch_tolerance = 50.0  # 센트 단위

    metrics = mir_eval.transcription.match_notes(
        reference_intervals,
        reference_pitches,
        estimated_intervals,
        estimated_pitches,
        onset_tolerance=offset_tolerance,
        pitch_tolerance=pitch_tolerance,
    )

    # 메트릭 값을 소수점 이하 3자리로 반올림
    rounded_metrics = round_metrics(metrics, decimals=3)

    return rounded_metrics
