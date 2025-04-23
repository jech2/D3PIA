from ..parser.container import Instrument, SymMusicContainer, Note
import numpy as np 

def is_same_inst(
    inst1: Instrument, inst2: Instrument, overlap_thres: float = 0.9
) -> bool:
    """check if two instruments are the same (considering the overlap notes)

    Args:
        inst1 (Instrument): compared instrument
        inst2 (Instrument): compared instrument
        overlap_thres (float, optional): percentage of overlap that decides the same instrument. Defaults to 0.9.

    Returns:
        bool: whether two instruments are the same
    """
    if inst1.program == inst2.program:
        inst1_notes_set = set(inst1.notes)
        inst2_notes_set = set(inst2.notes)
        inst1_inst2_inter = inst1_notes_set.intersection(inst2_notes_set)
        n_percentage_inter = len(inst1_inst2_inter) / min(
            len(inst2_notes_set), len(inst1_notes_set)
        )
        # print(n_percentage_inter)
        if n_percentage_inter > overlap_thres:
            return True
    else:
        return False
    return False


def find_matching_inst(
    sym_music_obj1: SymMusicContainer,
    sym_music_obj2: SymMusicContainer,
    overlap_thres: float = 0.9,
) -> list[Instrument]:
    """find matching instruments between two symbolic music objects

    Args:
        sym_music_obj1 (SymMusicContainer): symbolic music object
        sym_music_obj2 (SymMusicContainer): symbolic music object
        overlap_thres (float, optional): percentage of overlap that decides the same instrument. Defaults

    Returns:
        list: list of matching instruments
    """
    matching_inst = []
    for i, inst1 in enumerate(sym_music_obj1.instruments):
        for j, inst2 in enumerate(sym_music_obj2.instruments):
            if inst1.is_drum or inst2.is_drum:
                continue
            # print((i, j))
            if is_same_inst(inst1, inst2, overlap_thres):
                matching_inst.append((i, j))
                break
    return matching_inst


def check_exact_match_note_rate_fn(
    midi_infilling_fn: str, midi_ori_fn: str, midi_inpainted_fn: str
) -> dict:
    from midisym.parser.midi import MidiParser

    parser = MidiParser()

    midi_obj_infilling = parser.parse(midi_infilling_fn)
    midi_obj_ori = parser.parse(midi_ori_fn)
    midi_obj_inpainted = parser.parse(midi_inpainted_fn)

    # print(midi_obj_infilling.num_instruments)
    # print(midi_obj_ori.num_instruments)
    # print(midi_obj_inpainted.num_instruments)

    return check_exact_match_note_rate_sym_obj(
        midi_obj_infilling, midi_obj_ori, midi_obj_inpainted
    )


def check_exact_match_note_rate_sym_obj(
    midi_obj_infilling: SymMusicContainer,
    midi_obj_ori: SymMusicContainer,
    midi_obj_inpainted: SymMusicContainer,
) -> dict:
    from midisym.analysis.utils import find_matching_inst

    overlap_thres = 0.1
    matching_inst = find_matching_inst(midi_obj_infilling, midi_obj_ori, overlap_thres)
    matching_inst_2 = find_matching_inst(
        midi_obj_infilling, midi_obj_inpainted, overlap_thres
    )
    # print(matching_inst)
    # print('----')
    # print(matching_inst_2)

    # assert len(matching_inst) == min(midi_obj_infilling.num_instruments, midi_obj_ori.num_instruments)
    # assert len(matching_inst_2) == min(midi_obj_infilling.num_instruments, midi_obj_inpainted.num_instruments)

    exact_match_rates = []
    # print('original - infilling')
    for (i, j), (_, k) in zip(matching_inst, matching_inst_2):
        diff_notes = set(midi_obj_ori.instruments[j].notes) - set(
            midi_obj_infilling.instruments[i].notes
        )
        # print(len(diff_notes), '/', len(midi_obj_ori.instruments[i].notes))

        diff_notes2 = set(midi_obj_ori.instruments[j].notes) - set(
            midi_obj_inpainted.instruments[k].notes
        )
        # print(len(diff_notes2), '/', len(midi_obj_ori.instruments[j].notes))

        inter_origin_inpaint = diff_notes.intersection(diff_notes2)
        # print(len(inter_origin_inpaint), '/', len(diff_notes), len(inter_origin_inpaint) / len(diff_notes))
        # print(len(inter_origin_inpaint), '/', len(diff_notes2), len(inter_origin_inpaint) / len(diff_notes2))
        exact_match_rate1 = (
            len(inter_origin_inpaint) / len(diff_notes) if len(diff_notes) > 0 else 0
        )
        exact_match_rate2 = (
            len(inter_origin_inpaint) / len(diff_notes2) if len(diff_notes2) > 0 else 0
        )

        exact_match_rates.append(min(exact_match_rate1, exact_match_rate2))

        # print('----')

    return {
        "matching_inst_input_ori": matching_inst,
        "matching_inst_input_inpainted": matching_inst_2,
        "exact_match_rates": exact_match_rates,
    }


def get_inpainted_pos_notes(
    sym_music_obj_ori: SymMusicContainer, sym_music_obj_infilling: SymMusicContainer
) -> list:
    overlap_thres = 0.1
    matching_inst = find_matching_inst(
        sym_music_obj_infilling, sym_music_obj_ori, overlap_thres
    )

    exact_match_rates = []
    # print('original - infilling')
    notes = []
    for i, j in matching_inst:
        diff_notes = set(sym_music_obj_ori.instruments[j].notes) - set(
            sym_music_obj_infilling.instruments[i].notes
        )
        notes.extend(diff_notes)
    return notes


def get_all_notes(sym_music_obj: SymMusicContainer, exclude_drum: bool = True, select_inst: list = None) -> list:
    all_notes = []
    for i, inst in enumerate(sym_music_obj.instruments):
        if select_inst is not None and i not in select_inst:
            continue
        if exclude_drum and inst.is_drum:
            continue
        all_notes.extend(inst.notes)

    all_notes = set(all_notes)
    return all_notes


def time_to_ticks(time_seconds: float, ticks_per_beat: int, bpm: float) -> int:
    # 초당 tick 수(TPS) 계산
    ticks_per_second = (ticks_per_beat * bpm) / 60
    # 시간(초)을 tick으로 변환
    ticks = time_seconds * ticks_per_second
    return int(round(ticks))


def time_to_ticks_perf_midi(
    note: Note, time_seconds: float, ticks_per_beat: int, tempo_changes: list
) -> int:
    # find the nearest TempoChange event before the note
    tempo_change = None
    for tc in tempo_changes:
        if tc.time >= note.start:
            break
        tempo_change = tc

    if tempo_change is None:
        return time_to_ticks(time_seconds, ticks_per_beat, 120)
    else:
        cur_tempo = tempo_change.tempo
        seconds_per_beat = float(60) / float(cur_tempo)
        seconds_per_tick = seconds_per_beat / float(ticks_per_beat)
        time_seconds_as_tick = time_seconds / seconds_per_tick
        return int(round(time_seconds_as_tick))


def ticks_to_time(ticks: int, ticks_per_beat: int, bpm: float) -> float:
    # 초당 tick 수(TPS) 계산
    ticks_per_second = (ticks_per_beat * bpm) / 60
    # tick을 시간(초)으로 변환
    time_seconds = ticks / ticks_per_second
    return time_seconds


def is_monophonic(notes: list[Note]) -> bool:
    notes.sort(key=lambda x: x.start)
    for i in range(len(notes) - 1):
        if notes[i].end > notes[i + 1].start:
            return False
    return True

def get_all_marker_start_end_time(sym_obj: SymMusicContainer, grid: np.array) -> list[tuple]:
    # get all marker start and end
    all_markers = []
    for i, marker in enumerate(sym_obj.markers[:-1]):
        all_markers.append((marker.text, marker.time, sym_obj.markers[i+1].time))    
    # last marker
    if sym_obj.markers[-1].time < grid[-1]:
        all_markers.append((marker.text, sym_obj.markers[-1].time, grid[-1]))

    return all_markers
