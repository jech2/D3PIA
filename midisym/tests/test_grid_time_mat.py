import numpy as np

from midisym.parser.utils import get_ticks_to_seconds_grid
from midisym.analysis.utils import get_all_marker_start_end_time
from midisym.analysis.chord.chord_event import ChordEvent
from midisym.converter.matrix import get_absolute_time_mat, get_grid_quantized_time_mat, make_grid_quantized_notes, make_grid_quantized_note_prmat, visualize_pr_mat

from midisym.converter.constants import N_PITCH, PITCH_OFFSET, PR_RES, ONSET, SUSTAIN, CHORD_OFFSET
from midisym.constants import MELODY, BRIDGE, PIANO


def test_grid_time_mat(const_tempo_piano_chord_midi_parser):
    # get_grid_quantized_time_mat(const_tempo_piano_chord_midi_parser.sym_music_container, chord_style='pop909')

    sym_obj = const_tempo_piano_chord_midi_parser.sym_music_container
    add_chord_labels_to_pr=True
    chord_style='pop909'
    sym_data_type="analyzed performance MIDI -- grid from ticks"

    from midisym.converter.matrix import get_grid_quantized_time_mat


    piano_rolls, _ = get_grid_quantized_time_mat(sym_obj, chord_style=chord_style, add_chord_labels_to_pr=add_chord_labels_to_pr, sym_data_type=sym_data_type, melody_ins_ids=[0], arrangement_ins_ids=[1, 2])

    ls_mat = piano_rolls[0]
    arr_mat = piano_rolls[1]
    chord_mat = piano_rolls[2]

    
    visualize_pr_mat(ls_mat[:128], save_path='./tests/sample/pop909_ls.png')
    visualize_pr_mat(arr_mat[:128], save_path='./tests/sample/pop909_arr.png')
    visualize_pr_mat(chord_mat[:128], save_path='./tests/sample/pop909_chord.png')

    from midisym.converter.matrix import pianoroll2midi
    pianoroll2midi(arr_mat, ls_mat, arrangement=None, out_fp='./tests/sample/pop909_output.mid', pr_res=16, unit='quantize_grid')
