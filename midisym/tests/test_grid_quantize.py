from midisym.analysis.grid_quantize import (
    make_grid_quantized_notes,
    get_grid_from_tempo_changes,
)
from midisym.converter.matrix import (
    sym_to_pr_mat,
    make_grid_quantized_note_prmat,
    visualize_pr_mat,
    visualize_chord_mat,
    get_onset_matrix,
    get_most_common_onset_pattern,
)
from midisym.converter.matrix import make_grid_quantized_chord_prmat, get_onset_matrix_from_pr_mat, visualize_onset_mat


def test_get_grid_from_tempo_changes(analyzed_performance_midi_parser):
    print(analyzed_performance_midi_parser)
    grid = get_grid_from_tempo_changes(
        analyzed_performance_midi_parser.sym_music_container
    )
    print(grid[:12])
    print(analyzed_performance_midi_parser.sym_music_container.tempo_changes[:3])


def test_make_grid_quantized_notes(analyzed_performance_midi_parser):
    sym_obj = analyzed_performance_midi_parser.sym_music_container

    sym_obj, grid = make_grid_quantized_notes(
        sym_obj=sym_obj,
        sym_data_type="analyzed performance MIDI",
    )

    # dump the grid quantized notes
    analyzed_performance_midi_parser.dump("tests/sample/grid_quantized.mid")

    pr_mat = make_grid_quantized_note_prmat(sym_obj, grid)
    
def test_make_grid_quantized_notes(analyzed_performance_midi_parser):
    sym_obj = analyzed_performance_midi_parser.sym_music_container

    sym_obj, grid = make_grid_quantized_notes(
        sym_obj=sym_obj,
        sym_data_type="analyzed performance MIDI -- grid from ticks",
    )

    # dump the grid quantized notes
    analyzed_performance_midi_parser.dump("tests/sample/grid_quantized_ticks.mid")

    # pr_mat = make_grid_quantized_note_prmat(sym_obj, grid)

def test_make_grid_quantized_notes_pop1k7(analyzed_performance_midi_parser_pop1k7):
    sym_obj = analyzed_performance_midi_parser_pop1k7.sym_music_container

    sym_obj, grid = make_grid_quantized_notes(
        sym_obj=sym_obj,
        sym_data_type="analyzed performance MIDI -- grid from ticks",
    )

    # dump the grid quantized notes
    analyzed_performance_midi_parser_pop1k7.dump("tests/sample/grid_quantized_pop1k7.mid")

    # pr_mat = make_grid_quantized_note_prmat(sym_obj, grid)


def test_sym_to_prmat(analyzed_performance_midi_parser):
    sym_obj = analyzed_performance_midi_parser.sym_music_container

    pr_mat = sym_to_pr_mat(sym_obj)

    visualize_pr_mat(pr_mat, "tests/sample/pr_mat.png")


def test_get_onset_matrix(analyzed_performance_midi_parser):
    sym_obj = analyzed_performance_midi_parser.sym_music_container

    sym_obj, grid = make_grid_quantized_notes(
        sym_obj=sym_obj,
        sym_data_type="analyzed performance MIDI",
    )

    onset_mat = get_onset_matrix(sym_obj, grid)
    print(onset_mat, len(onset_mat))

    onset_patterns, mc_onset_pattern = get_most_common_onset_pattern(onset_mat)
    print(onset_patterns, mc_onset_pattern)


def test_get_onset_triplet_matrix(analyzed_performance_midi_parser):
    sym_obj = analyzed_performance_midi_parser.sym_music_container

    sym_obj, grid = make_grid_quantized_notes(
        sym_obj=sym_obj,
        sym_data_type="analyzed performance MIDI",
        quantize_resolution=3,
    )

    onset_mat = get_onset_matrix(sym_obj, grid, consider_n_voice=True)

    onset_patterns, mc_onset_pattern = get_most_common_onset_pattern(
        onset_mat, bar_quantize_resolution=12
    )

    print(onset_patterns, mc_onset_pattern)


def test_get_onset_triplet_matrix2(analyzed_performance_midi_parser):
    sym_obj = analyzed_performance_midi_parser.sym_music_container

    grid_3 = get_grid_from_tempo_changes(sym_obj, quantize_resolution=3)

    sym_obj, summation_grid = make_grid_quantized_notes(
        sym_obj=sym_obj,
        sym_data_type="analyzed performance MIDI",
        sum_grid=True,
    )

    onset_mat = get_onset_matrix(sym_obj, grid_3, consider_n_voice=True)

    onset_patterns, mc_onset_pattern = get_most_common_onset_pattern(
        onset_mat, bar_quantize_resolution=12
    )

    print(onset_patterns, mc_onset_pattern)


def test_get_left_hand_onset_matrix(analyzed_performance_midi_parser):
    sym_obj = analyzed_performance_midi_parser.sym_music_container

    sym_obj, grid = make_grid_quantized_notes(
        sym_obj=sym_obj,
        sym_data_type="analyzed performance MIDI",
    )

    pr_mat = make_grid_quantized_note_prmat(sym_obj, grid)
    from midisym.converter.matrix import get_left_right_hand_onset_matrix

    left_onset_mat, right_onset_mat = get_left_right_hand_onset_matrix(
        pr_mat, grid, consider_n_voice=True
    )

    print(left_onset_mat, len(left_onset_mat))
    print(right_onset_mat, len(right_onset_mat))

    left_onset_patterns, mc_left_onset_pattern = get_most_common_onset_pattern(
        left_onset_mat
    )
    right_onset_patterns, mc_right_onset_pattern = get_most_common_onset_pattern(
        right_onset_mat
    )

def test_const_tempo_piano_chord_midi(const_tempo_piano_chord_midi_parser):
    midi_parser = const_tempo_piano_chord_midi_parser
    
    # sym to pr matrix
    sym_obj = midi_parser.sym_music_container
    q_sym_obj, grid = make_grid_quantized_notes(
        sym_obj=sym_obj,
        sym_data_type="constant tempo MIDI"
    )

    pr_mat = make_grid_quantized_note_prmat(q_sym_obj, grid, value= 'onset_velocity', pitch_range=(21, 108))
    print(pr_mat.shape)
    n_bars = pr_mat.shape[0] // 16
    n_pitch = 88
    pitch_range = (21, 108)
    
    onset_mat = get_onset_matrix(q_sym_obj, grid, consider_n_voice=True)
    
    onset_mat = onset_mat.reshape((n_bars, -1, 1))
    print(onset_mat.shape)
    visualize_onset_mat(onset_mat[0].squeeze(), "tests/sample/onset_mat.png")
    
    pr_mat = pr_mat.reshape((n_bars, -1, n_pitch))
    print(pr_mat.shape)

    visualize_pr_mat(pr_mat[0].squeeze(), "tests/sample/onset_pr_mat.png", pitch_range=pitch_range)
    
    pr_mat2 = make_grid_quantized_note_prmat(q_sym_obj, grid, value='frame_velocity', pitch_range=pitch_range, inst_ids=[2])
    pr_mat2 = pr_mat2.reshape((n_bars, -1, n_pitch))
    print(pr_mat2.shape)
    visualize_pr_mat(pr_mat2[0].squeeze(), "tests/sample/frame_velocity_pr_mat.png", pitch_range=pitch_range)
    
    pr_mat3 = make_grid_quantized_note_prmat(q_sym_obj, grid, value='offset_velocity', pitch_range=pitch_range)
    pr_mat3 = pr_mat3.reshape((n_bars, -1, n_pitch))
    print(pr_mat3.shape)
    visualize_pr_mat(pr_mat3[0].squeeze(), "tests/sample/offset_velocity_pr_mat.png", pitch_range=pitch_range)
    
    chord_mat = make_grid_quantized_chord_prmat(q_sym_obj, grid)
    chord_mat = chord_mat.reshape((n_bars, -1, 12))
    print(chord_mat.shape)
    
    visualize_chord_mat(chord_mat[0].squeeze(), "tests/sample/chord_pr_mat.png")
    