from midisym.analysis.utils import get_all_notes
from midisym.analysis.feature_extraction import (
    get_pc_one_hot,
    get_octave_one_hot,
    get_pc_one_hot_from_note_array,
    get_octave_one_hot_from_note_array,
)
import numpy as np


def test_pitch_one_hot(midi_parser):
    all_notes = get_all_notes(midi_parser.sym_music_container, exclude_drum=True)
    all_notes = sorted(all_notes, key=lambda x: x.start)

    note_array = np.array(
        [(n.start, n.end - n.start, n.pitch) for n in all_notes],
        dtype=[("onset_div", "i4"), ("duration_div", "i4"), ("pitch", "i4")],
    )

    one_hot = get_pc_one_hot(all_notes)
    one_hot_np = get_pc_one_hot_from_note_array(note_array)

    assert np.array_equal(one_hot, one_hot_np)


def test_octave_one_hot(midi_parser):
    all_notes = get_all_notes(midi_parser.sym_music_container, exclude_drum=True)
    all_notes = sorted(all_notes, key=lambda x: x.start)

    note_array = np.array(
        [(n.start, n.end - n.start, n.pitch) for n in all_notes],
        dtype=[("onset_div", "i4"), ("duration_div", "i4"), ("pitch", "i4")],
    )

    one_hot = get_octave_one_hot(all_notes)
    one_hot_np = get_octave_one_hot_from_note_array(note_array)

    assert np.array_equal(one_hot, one_hot_np)


# benchmarks
def pitch_one_hot_from_note_array(midi_parser):
    all_notes = get_all_notes(midi_parser.sym_music_container, exclude_drum=True)
    all_notes = sorted(all_notes, key=lambda x: x.start)

    note_array = np.array(
        [(n.start, n.end - n.start, n.pitch) for n in all_notes],
        dtype=[("onset_div", "i4"), ("duration_div", "i4"), ("pitch", "i4")],
    )

    get_pc_one_hot_from_note_array(note_array)


def octave_one_hot_from_note_array(midi_parser):
    all_notes = get_all_notes(midi_parser.sym_music_container, exclude_drum=True)
    all_notes = sorted(all_notes, key=lambda x: x.start)

    note_array = np.array(
        [(n.start, n.end - n.start, n.pitch) for n in all_notes],
        dtype=[("onset_div", "i4"), ("duration_div", "i4"), ("pitch", "i4")],
    )

    get_octave_one_hot_from_note_array(note_array)


def pc_one_hot_from_all_notes(midi_parser):
    all_notes = get_all_notes(midi_parser.sym_music_container, exclude_drum=True)
    all_notes = sorted(all_notes, key=lambda x: x.start)

    get_pc_one_hot(all_notes)


def octave_one_hot_from_all_notes(midi_parser):
    all_notes = get_all_notes(midi_parser.sym_music_container, exclude_drum=True)
    all_notes = sorted(all_notes, key=lambda x: x.start)

    get_octave_one_hot(all_notes)


# benchmarks
# def test_benchmark_pitch_one_hot_from_note_array(benchmark, midi_parser):
#     benchmark(pitch_one_hot_from_note_array, midi_parser)


# def test_benchmark_octave_one_hot_from_note_array(benchmark, midi_parser):
#     benchmark(octave_one_hot_from_note_array, midi_parser)


# def test_benchmark_pc_one_hot_from_all_notes(benchmark, midi_parser):
#     benchmark(pc_one_hot_from_all_notes, midi_parser)


# def test_benchmark_octave_one_hot_from_all_notes(benchmark, midi_parser):
#     benchmark(octave_one_hot_from_all_notes, midi_parser)


from midisym.analysis.feature_extraction import feature_extraction_gather_all_feat


def test_feature_extraction(midi_parser):
    feat_0 = feature_extraction_gather_all_feat(midi_parser.sym_music_container)
    print(feat_0)
    print(feat_0.shape)

from midisym.analysis.feature_extraction import get_global_tempo, get_avg_velocity, get_mean_IOI

def test_get_global_tempo(midi_parser):
    tempo = get_global_tempo(midi_parser.sym_music_container, mode="mean")
    print('mean', tempo)
    tempo = get_global_tempo(midi_parser.sym_music_container, mode="median")    
    print('median', tempo)
    
def test_get_avg_velocity(midi_parser):
    avg_velocity = get_avg_velocity(get_all_notes(midi_parser.sym_music_container))
    print('avg_velocity', avg_velocity)
    
def test_get_mean_IOI(analyzed_performance_midi_parser):
    from midisym.parser.utils import get_ticks_to_seconds_grid
    sym_music_obj = analyzed_performance_midi_parser.sym_music_container
    
    all_notes = get_all_notes(sym_music_obj, exclude_drum=True)
    
    tick_to_seconds = get_ticks_to_seconds_grid(sym_music_obj)
    
    mean_IOI = get_mean_IOI(all_notes, tick_to_seconds)
    print('mean_IOI (sec)', mean_IOI)
    
    # note with pitches under middle C
    all_notes_under_middle_C = [note for note in all_notes if note.pitch < 60]
    mean_IOI = get_mean_IOI(all_notes_under_middle_C, tick_to_seconds)
    print('mean_IOI under middle C (sec)', mean_IOI)

    from midisym.analysis.feature_extraction import get_rhythmic_intensity
    rhythmic_intensity = get_rhythmic_intensity(all_notes, tick_to_seconds)
    print('rhythmic_intensity', rhythmic_intensity)
    
def test_grooving_similarity(midi_parser):
    # get the bar level onset features
    # sym to pr matrix
    from midisym.analysis.grid_quantize import make_grid_quantized_notes
    from midisym.converter.matrix import get_onset_matrix
    from midisym.analysis.feature_extraction import get_grooving_similarity, get_voice_number, get_rhythmic_density
    
    
    sym_obj = midi_parser.sym_music_container
    q_sym_obj, grid = make_grid_quantized_notes(
        sym_obj=sym_obj,
        sym_data_type="constant tempo MIDI"
    )

    grv_sim = get_grooving_similarity(q_sym_obj, grid)
    print('grooving similarity', grv_sim)
    
    voice_number = get_voice_number(q_sym_obj, grid)
    print('voice number', voice_number)
    
    voice_number = get_voice_number(q_sym_obj, grid, count_only_onset_position=False)
    print('voice number', voice_number)
    
    rhythmic_density = get_rhythmic_density(q_sym_obj, grid)
    print('rhythmic density', rhythmic_density)

    from midisym.analysis.feature_extraction import get_pitch_class_entropy
    p1 = get_pitch_class_entropy(q_sym_obj, grid, window_size=1)
    print('pitch class entropy - 1bar', p1)

    p4 = get_pitch_class_entropy(q_sym_obj, grid, window_size=4)
    print('pitch class entropy - 4bar', p4)

def test_get_symbolic_music_feature(analyzed_performance_midi_parser):
    from midisym.analysis.feature_extraction import get_symbolic_features
    sym_music_obj = analyzed_performance_midi_parser.sym_music_container
    sf = get_symbolic_features(sym_music_obj, 'analyzed performance MIDI -- grid from ticks')
    print('analyzed performance midi: features', sf)

def test_get_symbolic_music_feature2(midi_parser):
    from midisym.analysis.feature_extraction import get_symbolic_features
    sym_music_obj = midi_parser.sym_music_container
    sf = get_symbolic_features(sym_music_obj, 'constant tempo MIDI')
    print('const tempo midi: features', sf)
