import numpy as np
import scipy
import itertools

from ..parser.container import Note
from ..parser.container import SymMusicContainer
from ..analysis.utils import get_all_notes
from ..converter.matrix import get_onset_matrix, make_grid_quantized_note_prmat
from ..analysis.grid_quantize import make_grid_quantized_notes
from ..parser.utils import get_ticks_to_seconds_grid
from ..constants import MIDI_MAX

def get_symbolic_features(sym_music_obj: SymMusicContainer, sym_data_type: str, keys = ['mean_IOI', 'mean_IOI_below_middle_C', 'global_tempo', 'avg_velocity', 'rhythmic_intensity', 'rhythmic_density', 'voice_number', 'grooving_similarity', 'pitch_class_entropy-1bar', 
                                                                                        'pitch_class_entropy-4bar'], select_inst: list = None):
    """Get symbolic features from symbolic music object

    Args:
        sym_music_obj (SymMusicContainer): symbolic music object
        keys (list[str]): list of keys to extract features

    Returns:
        dict: dictionary of features
    """
    features = {}
    all_notes = get_all_notes(sym_music_obj, exclude_drum=True, select_inst=select_inst)
    tick_to_seconds = get_ticks_to_seconds_grid(sym_music_obj)
    q_sym_music_obj, grid = make_grid_quantized_notes(
        sym_obj=sym_music_obj,
        sym_data_type=sym_data_type,
    )
    q_sym_music_obj.instruments = [inst for i, inst in enumerate(q_sym_music_obj.instruments) if i in select_inst]
    
    if 'mean_IOI' in keys:
        features['mean_IOI'] = get_mean_IOI(all_notes, tick_to_seconds)
    if 'mean_IOI_below_middle_C' in keys:
        all_notes_under_middle_C = [note for note in all_notes if note.pitch < 60]
        if len(all_notes_under_middle_C) == 0:
            features['mean_IOI_below_middle_C'] = 0
        else:
            features['mean_IOI_below_middle_C'] = get_mean_IOI(all_notes_under_middle_C, tick_to_seconds)
            if np.isnan(features['mean_IOI_below_middle_C']):
                features['mean_IOI_below_middle_C'] = 0
    if 'global_tempo' in keys:
        features['global_tempo'] = get_global_tempo(sym_music_obj, mode='mean')
    if 'avg_velocity' in keys:
        features['avg_velocity'] = get_avg_velocity(all_notes)
    if 'rhythmic_intensity' in keys:
        features['rhythmic_intensity'] = get_rhythmic_intensity(all_notes, tick_to_seconds)
    if 'rhythmic_density' in keys:
        features['rhythmic_density'] = get_rhythmic_density(q_sym_music_obj, grid)
    if 'voice_number' in keys:
        features['voice_number'] = get_voice_number(q_sym_music_obj, grid)
    if 'grooving_similarity' in keys:
        features['grooving_similarity'] = get_grooving_similarity(q_sym_music_obj, grid)
    if 'pitch_class_entropy-1bar' in keys:
        features['pitch_class_entropy-1bar'] = get_pitch_class_entropy(q_sym_music_obj, grid, window_size=1)
    if 'pitch_class_entropy-4bar' in keys:
        features['pitch_class_entropy-4bar'] = get_pitch_class_entropy(q_sym_music_obj, grid, window_size=4)
    
    return features

def get_pitch_class_histogram(pr_mat: np.array):
    # input is binary pitch matrix (n_time, pitch_range)
    # return is normalized pitch histogram (12, )
    assert pr_mat.shape[-1] == MIDI_MAX, f"last dimension should be {MIDI_MAX}, but got {pr_mat.shape[-1]}"
    pitch_hist = np.sum(pr_mat, axis=0) # (n_time, pitch_range) -> (pitch_range, )
    
    # pitch range 0~127 -> 0~11
    pitch_hist = np.array([np.sum(pitch_hist[i::12]) for i in range(12)])
    
    if np.sum(pitch_hist) == 0:
        return None
    else:
        pitch_hist = pitch_hist / np.sum(pitch_hist)
        return pitch_hist
    
def compute_histogram_entropy(hist: np.array):
    return scipy.stats.entropy(hist) / np.log(2)
    
def get_pitch_class_entropy(q_sym_obj: SymMusicContainer, grid: np.array, window_size: int = 1):
    # 1 or 4 bar level pitch class entropy
    pr_mat = make_grid_quantized_note_prmat(
        q_sym_obj, grid, value='onset_binary', do_slicing=False)
    bar_pr_mat = pr_mat.reshape(-1, 16, pr_mat.shape[-1])
    total_bars = bar_pr_mat.shape[0]
    
    pc_entropies = []
    for st_bar in range(total_bars - window_size + 1):
        windowed_pr_mat = bar_pr_mat[st_bar:st_bar + window_size]
        pitch_hist = get_pitch_class_histogram(windowed_pr_mat)
        if pitch_hist is None:
            print('No note in the window')
            continue
        pc_entropies.append(compute_histogram_entropy(pitch_hist))
    return np.mean(pc_entropies)

def get_mean_IOI(all_notes: list[Note], tick_to_seconds: np.array):
    """Get mean inter-onset interval

    Args:
        all_notes (list[Note]): list of all notes

    Returns:
        float: mean inter-onset interval
    """
    all_notes = sorted(all_notes, key=lambda x: (x.start, x.pitch))
    # IOI 계산, 인덱스 오류 방지
    IOIs = []
    for i in range(len(all_notes) - 1):
        start_time = tick_to_seconds[int(all_notes[i].start)]
        next_start_time = tick_to_seconds[int(all_notes[i + 1].start)]
        IOIs.append(next_start_time - start_time)

    return np.mean(IOIs)

def get_rhythmic_intensity(all_notes: list[Note], tick_to_seconds: np.array):
    all_notes = sorted(all_notes, key=lambda x: (x.start, x.pitch))
    
    notes_by_second = {}
    for note in all_notes:
        start_time = tick_to_seconds[int(note.start)]
        # 버림으로 처리
        start_time = int(start_time)
        if start_time not in notes_by_second:
            notes_by_second[start_time] = []
        notes_by_second[start_time].append(note)
    
    n_notes_by_second = [len(notes) for notes in notes_by_second.values()]
    return np.mean(n_notes_by_second)
    

def get_global_tempo(sym_music_obj: SymMusicContainer, mode='mean'):
    all_tempos = [tempo.tempo for tempo in sym_music_obj.tempo_changes]
    if mode == 'mean':
        return np.mean(all_tempos)
    elif mode == 'median':
        return np.median(all_tempos)
    else:
        raise ValueError("mode should be either 'mean' or 'median'")
    
def get_avg_velocity(all_notes: list[Note]):
    """Get average velocity of all notes

    Args:
        all_notes (list[Note]): list of all notes

    Returns:
        float: average velocity
    """
    return np.mean([note.velocity for note in all_notes])

def get_onset_xor_distance(matrix_a: np.array, matrix_b: np.array, time_resolution=16) -> float:
    """ get onset matrix xor distance between two onset matrix"""
    assert matrix_a.shape == matrix_b.shape == (time_resolution,), f"matrix shape should be ({time_resolution},), but got {matrix_a.shape} and {matrix_b.shape}"
    ist = np.sum( np.abs(matrix_a - matrix_b) ) / time_resolution
    assert 0 <= ist <= 1, f"IST should be in [0, 1] but got {ist}"
    return ist


def get_grooving_similarity(q_sym_obj: SymMusicContainer, grid: np.array) -> float:
    """get grooving similarity from q_sym_obj

    Args:
        q_sym_obj (SymMusicContainer): quantized symbolic music object from the make_grid_quantized_notes ftn from grid_quantize.py
        grid (np.array): grid from the make_grid_quantized_notes ftn from grid_quantize.py

    Returns:
        
        float: grooving similarity
    """
    onset_mat = get_onset_matrix(q_sym_obj, grid, consider_n_voice=False) # binary onset matrix

    bar_onset_mat = onset_mat.reshape(-1, 16)
    
    # for each bar onset matrix combinations, calculate the xor similarity
    pairs = list(itertools.combinations(range(bar_onset_mat.shape[0]), 2))
    
    grv_sims = []
    for pair in pairs:
        grv_sims.append(
            1 - get_onset_xor_distance(bar_onset_mat[pair[0]], bar_onset_mat[pair[1]])
        )
        
    return np.mean(grv_sims)

def get_rhythmic_density(q_sym_obj: SymMusicContainer, grid: np.array) -> float:
    """get rhythmic density from q_sym_obj

    Args:
        q_sym_obj (SymMusicContainer): quantized symbolic music object from the make_grid_quantized_notes ftn from grid_quantize.py
        grid (np.array): grid from the make_grid_quantized_notes ftn from grid_quantize.py

    Returns:
        float: rhythmic density
    """
    onset_mat = get_onset_matrix(q_sym_obj, grid, consider_n_voice=False) # binary onset matrix

    return np.mean(onset_mat)

def get_voice_number(q_sym_obj: SymMusicContainer, grid: np.array, count_only_onset_position: bool = True) -> int:
    """get number of voices from q_sym_obj

    Args:
        q_sym_obj (SymMusicContainer): quantized symbolic music object from the make_grid_quantized_notes ftn from grid_quantize.py
        grid (np.array): grid from the make_grid_quantized_notes ftn from grid_quantize.py

    Returns:
        int: average number of voices
    """
    onset_mat = get_onset_matrix(q_sym_obj, grid, consider_n_voice=True) # binary onset matrix

    if count_only_onset_position:
        return np.mean(onset_mat[onset_mat > 0])
    else:
        return np.mean(onset_mat)


def get_pc_one_hot(all_notes: list[Note]):
    """get pitch class one hot encoding for every notes

    Args:
        all_notes (list[Note]): list of all notes

    Returns:
        np.array: np.array of shape (len(all_notes), 12)
    """
    one_hot = np.zeros((len(all_notes), 12))
    pitches = [note.pitch for note in all_notes]
    idx = (np.arange(len(all_notes)), np.remainder(pitches, 12))
    one_hot[idx] = 1

    return one_hot


def get_octave_one_hot(all_notes: list[Note]):
    one_hot = np.zeros((len(all_notes), 10))
    pitches = [note.pitch for note in all_notes]
    idx = (np.arange(len(all_notes)), np.floor_divide(pitches, 12))
    one_hot[idx] = 1

    return one_hot


def get_duration_feature(all_notes: list[Note], ts_beats=4, ticks_per_beat=480):
    ts_beats = 4  # time signature denominator
    ticks_per_beat = 480  # default ticks per beat

    all_notes_durations = [
        (note.end - note.start) / ticks_per_beat / ts_beats for note in all_notes
    ]

    duration_feature = np.expand_dims(
        1 - np.tanh(np.array(all_notes_durations)),
        1,
    )

    return duration_feature


def feature_extraction_gather_all_feat(sym_music_obj: SymMusicContainer):

    all_notes = get_all_notes(sym_music_obj, exclude_drum=True)
    all_notes = sorted(all_notes, key=lambda x: x.start)

    pc_one_hot = get_pc_one_hot(all_notes)
    octave_one_hot = get_octave_one_hot(all_notes)
    ts = (
        sym_music_obj.time_signature_changes[0].denominator
        if len(sym_music_obj.time_signature_changes) != 0
        else 4
    )
    duration_feature = get_duration_feature(
        all_notes,
        ts_beats=ts,
        ticks_per_beat=sym_music_obj.ticks_per_beat,
    )

    feat_0 = np.hstack((duration_feature, pc_one_hot, octave_one_hot))

    return feat_0


def get_pc_one_hot_from_note_array(note_array):
    """Get one-hot encoding of pitch class."""
    one_hot = np.zeros((len(note_array), 12))
    idx = (np.arange(len(note_array)), np.remainder(note_array["pitch"], 12))
    one_hot[idx] = 1
    return one_hot


def get_octave_one_hot_from_note_array(note_array):
    """Get one-hot encoding of octave."""
    one_hot = np.zeros((len(note_array), 10))
    idx = (np.arange(len(note_array)), np.floor_divide(note_array["pitch"], 12))
    one_hot[idx] = 1
    return one_hot


def feature_extraction_score(
    note_array, sym_music_container, score=None, include_meta=False
):
    """Extract features from note_array.
    Parameters
    ----------
    note_array : structured array
        The partitura note_array object. Every entry has 5 attributes, i.e. onset_time, note duration, note velocity, voice, id.
    score : partitura score object (optional)
        The partitura score object. If provided, the meta features can be extracted.
    include_meta : bool
        Whether to include meta features. ()

    Returns
    -------
    features : np.array
        level 0 features: duration (1), pitch class one hot (12), octave one hot (10).
        level 1 features: 61 dim
    """
    # Solution for the problem of note tied in make_note_features() but it takes longer to parse each score.
    # if include_meta and isinstance(score, pt.score.Score):
    #     score = pt.score.merge_parts(score.parts)
    #     note_array = score.note_array()
    pc_oh = get_pc_one_hot(note_array)
    octave_oh = get_octave_one_hot(note_array)

    ts_beats = 4  # time signature denominator
    ticks_per_beat = 480  # default ticks per beat
    duration_feature = np.expand_dims(
        1
        - np.tanh(
            (note_array["end"] - note_array["start"]) / ticks_per_beat / ts_beats
        ),
        1,
    )
    # duration is normalized to 1 - tanh(duration / ts_beats) = [-1, 1]

    feat_0, feat_1 = np.hstack((duration_feature, pc_oh, octave_oh)), None
    return feat_0, feat_1
