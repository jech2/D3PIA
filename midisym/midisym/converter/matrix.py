import numpy as np
from ..parser.container import SymMusicContainer
from ..analysis.grid_quantize import (
    make_grid_quantized_notes,
)
from ..analysis.utils import get_all_marker_start_end_time
from ..analysis.chord.chord_event import chord_name_to_chroma

from ..constants import PITCH_ID_TO_NAME, PITCH_RANGE, MIDI_MAX, MELODY, BRIDGE, PIANO, QUANTIZE_RESOLUTION
from collections import Counter
import numpy as np

from ..parser.utils import get_ticks_to_seconds_grid
from ..analysis.chord.chord_event import ChordEvent

from ..converter.constants import N_PITCH, PITCH_OFFSET, PR_RES, ONSET, SUSTAIN, CHORD_OFFSET, POP1k7_MELODY, POP1k7_ARRANGEMENT
from ..parser.container import Note, Instrument
from ..parser.midi import MidiParser
from ..parser.utils import parse_chord    
from ..analysis.chord.chord_interval import ChordInterval 

def sym_to_pr_mat(sym_obj: SymMusicContainer, sym_data_type="analyzed performance MIDI"):
    sym_obj, grid = make_grid_quantized_notes(
        sym_obj=sym_obj,
        sym_data_type=sym_data_type,
    )

    # dump the grid quantized notes
    pr_mat = make_grid_quantized_note_prmat(sym_obj, grid)
    return pr_mat


def make_grid_quantized_note_prmat(sym_obj: SymMusicContainer, 
                                   grid: np.array, 
                                   value: str = 'duration', 
                                   pitch_range: tuple[int, int] = PITCH_RANGE, inst_ids = [MELODY, BRIDGE, PIANO],
                                   do_slicing: bool = True):
    # grid 인덱스에 맞는 note에 대해 pr_mat에 등록
    
    min_pitch, max_pitch = pitch_range
    pr_mat = np.zeros((len(grid), MIDI_MAX), dtype=np.int64)
    for inst_idx, inst in enumerate(sym_obj.instruments):
        if inst_idx not in inst_ids:
            print(f"Skipping instrument {inst_idx}, not in {inst_ids}")
            continue
        
        for note in inst.notes:
            try:
                start_idx = np.where(grid == note.start)[0][0]
                end_idx = np.where(grid == note.end)[0][0]
                if value == 'duration':
                    pr_mat[start_idx, note.pitch] = end_idx - start_idx
                elif value == 'binary':
                    pr_mat[start_idx:end_idx, note.pitch] = 1
                elif value == 'onset_binary':
                    pr_mat[start_idx, note.pitch] = 1
                elif value == 'onset_velocity':
                    pr_mat[start_idx, note.pitch] = note.velocity
                elif value == 'frame_velocity':
                    pr_mat[start_idx:end_idx, note.pitch] = note.velocity
                elif value == 'offset_velocity':
                    pr_mat[end_idx, note.pitch] = note.velocity
                elif value == 'onset_frame':
                    pr_mat[start_idx, note.pitch] = ONSET
                    pr_mat[start_idx+1:end_idx, note.pitch] = SUSTAIN
                else:
                    raise ValueError('value should be in [duration, binary, onset_binary, onset_velocity, offset_velocity, frame_velocity]')
            except:
                pass

    # slicing the pitch range
    if do_slicing:
        pr_mat = pr_mat[:, min_pitch:max_pitch+1]

    return pr_mat


def make_grid_quantized_chord_prmat(sym_obj: SymMusicContainer, grid: np.array):
    chord_pr_mat = np.zeros((len(grid), 12), dtype=np.int64)
    all_markers = get_all_marker_start_end_time(sym_obj, grid)
    for marker in all_markers:
        chord, start, end = marker
        start_idx = np.where(grid == start)[0][0]
        end_idx = np.where(grid == end)[0][0]
        chroma = chord_name_to_chroma(chord, start_idx, end_idx)
        
        chord_pr_mat[start_idx:end_idx, chroma] = 1
    return chord_pr_mat
        
def get_onset_matrix(
    sym_obj: SymMusicContainer, grid: np.array, consider_n_voice: bool = False, inst_ids = [MELODY, BRIDGE, PIANO]
):
    pr_mat = make_grid_quantized_note_prmat(sym_obj, grid, value='duration', inst_ids=inst_ids)
    onset_mat = np.zeros(pr_mat.shape[0], dtype=np.int64)
    for i in range(pr_mat.shape[0]):
        if consider_n_voice:
            onset_mat[i] = np.sum(pr_mat[i] > 0)  # number of voices
        else:
            onset_mat[i] = np.sum(pr_mat[i]) > 0  # binary
    return onset_mat


def get_onset_matrix_from_pr_mat(pr_mat: np.array, consider_n_voice: bool = False):
    onset_mat = np.zeros(pr_mat.shape[0], dtype=np.int64)
    for i in range(pr_mat.shape[0]):
        if consider_n_voice:
            onset_mat[i] = np.sum(pr_mat[i] > 0)  # number of voices
        else:
            onset_mat[i] = np.sum(pr_mat[i]) > 0  # binary
    return onset_mat


def get_left_right_hand_onset_matrix(
    pr_mat: np.array, grid: np.array, consider_n_voice: bool = False
):
    # find the median pitch of the total pitches
    median_pitch = np.median(np.where(pr_mat > 0)[1])
    print(f"Median pitch: {median_pitch}")
    left_hand_pr_mat = pr_mat[:, : int(median_pitch)]
    right_hand_pr_mat = pr_mat[:, int(median_pitch) :]
    left_hand_onset_mat = get_onset_matrix_from_pr_mat(
        left_hand_pr_mat, consider_n_voice=consider_n_voice
    )
    right_hand_pr_mat = get_onset_matrix_from_pr_mat(
        right_hand_pr_mat, consider_n_voice=consider_n_voice
    )
    return left_hand_onset_mat, right_hand_pr_mat


def get_most_common_onset_pattern(onset_mat, bar_quantize_resolution=16):
    total_bars = onset_mat.shape[0] // bar_quantize_resolution
    onset_patterns = []
    for i in range(total_bars):
        onset_pattern = onset_mat[
            i * bar_quantize_resolution : (i + 1) * bar_quantize_resolution
        ]
        onset_patterns.append(onset_pattern)

    onset_patterns_str = []
    for pattern in onset_patterns:
        onset_patterns_str.append("".join([str(p) for p in pattern]))
    counter = Counter(onset_patterns_str)
    # value with the highest count
    most_common_onset_pattern = counter.most_common(1)[0][0]

    return onset_patterns, most_common_onset_pattern


def visualize_pr_mat(pr_mat: np.array, save_path: str = "pr_mat.png", pitch_range: tuple[int, int] = PITCH_RANGE):
    import matplotlib.pyplot as plt

    assert pr_mat.shape[1] == 128 or pr_mat.shape[1] == 88
    plt.imshow(pr_mat.T, aspect="auto", cmap="viridis", interpolation="none")  # Use a clearer colormap
    # flip y axis
    plt.gca().invert_yaxis()
    plt.ylabel("Pitch")
    # yticks values according to the pitch range
    min_pitch, max_pitch = pitch_range
    # min ~ max 사이의 C1, C2, ... 표시
    ytick_pitches = [i for i in range(min_pitch, max_pitch+1) if i % 12 == 0]
    print(ytick_pitches)
    pitch_labels = [f"{PITCH_ID_TO_NAME[i%12]}{i // 12 - 1}" for i in ytick_pitches]
    ytick_pitches = [i - min_pitch for i in ytick_pitches]
    plt.yticks(ytick_pitches, pitch_labels)
    plt.xlabel("Time Step")
    plt.savefig(save_path)
    plt.close()

def visualize_chord_mat(chord_mat: np.array, save_path: str = "chord_mat.png"):
    import matplotlib.pyplot as plt

    assert chord_mat.shape[1] == 12
    plt.imshow(chord_mat.T, aspect="auto", cmap="viridis", interpolation="none")  # Use a clearer colormap
    # set yticks as pitch class
    plt.gca().invert_yaxis()

    plt.yticks(np.arange(12), [f"{PITCH_ID_TO_NAME[i]}" for i in range(12)])
    # flip the y ticks
    plt.ylabel("Pitch Class")
    plt.xlabel("Time Step")
    plt.savefig(save_path)
    plt.close()
    
def visualize_onset_mat(onset_mat: np.array, save_path: str = "onset_mat.png"):
    import matplotlib.pyplot as plt

    plt.plot(onset_mat)
    plt.ylabel("Onset")
    plt.xlabel("Time Step")
    plt.savefig(save_path)  
    plt.close()
    
def get_absolute_time_mat(sym_obj: SymMusicContainer, pr_res=PR_RES, add_chord_labels_to_pr=True, chord_style='chorder'):  
    ticks_to_seconds = get_ticks_to_seconds_grid(sym_obj)
    last_sec = ticks_to_seconds[-1]
    # making 32 Hz piano roll
    piano_roll_xs = np.arange(0, last_sec, 1/pr_res) # len = 32 * last_sec
    
    piano_rolls = []

    for idx, inst in enumerate(sym_obj.instruments):
        note_infos = []
        for note in inst.notes:
            note.start = ticks_to_seconds[note.start]
            note.end = ticks_to_seconds[note.end]
            note_info = (note.start, note.end, note.pitch, note.velocity)
            note_infos.append(note_info)
            
        piano_roll = np.zeros((len(piano_roll_xs), N_PITCH))
            
        for note_info in note_infos:
            start, end, pitch, velocity = note_info
            start_idx = int(start * pr_res)
            end_idx = int(end * pr_res)
            if end_idx - start_idx < 2:
                end_idx = start_idx + 2
                continue
            pr_pitch = pitch - PITCH_OFFSET
        
            # piano_roll[start_idx:end_idx, pr_pitch] = velocity
            piano_roll[start_idx, pr_pitch] = ONSET
            piano_roll[start_idx+1:end_idx, pr_pitch] = SUSTAIN
            
        if add_chord_labels_to_pr:
            all_markers = get_all_marker_start_end_time(sym_obj, piano_roll_xs)
            all_markers_sec = [(marker[0], ticks_to_seconds[marker[1]], ticks_to_seconds[marker[2]]) for marker in all_markers if marker[1] < len(ticks_to_seconds) and marker[2] < len(ticks_to_seconds)]

            if idx == POP1k7_MELODY:
                prev_chord = None
                for i, marker in enumerate(all_markers_sec):
                    text, start, end = marker
                    if 'bpm' in text:
                        continue
                    else:
                        from midisym.parser.utils import parse_chord    
                        root, quality, bass = parse_chord(text, chord_style)
                        if root in ['N', None]:
                            if prev_chord is not None:
                                root, quality, bass = prev_chord
                            else:
                                continue
                        chord = ChordEvent(root, quality, -1, -1, None)
                        chord_pitches = chord.to_pitches(as_name=False)
                        chord_pitches = [pitch - PITCH_OFFSET for pitch in chord_pitches]
                        chord_pitches = [pitch - CHORD_OFFSET for pitch in chord_pitches if pitch - CHORD_OFFSET >= 0] # not to make the overlap with melody
                        prev_chord = (root, quality, bass)
                
                    for pitch in chord_pitches:
                        start_idx = int(start * pr_res)
                        end_idx = int(end * pr_res)
                        # piano_roll[start_idx:end_idx, pitch] = 100
                        piano_roll[start_idx, pitch] = ONSET
                        piano_roll[start_idx+1:end_idx, pitch] = SUSTAIN 
        piano_rolls.append(piano_roll)
                    
    return piano_rolls, piano_roll_xs, note_infos


# input midi inst: melody, bridge, arrangement 순서
# output: (melody + bridge) + chord, (melody + bridge) + arrangement

def get_grid_quantized_time_mat(sym_obj: SymMusicContainer, add_chord_labels_to_pr=True, chord_style='chorder', sym_data_type="analyzed performance MIDI -- grid from ticks"):  
    sym_obj, grid = make_grid_quantized_notes(
        sym_obj=sym_obj,
        sym_data_type=sym_data_type,
    )

    print(f"Grid shape: {grid.shape}")

    # dump the grid quantized notes
    track_mat = []
    for inst_idx, inst in enumerate(sym_obj.instruments):
        track_mat.append(make_grid_quantized_note_prmat(sym_obj, grid, value='onset_frame', pitch_range=(21, 108), inst_ids=[inst_idx]))
        
    # ls_mat = make_grid_quantized_note_prmat(sym_obj, grid, value='onset_frame', pitch_range=(21, 108), inst_ids=melody_ins_ids)
    # arr_mat = make_grid_quantized_note_prmat(sym_obj, grid, value='onset_frame', pitch_range=(21, 108), inst_ids=arrangement_ins_ids)
    chord_mat = np.zeros_like(track_mat[0]) # chord only matrix
        
    all_markers = get_all_marker_start_end_time(sym_obj, grid)
    # all_markers_sec = [(marker[0], ticks_to_seconds[marker[1]], ticks_to_seconds[marker[2]]) for marker in all_markers if marker[1] < len(ticks_to_seconds) and marker[2] < len(ticks_to_seconds)]

    prev_chord = None
    for i, marker in enumerate(all_markers):
        text, start, end = marker
        if 'bpm' in text:
            continue
        else:
            root, quality, bass = parse_chord(text, chord_style)
            if root in ['N', None]:
                if prev_chord is not None:
                    root, quality, bass = prev_chord
                else:
                    continue
            chord = ChordEvent(root, quality, -1, -1, None)
            chord_pitches = chord.to_pitches(as_name=False)
            chord_pitches = [pitch - PITCH_OFFSET for pitch in chord_pitches]
            chord_pitches = [pitch - CHORD_OFFSET for pitch in chord_pitches if pitch - CHORD_OFFSET >= 0] # not to make the overlap with melody
            prev_chord = (root, quality, bass)
    
        for pitch in chord_pitches:
            start_idx = np.where(grid == start)[0][0]
            end_idx = np.where(grid == end)[0][0]
            # piano_roll[start_idx:end_idx, pitch] = 100
            chord_mat[start_idx, pitch] = ONSET
            chord_mat[start_idx+1:end_idx, pitch] = SUSTAIN
    
    n_beats = len(grid) // QUANTIZE_RESOLUTION
    polydis_chord_mat = np.zeros((n_beats, 36))
    for i, marker in enumerate(all_markers):
        text, start, end = marker
        # beat level label
        current_beat = start // sym_obj.ticks_per_beat # n_th beat
        current_beat_end = end // sym_obj.ticks_per_beat
        chord_vector = chord_labels_to_one_hot(text, chord_style=chord_style)  # shape (36,)
        polydis_chord_mat[current_beat:current_beat_end] = chord_vector
 
    piano_rolls = {'track_mat': track_mat, 'chord_mat': chord_mat, 'polydis_chord_mat': polydis_chord_mat}
    return piano_rolls, grid


def pitch_to_chord_polydis_chord_input(pitches, root_number, bass_number):
    chroma = np.zeros(12)
    for pitch in pitches:
        chroma[pitch % 12] = 1
    
    root_one_hot = np.zeros(12)
    root_one_hot[root_number % 12] = 1
    
    bass_one_hot = np.zeros(12)
    bass_one_hot[bass_number % 12] = 1
    
    return np.concatenate([root_one_hot, chroma, bass_one_hot])


def chord_labels_to_one_hot(chd_str, chord_style='pop909'):
    root, quality, bass = parse_chord(chd_str, chord_style=chord_style)
    chord = ChordEvent(root, quality, 0, 16, bass)
    if root not in ['None', None]:
        ci = ChordInterval()
        root_number = ci.pitch_to_midi_pitch(root, 4)
        if bass is not None:
            bass_number = ci.pitch_to_midi_pitch(bass, 4)
        else:
            bass_number = root_number

        pitches = chord.to_pitches(as_name=False)
    
        return pitch_to_chord_polydis_chord_input(pitches, root_number, bass_number)
    else:
        # zeros
        return np.zeros(36)

def pianoroll2notes(piano_rolls, ticks_per_beat, pr_res=32, unit='Hz'):
    # piano roll is (T, 88)
    # piano roll to midi
    onset_times = [ -1 for _ in range(88) ]
    offset_times = [ -1 for _ in range(88) ]
    
    notes = []
    if unit == 'quantize_grid':
        interval = int(ticks_per_beat/(pr_res / 4))
        grid = np.arange(0, piano_rolls.shape[0] * interval, interval)

    for t in range(piano_rolls.shape[0]):
        for p in range(piano_rolls.shape[1]):
            if piano_rolls[t, p] == 1:
                if onset_times[p] == -1:
                    onset_times[p] = t
                elif offset_times[p] != -1:
                    # seconds to ticks
                    if unit == 'Hz':
                        start = round(onset_times[p] * ticks_per_beat * 2 / pr_res)
                        end = round((offset_times[p] + 1) * ticks_per_beat * 2 / pr_res)
                    elif unit == 'quantize_grid':
                        start = grid[onset_times[p]]
                        end = grid[offset_times[p] + 1]

                    notes.append(Note(
                        pitch=p+21, 
                        velocity=64,
                        start=start,
                        end=end
                    ))
                    onset_times[p] = t
                    offset_times[p] = -1
                    # print(f'pitch: {p+21}, start: {start}, end: {end}')
            elif piano_rolls[t, p] == 2:
                offset_times[p] = t
            elif piano_rolls[t, p] == 0:
                if piano_rolls[t-1, p] == 1:
                    offset_times[p] = t
                if onset_times[p] != -1 and offset_times[p] != -1:
                    if unit == 'Hz':
                        start = round(onset_times[p] * ticks_per_beat * 2 / pr_res)
                        end = round((offset_times[p] + 1) * ticks_per_beat * 2 / pr_res)
                    elif unit == 'quantize_grid':
                        if offset_times[p] + 1 >= len(grid):
                            continue
                        start = grid[onset_times[p]]
                        end = grid[offset_times[p] + 1]
                    
                    notes.append(Note(
                        pitch=p+21, 
                        velocity=64,
                        start=start,
                        end=end
                    ))
                    onset_times[p] = -1
                    offset_times[p] = -1
                    # print(f'pitch: {p+21}, start: {start}, end: {end}')
                        
        inst = Instrument()    
        inst.notes = notes
        
    return notes, inst
    
def pianoroll2midi(piano_rolls, leadsheet=None, arrangement=None, out_fp='output.mid', pr_res=32, unit='Hz'):

    midi = MidiParser()

    print(midi.sym_music_container.ticks_per_beat)

    ticks_per_beat = midi.sym_music_container.ticks_per_beat

    # onset_time_idxs = np.where(piano_rolls == 1)
    # print(onset_time_idxs)

    _, inst = pianoroll2notes(piano_rolls, ticks_per_beat, pr_res=pr_res, unit=unit)
    midi.sym_music_container.instruments.append(inst)

    if leadsheet is not None:
        _, inst_lead = pianoroll2notes(leadsheet, ticks_per_beat, pr_res=pr_res, unit=unit)
        midi.sym_music_container.instruments.append(inst_lead)
    
    if arrangement is not None:
        _, inst_arr = pianoroll2notes(arrangement, ticks_per_beat, pr_res=pr_res, unit=unit)
        midi.sym_music_container.instruments.append(inst_arr)
    
    midi.dump(out_fp)
    print('midi saved')
