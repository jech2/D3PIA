import sys
from collections import defaultdict
import torch as th
import numpy as np
from mir_eval.util import midi_to_hz
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes

import numpy as np
import torch
from mir_eval.util import midi_to_hz

eps = sys.float_info.epsilon

SR = 16000
HOP = 512
MIN_MIDI = 21
MAX_MIDI = 108

def extract_notes(onsets, 
                  frames, 
                  velocity=None, 
                  onset_threshold=0.5, 
                  frame_threshold=0.5, 
                  defalut_velocity=64, 
                  reset_offset=True):
    """
    Finds the note timings based on the onsets and frames information

    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    velocity: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float

    Returns
    -------
    pitches: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    velocities: np.ndarray of velocity values
    """
    onsets = (onsets > onset_threshold).type(torch.int).cpu()
    frames = (frames > frame_threshold).type(torch.int).cpu()
    onset_diff = torch.cat(
        [onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1

    if velocity is None:
        velocity = torch.ones_like(onsets) * defalut_velocity

    pitches = []
    intervals = []
    velocities = []

    for nonzero in onset_diff.nonzero():
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame
        velocity_samples = []

        while onsets[offset, pitch].item() or frames[offset, pitch].item():
            if onsets[offset, pitch].item():
                velocity_samples.append(velocity[offset, pitch].item())
            offset += 1
            if offset == onsets.shape[0]:
                break
            if reset_offset and (offset != onset) and onsets[offset, pitch].item():
                break

        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])
            velocities.append(np.mean(velocity_samples) if len(velocity_samples) > 0 else 0)

    return np.array(pitches), np.array(intervals), np.array(velocities)


def notes_to_frames(pitches, intervals, shape):
    """
    Takes lists specifying notes sequences and return

    Parameters
    ----------
    pitches: list of pitch bin indices
    intervals: list of [onset, offset] ranges of bin indices
    shape: the shape of the original piano roll, [n_frames, n_bins]

    Returns
    -------
    time: np.ndarray containing the frame indices
    freqs: list of np.ndarray, each containing the frequency bin indices
    """
    roll = np.zeros(tuple(shape))
    for pitch, (onset, offset) in zip(pitches, intervals):
        roll[onset:offset, pitch] = 1

    time = np.arange(roll.shape[0])
    freqs = [roll[t, :].nonzero()[0] for t in time]
    return time, freqs

def evaluate(sample, label, sample_vel=None, vel_ref=None, band_eval=False):
    metrics = defaultdict(list)
    
    onset_est = ((sample == 2) + (sample == 4))
    frame_est = ((sample == 2) + (sample == 3) + (sample == 4))
    onset_ref = ((label == 2) + (label == 4))
    frame_ref = ((label == 2) + (label == 3) + (label == 4))

    if sample_vel is not None:
        vel_est = th.clamp(sample_vel*128, min=0, max=128)
    else:
        vel_est = th.ones_like(sample)
        vel_ref = th.ones_like(sample)
    p_est, i_est, v_est = extract_notes(onset_est, frame_est, vel_est)
    p_ref, i_ref, v_ref = extract_notes(onset_ref, frame_ref, vel_ref)

    t_est, f_est = notes_to_frames(p_est, i_est, frame_est.shape)
    t_ref, f_ref = notes_to_frames(p_ref, i_ref, frame_ref.shape)

    scaling = HOP / SR
    i_ref = (i_ref * scaling).reshape(-1, 2)
    p_ref = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_ref])
    i_est = (i_est * scaling).reshape(-1, 2)
    p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

    t_ref = t_ref.astype(np.float64) * scaling
    f_ref = [np.array([midi_to_hz(MIN_MIDI + midi)
                        for midi in freqs]) for freqs in f_ref]
    t_est = t_est.astype(np.float64) * scaling
    f_est = [np.array([midi_to_hz(MIN_MIDI + midi)
                        for midi in freqs]) for freqs in f_est]

    p, r, f, o = evaluate_notes(
        i_ref, p_ref, i_est, p_est, offset_ratio=None)
    metrics['metric/note/precision'].append(p)
    metrics['metric/note/recall'].append(r)
    metrics['metric/note/f1'].append(f)
    metrics['metric/note/overlap'].append(o)

    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
    metrics['metric/note-with-offsets/precision'].append(p)
    metrics['metric/note-with-offsets/recall'].append(r)
    metrics['metric_note_with_offsets_f1'].append(f)
    metrics['metric/note-with-offsets/overlap'].append(o)

    if band_eval:
        bands = defaultdict(list)
        band_edges = midi_to_hz(np.arange(21+22, 108, step=22))
        def get_band(p, i, type='ref'):
            for n in range(len(p)):
                if p[n] < band_edges[0]:
                    bands[f'p_{type}_0'].append(p[n])
                    bands[f'i_{type}_0'].append(i[n])
                elif p[n] < band_edges[1]:
                    bands[f'p_{type}_1'].append(p[n])
                    bands[f'i_{type}_1'].append(i[n])
                elif p[n] < band_edges[2]:
                    bands[f'p_{type}_2'].append(p[n])
                    bands[f'i_{type}_2'].append(i[n])
                else:
                    bands[f'p_{type}_3'].append(p[n])
                    bands[f'i_{type}_3'].append(i[n])
        get_band(p_ref, i_ref, type='ref')
        get_band(p_est, i_est, type='est')
                    
        for k, v in bands.items():
            bands[k] = np.asarray(v)
        for band in range(4):
            if len(bands[f'i_ref_{band}']) == 0:
                continue 
            if len(bands[f'i_est_{band}']) == 0:
                metrics[f'metric/note_band{band}/precision'].append(0.0)
                metrics[f'metric/note_band{band}/recall'].append(0.0)
                metrics[f'metric/note_band{band}/f1'].append(0.0)
                metrics[f'metric/note_band{band}/overlap'].append(0.0)
                metrics[f'metric/note_band{band}_w_offset/precision'].append(0.0)
                metrics[f'metric/note_band{band}_w_offset/recall'].append(0.0)
                metrics[f'metric/note_band{band}_w_offset/f1'].append(0.0)
                metrics[f'metric/note_band{band}_w_offset/overlap'].append(0.0)
                continue
            p, r, f, o = evaluate_notes(
                bands[f'i_ref_{band}'], bands[f'p_ref_{band}'],
                bands[f'i_est_{band}'], bands[f'p_est_{band}'], offset_ratio=None)
            metrics[f'metric/note_band{band}/precision'].append(p)
            metrics[f'metric/note_band{band}/recall'].append(r)
            metrics[f'metric/note_band{band}/f1'].append(f)
            metrics[f'metric/note_band{band}/overlap'].append(o)

            p, r, f, o = evaluate_notes(
                bands[f'i_ref_{band}'], bands[f'p_ref_{band}'],
                bands[f'i_est_{band}'], bands[f'p_est_{band}'])
            metrics[f'metric/note_band{band}_w_offset/precision'].append(p)
            metrics[f'metric/note_band{band}_w_offset/recall'].append(r)
            metrics[f'metric/note_band{band}_w_offset/f1'].append(f)
            metrics[f'metric/note_band{band}_w_offset/overlap'].append(o)
    
    return metrics