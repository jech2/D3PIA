from .container import SymMusicContainer
from typing import Dict
import numpy as np

def seconds_to_ticks(seconds, ticks_per_beat, tempo, tempo_ref='microseconds'):
    """Convert seconds to MIDI ticks."""

    # mido tempo is in microseconds per beat
    if tempo_ref == "microseconds":   
        microseconds_per_tick = tempo / ticks_per_beat
        ticks = (seconds * 1000000) / microseconds_per_tick
    elif tempo_ref == 'BPM':
        ticks = (seconds * tempo) / 60 / ticks_per_beat
    else:
        raise ValueError(f"tempo_ref must be 'microseconds' or 'BPM'. Got {tempo_ref}")
    return int(ticks)

def get_ticks_to_seconds_grid(midi_obj: SymMusicContainer) -> np.array:
    # max_tick = max([note.end for inst in midi_obj.instruments for note in inst.notes])
    # max_tick = max(max_tick, max([note.start for inst in midi_obj.instruments for note in inst.notes]))
    # max_tick = max(max_tick, midi_obj.max_tick)
    
    ticks_to_seconds = np.zeros(midi_obj.max_tick + 1)
    ticks_per_beat = midi_obj.ticks_per_beat
    tempo_changes = midi_obj.tempo_changes  # Assumed to be a list of (tick, bpm) tuples

    # Initialize time tracking variables
    accumulated_time_seconds = 0.0
    current_tempo_index = 0
    current_tempo_change = tempo_changes[current_tempo_index]
    current_bpm = current_tempo_change.tempo
    seconds_per_beat = 60.0 / current_bpm
    ticks_per_second = ticks_per_beat / seconds_per_beat
    
    # Iterate through each tick up to the maximum tick value
    final_tick = midi_obj.max_tick

    for tick in range(final_tick + 1):
        # Update tempo if we reach a new tempo change tick
        if current_tempo_index < len(tempo_changes) - 1 and tick == tempo_changes[current_tempo_index + 1].time:
            current_tempo_index += 1
            current_tempo_change = tempo_changes[current_tempo_index]
            current_bpm = current_tempo_change.tempo
            seconds_per_beat = 60.0 / current_bpm
            ticks_per_second = ticks_per_beat / seconds_per_beat

        # Store accumulated time for the current tick
        ticks_to_seconds[tick] = accumulated_time_seconds

        # Advance accumulated time by 1 tick in seconds
        accumulated_time_seconds += 1 / ticks_per_second

    return ticks_to_seconds


def parse_chord(chord_text, chord_style):
    bass = None
    try:
        if chord_style == 'pop909':
            if chord_text == 'N':
                return None, None, None
            root, chord = chord_text.split('_')[-1].split(":")
            if '/' in chord:
                chord, _ = chord.split('/')
                # TODO: handle bass
            if '(' in chord:
                chord, _ = chord.split('(')
        elif chord_style == 'chorder':
            root, chord, bass = chord_text.split('_')
        elif chord_style == 'maj_min':
            # majmin style
            if 'maj' in chord_text:
                root = chord_text.split('maj')[0]
                chord = 'M'
            elif 'min' in chord_text:
                root = chord_text.split('min')[0]
                chord = 'm'
    except:
        raise ValueError(f"Cannot parse chord {chord_text} with style {chord_style}")
    return root, chord, bass

def resample_ticks_per_beat(sym_obj: SymMusicContainer, ticks_per_beat: int):
    """ 
    resample the ticks per beat of a sym_obj and all its notes, tempo changes, and markers
    """
    ori_ticks_per_beat = sym_obj.ticks_per_beat
    sym_obj.ticks_per_beat = ticks_per_beat
    # note
    for inst in sym_obj.instruments:
        for note in inst.notes:
            note.start = note.start * ticks_per_beat // ori_ticks_per_beat
            note.end = note.end * ticks_per_beat // ori_ticks_per_beat
    # tempo
    for tempo in sym_obj.tempo_changes:
        tempo.time = tempo.time * ticks_per_beat // ori_ticks_per_beat
    
    # marker
    for marker in sym_obj.markers:
        marker.time = marker.time * ticks_per_beat // ori_ticks_per_beat
        
    # max tick
    sym_obj.max_tick = sym_obj.max_tick * ticks_per_beat // ori_ticks_per_beat
    
    return sym_obj 