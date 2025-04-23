## code from https://github.com/wazenmai/MIDI-BERT/blob/CP/melody_extraction/skyline/analyzer.py

import numpy as np
from typing import Union

from ..analysis.utils import get_all_notes
from ..parser.container import Instrument, Note, SymMusicContainer
from ..parser.midi import MidiParser
from ..parser.renderer import write_audio
import copy
from pathlib import Path


def quantize_melody(notes: list[Note], tick_resol: int = 240):
    melody_notes = []
    for note in notes:
        # cut too long notes
        if note.end - note.start > tick_resol * 8: # max 2 bars
            note.end = note.start + tick_resol * 4

        # quantize
        note.start = int(np.round(note.start / tick_resol) * tick_resol)
        note.end = int(
            np.round(note.end / tick_resol) * tick_resol
        )  # round to the nearest tick_resol (1/4 단위)

        # append
        melody_notes.append(note)
    return melody_notes


def extract_melody(notes: list[Note], tick_resol: Union[int, None] = None, do_quantize=False, make_monophonic=True, mode='highest pitch'):
    # quantize
    if tick_resol is not None and do_quantize:
        notes = quantize_melody(notes, tick_resol)
    melody_notes = list(notes)

    # sort by start, pitch from high to low
    
    if mode == 'highest pitch':
        melody_notes.sort(key=lambda x: (x.start, -x.pitch))
    elif mode == 'highest velocity':
        melody_notes.sort(key=lambda x: (x.start, -x.velocity))
    else:
        raise ValueError(f"mode {mode} is not supported")
    # exclude notes < 60
    bins = []
    prev = 0
    tmp_list = []
    if tick_resol: 
        rel_time_unit = tick_resol / 64.0 # very small
    else:
        rel_time_unit = 0
        
    for nidx in range(len(melody_notes)):
        note = melody_notes[nidx]
        if note.pitch >= 60:
            if abs(note.start - prev) > rel_time_unit:
                if tmp_list:
                    bins.append(tmp_list)
                tmp_list = [note]
            else:
                tmp_list.append(note)
            prev = note.start

    # preserve only highest one at each step
    notes_out = []
    for b in bins:
        notes_out.append(b[0])

    # avoid overlapping
    notes_out.sort(key=lambda x: x.start)
    if make_monophonic:
        for idx in range(len(notes_out) - 1):
            if notes_out[idx].end >= notes_out[idx + 1].start:
                notes_out[idx].end = notes_out[idx + 1].start

    # delete note having no duration
    notes_clean = []
    for note in notes_out:
        if note.start != note.end:
            notes_clean.append(note)

    # filtered by interval
    notes_final = [notes_clean[0]]
    for i in range(1, len(notes_clean) - 1):
        if ((notes_clean[i].pitch - notes_clean[i - 1].pitch) <= -9) and (
            (notes_clean[i].pitch - notes_clean[i + 1].pitch) <= -9
        ):
            continue
        else:
            notes_final.append(notes_clean[i])
    notes_final += [notes_clean[-1]]
    return notes_final


def save_extracted_melody(
    sym_music_container: SymMusicContainer,
    output_midi_path: Union[str, Path],
    all_notes: Union[list[Note], None] = None,
    tick_resol: int = 240,
    do_quantize: bool = False,
    make_monophonic: bool = True,
    do_write_audio: bool = False,
    dump_prev_midi: bool = False,
):
    sym_music_obj = sym_music_container
    if all_notes is None:
        all_notes = get_all_notes(sym_music_obj)

    melody_notes = extract_melody(all_notes, tick_resol=tick_resol, do_quantize=do_quantize, make_monophonic=make_monophonic)

    # save the melody notes to a new midi file
    # make a copy of the sym_music_obj
    sym_music_obj_copy = copy.copy(sym_music_obj)
    # make a new instrument
    inst = Instrument(
        program=0,
        name="melody",
    )

    inst.notes = melody_notes

    if dump_prev_midi:
        sym_music_obj_copy.instruments.append(inst)
    else:
        sym_music_obj_copy.instruments = [inst]

    # dump
    new_midi_parser = MidiParser(sym_music_container=sym_music_obj_copy)

    if isinstance(output_midi_path, str):
        output_midi_path = Path(output_midi_path)

    new_midi_parser.dump(output_midi_path)
    if do_write_audio:
        write_audio(new_midi_parser, output_midi_path.with_suffix(".wav"))

    return melody_notes
