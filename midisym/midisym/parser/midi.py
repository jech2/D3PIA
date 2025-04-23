import os
import mido
from typing import List
from pathlib import Path
import functools
import numpy as np

from .container import (
    SymMusicContainer,
    Note,
    Instrument,
    TimeSignature,
    KeySignature,
    TempoChange,
    Marker,
)
from ..constants import MAX_CHANNELS, DEFAULT_BPM, KEY_NUMBER_TO_MIDO_KEY_NAME
from symusic import Score
from .utils import resample_ticks_per_beat


# We "hack" mido's Note_on messages checks to allow to add an "end" attribute, that
# will serve us to sort the messages in the good order when writing a MIDI file.
new_set = {"end", *mido.messages.SPEC_BY_TYPE["note_on"]["attribute_names"]}
mido.messages.SPEC_BY_TYPE["note_on"]["attribute_names"] = new_set
mido.messages.checks._CHECKS["end"] = mido.messages.checks.check_time


class MidiParser:
    def __init__(
        self,
        file_path: Path | str | None = None,
        use_symusic: bool = False,
        sym_music_container: SymMusicContainer | None = None,
        verbose: bool = False,
        ticks_per_beat: int | None = None,
    ):
        # for parsing
        self.use_symusic = use_symusic
        self.verbose = verbose
        self.init_parser()
        if file_path:
            self.sym_music_container = self.parse(file_path)
        elif sym_music_container:
            self.sym_music_container = sym_music_container
            
        if ticks_per_beat != None and ticks_per_beat != self.sym_music_container.ticks_per_beat:
            print('resampling the ticks per beat as ', ticks_per_beat)
            self.sym_music_container = resample_ticks_per_beat(self.sym_music_container, ticks_per_beat)

    def init_parser(self):
        self._current_instrument_name = ""
        self._current_playing_notes = {}
        self._channel_program_map = {i: 0 for i in range(MAX_CHANNELS)}
        self._accum_time = 0

        self.sym_music_container = SymMusicContainer()

    def parse(self, file_path: str) -> SymMusicContainer:
        """
        Reads a MIDI file and returns the messages.

        Args:
            file_path (str): Path to the MIDI file.

        Returns:
            List[mido.Message]: A list of MIDI messages.
        """
        self.init_parser()
        # try:
        if self.use_symusic:
            return self.parse_symusic(file_path)
        else:
            return self.parse_mido(file_path)

    def parse_symusic(self, file_path: str) -> SymMusicContainer:
        score = Score(file_path)
        self.sym_music_container.from_symusic(score)
        return self.sym_music_container

    def parse_mido(self, file_path: str) -> SymMusicContainer:
        self.mido_obj = mido.MidiFile(file_path)
        self.sym_music_container.ticks_per_beat = self.mido_obj.ticks_per_beat
        self.process_mido_messages()
        return self.sym_music_container
        # except Exception as e:
        #     print(f"Error reading MIDI file: {e}")
        #     return None

    def process_mido_messages(self):
        """
        Processes MIDI messages.
        """
        for i, track in enumerate(self.mido_obj.tracks):
            note_events = []
            instrument = None
            self._current_instrument_name = ""
            self._accum_time = 0  # reset accumulated time for every track
            for msg in track:
                event = self.process_one_mido_message(msg)
                if event:
                    if isinstance(event, Note):
                        note_events.append(event)
                    elif isinstance(event, Instrument):
                        instrument = event
                    elif isinstance(event, TimeSignature):
                        self.sym_music_container.time_signature_changes.append(event)
                    elif isinstance(event, KeySignature):
                        self.sym_music_container.key_signature_changes.append(event)
                    elif isinstance(event, TempoChange):
                        if event.time == 0:
                            self.sym_music_container.tempo_changes = [event]
                        else:
                            last_tempo = self.sym_music_container.tempo_changes[
                                -1
                            ].tempo
                            if event.tempo == last_tempo:
                                continue
                            else:
                                self.sym_music_container.tempo_changes.append(event)
                    elif isinstance(event, Marker):
                        self.sym_music_container.markers.append(event)

            if instrument and note_events:
                instrument.notes = sorted(note_events, key=lambda x: (x.start, x.pitch))
                self.sym_music_container.instruments.append(instrument)
                self.sym_music_container.max_tick = max(
                    self.sym_music_container.max_tick,
                    max([note.end for note in note_events]),
                )

        if self.verbose:
            # save messages to a file
            if not os.path.exists("midi_messages.txt"):
                with open("midi_messages.txt", "w") as f:
                    for track in self.mido_obj.tracks:
                        for msg in track:
                            f.write(f"{msg}\n")

    def process_one_mido_message(
        self, mido_msg: mido.Message
    ) -> Marker | Note | Instrument | TempoChange | TimeSignature | KeySignature:
        """
        Processes one MIDI message based on its type.

        Args:
            mido_msg (mido.Message): A MIDI message.
        """

        if mido_msg.type == "time_signature":
            return TimeSignature(
                mido_msg.numerator,
                mido_msg.denominator,
                self._update_accum_time(mido_msg.time),
            )
        elif mido_msg.type == "key_signature":
            return KeySignature(mido_msg.key, self._update_accum_time(mido_msg.time))
        elif mido_msg.type == "set_tempo":
            tempo = mido.tempo2bpm(mido_msg.tempo)
            tick = self._update_accum_time(mido_msg.time)
            return TempoChange(tempo, tick)
        elif mido_msg.type == "track_name":
            self._current_instrument_name = mido_msg.name
        elif mido_msg.type == "program_change":
            instrument = Instrument(
                mido_msg.program,
                channel=mido_msg.channel,
                name=self._current_instrument_name,
            )
            return instrument
        elif mido_msg.type == "marker":
            return Marker(mido_msg.text, self._update_accum_time(mido_msg.time))
        elif mido_msg.type == "note_on":
            if mido_msg.velocity == 0:
                # Handle as note_off
                return self._handle_note_off(mido_msg.note, mido_msg.time)
            else:
                return self._handle_note_on(mido_msg.note, mido_msg.velocity, mido_msg.time)
        elif mido_msg.type == "note_off":
            return self._handle_note_off(mido_msg.note, mido_msg.time)
        else:
            self._update_accum_time(mido_msg.time)
        return

    def _handle_note_on(self, pitch, velocity, time):
        """
        Handles a note_on event, allowing overlapping notes.
        """
        if pitch not in self._current_playing_notes:
            self._current_playing_notes[pitch] = []

        # Add the new note to the list of active notes for this pitch
        new_note = Note(
            pitch,
            velocity,
            self._update_accum_time(time),
            None,
        )
        self._current_playing_notes[pitch].append(new_note)
        return None  # New note is just added, no need to return immediately

    def _handle_note_off(self, pitch, time):
        """
        Handles a note_off event, ending the earliest active note for the pitch.
        """
        if pitch in self._current_playing_notes and self._current_playing_notes[pitch]:
            # End the earliest note for this pitch
            note_event = self._current_playing_notes[pitch].pop(0)
            note_event.end = self._update_accum_time(time)
            # Clean up if no more notes are playing for this pitch
            if not self._current_playing_notes[pitch]:
                del self._current_playing_notes[pitch]
            return note_event
        else:
            print(f"No note to turn off: pitch={pitch}, time={time}")


    def _update_accum_time(self, time: int) -> int:
        """
        Updates the accumulated time.

        Args:
            time (int): Time in ticks.

        Returns:
            int: Updated accumulated time.
        """
        self._accum_time += time
        return self._accum_time

    def dump(
        self,
        filename: str | Path | None = None,
        file=None,
        instrument_idx: int | None = None,
    ):

        # comparison function
        def event_compare(event1, event2):
            if event1.time != event2.time:
                return event1.time - event2.time

            # If its two note_on (at the same tick), sort by expected note_off in a FIFO logic
            # This is required in case where the MIDI has notes starting at the same tick and one
            # with a higher duration is listed before one with a shorter one. In this case, the note
            # with the higher duration should come after, otherwise it will be ended first by the
            # following note_off event. Ultimately, as the notes have the same starting time and pitch,
            # the only thing that could be missed is their velocities. This check prevents this.
            if event1.type == event2.type == "note_on":
                return event1.end - event2.end

            secondary_sort = {
                "set_tempo": 1,
                "time_signature": 2,
                "key_signature": 3,
                "marker": 4,
                "lyrics": 5,
                "program_change": 6,
                "pitchwheel": 7,
                "control_change": 8,
                "note_off": 9,
                "note_on": 10,
                "end_of_track": 11,
            }

            if event1.type in secondary_sort and event2.type in secondary_sort:
                return secondary_sort[event1.type] - secondary_sort[event2.type]

            # Events have the same order / position, no change between position
            return 0

        if (filename is None) and (file is None):
            raise OSError("please specify the output.")

        if instrument_idx is None:
            pass
        elif isinstance(instrument_idx, int):
            instrument_idx = [instrument_idx]
        elif isinstance(instrument_idx, list) and len(instrument_idx) == 0:
            pass
        else:
            raise ValueError("Invalid instrument index")

        # Create file
        midi_parsed = mido.MidiFile(
            ticks_per_beat=self.sym_music_container.ticks_per_beat
        )

        # Create track 0 with timing information

        # 1. Time signature
        # add default
        add_ts = True
        ts_list = []
        if self.sym_music_container.time_signature_changes:
            add_ts = (
                min([ts.time for ts in self.sym_music_container.time_signature_changes])
                > 0
            )
        if add_ts:
            ts_list.append(
                mido.MetaMessage("time_signature", time=0, numerator=4, denominator=4)
            )

        # add each
        for ts in self.sym_music_container.time_signature_changes:
            ts_list.append(
                mido.MetaMessage(
                    "time_signature",
                    time=ts.time,
                    numerator=ts.numerator,
                    denominator=ts.denominator,
                )
            )

        # 2. Tempo
        # - add default
        add_t = True
        tempo_list = []
        if self.sym_music_container.tempo_changes:
            add_t = min([t.time for t in self.sym_music_container.tempo_changes]) > 0.0
        if add_t:
            tempo_list.append(
                mido.MetaMessage("set_tempo", time=0, tempo=mido.bpm2tempo(DEFAULT_BPM))
            )

        # - add each
        for t in self.sym_music_container.tempo_changes:
            tempo_list.append(
                mido.MetaMessage(
                    "set_tempo", time=t.time, tempo=mido.bpm2tempo(t.tempo)
                )
            )

        # 3. Markers
        markers_list = []
        for m in self.sym_music_container.markers:
            markers_list.append(mido.MetaMessage("marker", time=m.time, text=m.text))

        # 4. Key
        key_list = []
        for ks in self.sym_music_container.key_signature_changes:
            key_list.append(
                mido.MetaMessage(
                    "key_signature",
                    time=ks.time,
                    key=KEY_NUMBER_TO_MIDO_KEY_NAME[ks.key_number],
                )
            )

        meta_track = ts_list + tempo_list + markers_list + key_list

        # sort
        meta_track.sort(key=functools.cmp_to_key(event_compare))

        # end of meta track
        meta_track.append(
            mido.MetaMessage("end_of_track", time=meta_track[-1].time + 1)
        )
        midi_parsed.tracks.append(meta_track)

        # -- instruments -- #
        channels = list(range(16))
        channels.remove(9)  # for durm
        for cur_idx, instrument in enumerate(self.sym_music_container.instruments):
            if instrument_idx:
                if cur_idx not in instrument_idx:
                    continue

            track = mido.MidiTrack()
            # segment-free
            # track name
            if instrument.name:
                track.append(
                    mido.MetaMessage("track_name", time=0, name=instrument.name)
                )

            # If it's a drum event, we need to set channel to 9
            if instrument.is_drum:
                channel = 9

            # Otherwise, choose a channel from the possible channel list
            else:
                channel = channels[cur_idx % len(channels)]

            # Set the program number
            track.append(
                mido.Message(
                    "program_change",
                    time=0,
                    program=instrument.program,
                    channel=channel,
                )
            )

            # segment-related
            # Add all pitch bend events
            bend_list = []
            for bend in instrument.pitch_bends:
                bend_list.append(
                    mido.Message(
                        "pitchwheel", time=bend.time, channel=channel, pitch=bend.pitch
                    )
                )

            # Add all control change events
            cc_list = []
            if instrument.control_changes:
                for control_change in instrument.control_changes:
                    track.append(
                        mido.Message(
                            "control_change",
                            time=control_change.time,
                            channel=channel,
                            control=control_change.number,
                            value=control_change.value,
                        )
                    )
            else:
                for pedals in instrument.pedals:
                    # append for pedal-on (127)
                    cc_list.append(
                        mido.Message(
                            "control_change",
                            time=pedals.start,
                            channel=channel,
                            control=64,
                            value=127,
                        )
                    )
                    # append for pedal-off (0)
                    cc_list.append(
                        mido.Message(
                            "control_change",
                            time=pedals.end,
                            channel=channel,
                            control=64,
                            value=0,
                        )
                    )

            track += bend_list + cc_list

            # Add all note events
            for note in instrument.notes:
                track.append(
                    mido.Message(
                        "note_on",
                        time=note.start,
                        channel=channel,
                        note=note.pitch,
                        velocity=note.velocity,
                        end=note.end,
                    )
                )
                # Also need a note-off event
                track.append(
                    mido.Message(
                        "note_off",
                        time=note.end,
                        channel=channel,
                        note=note.pitch,
                        velocity=note.velocity,
                    )
                )
            track = sorted(track, key=functools.cmp_to_key(event_compare))

            # Finally, add in an end of track event
            track.append(mido.MetaMessage("end_of_track", time=track[-1].time + 1))
            # Add to the list of output tracks
            midi_parsed.tracks.append(track)

        # Cumulative timing to delta
        for track in midi_parsed.tracks:
            tick = 0
            for event in track:
                event.time -= tick
                tick += event.time

        # Write it out
        if filename:
            midi_parsed.save(filename=filename)
        else:
            midi_parsed.save(file=file)

    def get_tick_to_time_mapping(self) -> np.ndarray:
        return _get_tick_to_second_mapping(
            self.ticks_per_beat, self.max_tick, self.tempo_changes
        )

    def _get_tick_to_second_mapping(
        ticks_per_beat: int, max_tick: int, tempo_changes: list[TempoChange]
    ) -> np.ndarray:
        tick_to_time = np.zeros(max_tick + 1)
        num_tempi = len(tempo_changes)
        acc_time = 0

        for idx in range(num_tempi):
            start_tick = tempo_changes[idx].time
            cur_tempo = tempo_changes[idx].tempo

            # compute tick scale
            seconds_per_beat = 60 / cur_tempo
            seconds_per_tick = seconds_per_beat / float(ticks_per_beat)

            # set end tick of interval
            end_tick = (
                tempo_changes[idx + 1].time if (idx + 1) < num_tempi else max_tick
            )

            # write interval
            ticks = np.arange(end_tick - start_tick + 1)
            tick_to_time[start_tick : end_tick + 1] = (
                acc_time + seconds_per_tick * ticks
            )
            acc_time = tick_to_time[end_tick]
        return tick_to_time
