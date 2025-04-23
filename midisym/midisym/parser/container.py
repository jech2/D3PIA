"""general container for symbolic music representation

"""

from dataclasses import dataclass
from ..constants import (
    DEFAULT_BPM,
    DEFAULT_TICKS_PER_BEAT,
    DRUM_CHANNEL,
    MAJOR_NAMES,
    MINOR_NAMES,
)
import re
from symusic import Score


@dataclass
class TimeSignature:
    numerator: int = 4
    denominator: int = 4
    time: int = 0

    def from_symusic(self, time_signature):
        self.numerator = time_signature.numerator
        self.denominator = time_signature.denominator
        self.time = time_signature.time
        return self


@dataclass
class KeySignature:
    key_name: str = "C"
    time: int = 0

    def __post_init__(self):
        if self.time < 0:
            raise ValueError(f"{self.time} is not a valid `time` value")

        self.key_number = _key_name_to_key_number(self.key_name)
        if not (0 <= self.key_number < 24):
            raise ValueError(
                f"{self.key_number} is not a valid `key_number` type or value"
            )

    def from_symusic(self, key_signature):
        pass
        return self


@dataclass
class TempoChange:
    tempo: float | int = 0
    time: int = 0

    def from_symusic(self, tempo_change):
        self.tempo = tempo_change.qpm
        self.time = tempo_change.time
        return self


@dataclass
class Marker:
    text: str
    time: int


@dataclass
class Note:
    """baseclass for message as note-level object"""

    pitch: int
    velocity: int
    start: int
    end: int

    def to_immutable(self):
        return (self.pitch, self.velocity, self.start, self.end)

    def __hash__(self):
        # Use the values that are considered in equality comparison
        return hash((self.pitch, self.velocity, self.start, self.end))


class Instrument:
    def __init__(
        self,
        program: int = 0,
        channel: int = 0,
        name: str = "",
        notes: list[Note] | None = None,
        pitch_bends: list | None = None,
        control_changes: list | None = None,
        pedals: list | None = None,
    ):
        self.program = program
        self.is_drum = channel == DRUM_CHANNEL
        self.name = name
        self.notes = notes if notes else []
        self.pitch_bends = pitch_bends if pitch_bends else []
        self.control_changes = control_changes if control_changes else []
        self.pedals = pedals if pedals else []

    @property
    def num_notes(self) -> int:
        return len(self.notes)

    def __str__(self):
        return (
            f"Instrument({self.program}, {self.is_drum}, {self.name}, {self.num_notes})"
        )

    def __eq__(self, other):
        return (
            self.program == other.program
            and self.is_drum == other.is_drum
            and self.notes == other.notes
            and self.pitch_bends == other.pitch_bends
            and self.control_changes == other.control_changes
            and self.pedals == other.pedals
        )

    def add_note_event(
        self,
        pitch: int,
        velocity: int,
        start: int,
        end: int,
    ):
        """
        Adds a note message to the container.

        Args:
            pitch (int): MIDI pitch.
            velocity (int): MIDI velocity.
            start (int): Start time in ticks (cumulated).
            end (int): End time in ticks (cumulated).
        """
        # find the corresponding instrument of program
        self.notes.append(Note(pitch, velocity, start, end))

    def from_symusic(self, track):
        self.program = track.program
        self.is_drum = track.is_drum
        self.name = track.name
        self.notes = [
            Note(
                pitch=note.pitch,
                velocity=note.velocity,
                start=note.start,
                end=note.start + note.duration,
            )
            for note in track.notes
        ]
        return self


# class MetaMessage:


class SymMusicContainer:
    """baseclass for symbolic music container"""

    def __init__(self, ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT):
        self.ticks_per_beat = ticks_per_beat
        self.max_tick = 0
        self.tempo_changes = [TempoChange(DEFAULT_BPM, 0)]
        self.time_signature_changes = []
        self.key_signature_changes = []
        self.markers = []
        self.instruments = []
        self._instrument_programs = []

    @property
    def num_instruments(self) -> int:
        return len(self.instruments)

    def __str__(self):
        output_list = [
            f"ticks per beat: {self.ticks_per_beat}",
            f"max tick: {self.max_tick}",
            f"tempo changes: {len(self.tempo_changes)}",
            f"time sig: {len(self.time_signature_changes)}",
            f"key sig: {len(self.key_signature_changes)}",
            f"markers: {len(self.markers)}",
            f"instruments: {self.num_instruments}",
        ]
        output_str = "\n".join(output_list)
        return output_str

    def __repr__(self):
        return self.__str__()

    def from_symusic(self, score: Score):
        self.tick_per_beat = score.ticks_per_quarter
        self.max_tick = score.end()
        self.tempo_changes = [
            TempoChange().from_symusic(tempo) for tempo in score.tempos
        ]
        self.key_signature_changes = [
            KeySignature().from_symusic(key) for key in score.key_signatures
        ]
        self.time_signature_changes = [
            TimeSignature().from_symusic(time) for time in score.time_signatures
        ]
        self.instruments = [Instrument().from_symusic(track) for track in score.tracks]
        self.markers = [Marker(marker.text, marker.time) for marker in score.markers]
        
        print('from_symusic, currently key signature is not supported. Also, symusic ticks per quarter might not be accurate')
        return self


def _key_name_to_key_number(key_string: str) -> int:
    # Create lists of possible mode names (major or minor)
    # Construct regular expression for matching key
    pattern = re.compile(
        # Start with any of A-G, a-g
        "^(?P<key>[ABCDEFGabcdefg])"
        # Next, look for #, b, or nothing
        "(?P<flatsharp>[#b]?)"
        # Allow for a space between key and mode
        " ?"
        # Next, look for any of the mode strings
        "(?P<mode>(?:(?:"
        +
        # Next, look for any of the major or minor mode strings
        ")|(?:".join(MAJOR_NAMES + MINOR_NAMES)
        + "))?)$"
    )
    # Match provided key string
    result = re.match(pattern, key_string)
    if result is None:
        raise ValueError(f"Supplied key {key_string} is not valid.")
    # Convert result to dictionary
    result = result.groupdict()

    # Map from key string to pitch class number
    key_number = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}[
        result["key"].lower()
    ]
    # Increment or decrement pitch class if a flat or sharp was specified
    if result["flatsharp"]:
        if result["flatsharp"] == "#":
            key_number += 1
        elif result["flatsharp"] == "b":
            key_number -= 1
    # Circle around 12 pitch classes
    key_number = key_number % 12
    # Offset if mode is minor, or the key name is lowercase
    if result["mode"] in MINOR_NAMES or (
        result["key"].islower() and result["mode"] not in MAJOR_NAMES
    ):
        key_number += 12

    return key_number
