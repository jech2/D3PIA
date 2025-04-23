MELODY = 0
BRIDGE = 1
PIANO = 2

ONSET = 0
OFFSET = 1
FRAME = 2

INPUT_FEAT = 88
QUANTIZE_RESOLUTION = 4
DEFAULT_BEAT = 4
DEFAULT_FRACTION = QUANTIZE_RESOLUTION * DEFAULT_BEAT
CONST_VELOCITY = 80

DEFAULT_TICKS_PER_BEAT = 480
DEFAULT_BPM = 120
MAX_CHANNELS = 16
DRUM_CHANNEL = 9

class ValueRange:
    def __init__(self, min_value, max_value):
        self.min = int(min_value)
        self.max = int(max_value)

    def __contains__(self, value):
        return self.min <= value and value <= self.max
	
    def __repr__(self):
        return f"Value Range of ({self.min}, {self.max})"
    
    def __call__(self):
	    return self.min, self.max

PITCH_RANGE = (0, 127)
MIDI_MAX = 128
PITCH_ID_TO_NAME = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}

PITCH_NAME_TO_ID = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "N": "N",
    None: "N"
}

MAJOR_NAMES = ["M", "Maj", "Major", "maj", "major"]
MINOR_NAMES = ["m", "Min", "Minor", "min", "minor"]

KEY_NUMBER_TO_MIDO_KEY_NAME = [
    "C",
    "Db",
    "D",
    "Eb",
    "E",
    "F",
    "F#",
    "G",
    "Ab",
    "A",
    "Bb",
    "B",
    "Cm",
    "C#m",
    "Dm",
    "D#m",
    "Em",
    "Fm",
    "F#m",
    "Gm",
    "G#m",
    "Am",
    "Bbm",
    "Bm",
]
