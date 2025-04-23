import miditoolkit
from miditoolkit.midi.containers import Note, Instrument
import numpy as np
# from midi2audio import FluidSynth

# pattern is 1, 3, 5, 7, 8
class ChordInterval:
    def __init__(self):
        self.OCTAVE = 12
        self.interval_dict = {
            'M': [0, 4, 7],
            'maj': [0, 4, 7],
            'm': [0, 3, 7],
            'min': [0, 3, 7],
            'minmaj7': [0, 3, 7, 11],
            '7': [0, 4, 7, 10],
            'M7': [0, 4, 7, 11],
            'maj7': [0, 4, 7, 11],
            'm7': [0, 3, 7, 10],
            'min7': [0, 3, 7, 10],
            'dim': [0, 3, 6],
            'o': [0, 3, 6],
            'aug': [0, 4, 8],
            '+': [0, 4, 8],
            'sus4': [0, 5, 7],
            'sus2': [0, 2, 7],
            '6': [0, 4, 7, 9],
            'maj6': [0, 4, 7, 9],
            'm6': [0, 3, 7, 9],
            'min6': [0, 3, 7, 9],
            'mM7': [0, 3, 7, 11],
            'dim7': [0, 3, 6, 9],
            'o7': [0, 3, 6, 9],
            'hdim7': [0, 3, 6, 10],
            '/o7': [0, 3, 6, 10],
            'm7b5': [0, 3, 6, 10],
            '7sus4': [0, 5, 7, 10],
            'sus4(b7)': [0, 5, 7, 10],
            '7sus2': [0, 2, 7, 10],
            'maj9': [0, 4, 7, 10, 14],
            '9': [0, 4, 7, 10, 14],
            'min9': [0, 3, 7, 10, 14],
            'm9': [0, 3, 7, 10, 14],
            'M9': [0, 4, 7, 11, 14],
            '11': [0, 4, 7, 10, 14, 17],
            'm11': [0, 3, 7, 10, 14, 17],
            'min11': [0, 3, 7, 10, 14, 17],
            'M11': [0, 4, 7, 11, 14, 17],
            '13': [0, 4, 7, 10, 14, 17, 21],
            'm13': [0, 3, 7, 10, 14, 17, 21],
            'M13': [0, 4, 7, 11, 14, 17, 21],
            'add9': [0, 4, 7, 14],
            'madd9': [0, 3, 7, 14],
            # 'maj/3'
            # 'maj/5'
            # 'min/5'
            # 'min/b3'
            # 'sus4(b7)'
            # 'min7/b7'
            # 'maj7/7'
            # '7/b7'
            # 'maj7/3'
            # 'maj7/5'
            # 'min7/5'
            # '7/5'
            # '7/3'
        }
        self.pitch_class_dict = {
            'C': 0,
            'C#': 1,
            'Db': 1,
            'D': 2,
            'D#': 3,
            'Eb': 3,
            'E': 4,
            'F': 5,
            'F#': 6,
            'Gb': 6,
            'G': 7,
            'G#': 8,
            'Ab': 8,
            'A': 9,
            'A#': 10,
            'Bb': 10,
            'B': 11,
        }
        self.pitch_dict = {
            value: key for key, value in self.pitch_class_dict.items()
        }
        self.available_tension = {
            # available notes
            'M7': [2, 6, 9],
            'm7': [2, 5],
        }

    def get_available_notes(self, chord_type, chordtone_weight=1.0, tension_weight=1.0):
        pitches = list()
        weights = list()
        if chord_type in self.interval_dict.keys():
            pitches = self.interval_dict[chord_type] + self.available_tension[chord_type]
            weights = [chordtone_weight] * len(self.interval_dict[chord_type]) + [tension_weight] * len(self.available_tension[chord_type])
            return pitches, weights
        else:
            raise ValueError(f"chord_type {chord_type} is not available.") 

    def get_interval(self, chord_type, interval_number):
        if interval_number in [1, 3, 5, 7]:
            idx = int(interval_number / 2)
            if len(self.interval_dict[chord_type]) <= idx:
                return None
            ret_interval = self.interval_dict[chord_type][idx]
        elif interval_number == 8:
            return self.OCTAVE
        # elif interval_number == 12:
        else:
            raise ValueError("Not implemented interval number. Interval number should be in [1, 3, 5, 7, 8]")
        
        return ret_interval
    
    def get_intervals(self, chord_type, interval_numbers):
        # if monotonic increasing
        return [self.get_interval(chord_type, interval_number) for interval_number in interval_numbers]

    def get_chord_pitches(self, root, chord_type):
        pitch_intervals = self.interval_dict[chord_type]
        return [root + interval for interval in pitch_intervals]
    
    # from pitch to midi pitch number
    def pitch_to_midi_pitch(self, pitch_class, octave):
        return self.pitch_class_dict[pitch_class] + (octave - 4) * 12 + 60
    
    def midi_pitch_to_pitch(self, midi_pitch):
        pitch_class = self.pitch_dict[midi_pitch % 12]
        octave = int(midi_pitch / 12) - 1
        return pitch_class, octave
    
    def get_arpeggio_pitches(self, root, chord_type, arpeggio_pattern):
        # root_pitch = self.pitch_to_midi_pitch(root)
        # chord_midi_pitches = ci.get_chord_pitches(root_pitch, chord_type)
        arpeggio_intervals = self.get_intervals(chord_type, arpeggio_pattern)
        arpeggio_pitches = [root + interval for interval in arpeggio_intervals]
        return arpeggio_pitches
        

def pattern_to_midi(pitches, pitch_pattern, rhythm_pattern):
    if len(pitch_pattern) != len(rhythm_pattern):
        raise ValueError('length of pitch_pattern and rhythm_pattern should be same.')

    midi_obj = miditoolkit.midi.parser.MidiFile()
    midi_obj.ticks_per_beat = 480
    notes = list()
    time_offset = 0
    for pis, d_as_beat in zip(pitch_pattern, rhythm_pattern):
        if type(pis) == int:
            pis = [pis]
        for pi in pis:    
            pitch = pitches[pi]
            duration = int(d_as_beat * midi_obj.ticks_per_beat)
            notes.append(Note(pitch=pitch, start=time_offset, end=time_offset + duration, velocity=70))    
        time_offset += duration
    
    inst = Instrument(program=0, is_drum=False, name='piano')
    inst.notes = notes
    midi_obj.instruments.append(inst)
    file_path = 'pattern.mid'
    
    midi_obj.dump(file_path)
    # FluidSynth().midi_to_audio(file_path, file_path.replace('.mid', '.wav'))

def debug_pattern_to_midi():
    pattern_to_midi(pitches=[60, 64, 67], pitch_pattern=[0, 2, 0, 1], rhythm_pattern=[0.5, 0.5, 2.5, 0.5])
    
if __name__ == '__main__':
    debug_pattern_to_midi()
