from midisym.analysis.melody_extraction import save_extracted_melody
from midisym.analysis.utils import get_all_notes
from midisym.parser.container import Instrument
from midisym.parser.midi import MidiParser
from midisym.parser.renderer import write_audio
import copy


def test_melody_extraction(midi_parser):
    save_extracted_melody(
        midi_parser.sym_music_container, "test.mid", do_write_audio=True
    )
