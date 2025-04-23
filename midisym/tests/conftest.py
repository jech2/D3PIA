import pytest


@pytest.fixture
def midi_parser():
    from midisym.parser.midi import MidiParser

    file_path = "tests/sample/multi-instrumental.mid"
    midi_parser = MidiParser(file_path)
    return midi_parser


@pytest.fixture
def analyzed_performance_midi_parser():
    from midisym.parser.midi import MidiParser

    file_path = "./tests/sample/analyzed_performance_piano.mid"
    midi_parser = MidiParser(file_path)
    return midi_parser

@pytest.fixture
def const_tempo_piano_chord_midi_parser():
    from midisym.parser.midi import MidiParser

    file_path = "./tests/sample/pop909_001_including_chord_marker.mid"
    midi_parser = MidiParser(file_path)
    return midi_parser

@pytest.fixture
def analyzed_performance_midi_parser_pop1k7():
    from midisym.parser.midi import MidiParser

    file_path = "./tests/sample/pop1k7_8.mid"
    midi_parser = MidiParser(file_path)
    return midi_parser

@pytest.fixture
def quantized_midi_parser_pop2piano():
    from midisym.parser.midi import MidiParser
    file_path = './tests/sample/YHtysd6SKhA.qmidi.mid'
    midi_parser = MidiParser(file_path)
    return midi_parser