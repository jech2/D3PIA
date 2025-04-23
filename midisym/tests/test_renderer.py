from midisym.parser.renderer import write_audio


def test_write_audio(midi_parser):
    write_audio(midi_parser, "out.wav")
