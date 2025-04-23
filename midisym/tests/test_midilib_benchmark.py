def call_midiparser():
    from midisym.parser.midi import MidiParser

    file_path = "tests/sample/multi-instrumental.mid"
    midi_parser = MidiParser(file_path)
    sym_obj = midi_parser.sym_music_container
    return sym_obj


def call_midiparser_symusic():
    from midisym.parser.midi import MidiParser

    file_path = "tests/sample/multi-instrumental.mid"
    midi_parser_symusic = MidiParser(file_path, use_symusic=True)
    sym_obj2 = midi_parser_symusic.sym_music_container
    return sym_obj2


def call_mido():
    file_path = "tests/sample/multi-instrumental.mid"
    from mido import MidiFile

    mido_obj = MidiFile(file_path)


def call_symusic():
    file_path = "tests/sample/multi-instrumental.mid"
    from symusic import Score

    score = Score.from_file(file_path)


def call_miditoolkit():
    from miditoolkit.midi.parser import MidiFile

    file_path = "tests/sample/multi-instrumental.mid"
    miditoolkit_obj = MidiFile(file_path)


def call_pretty_midi():
    import pretty_midi

    file_path = "tests/sample/multi-instrumental.mid"
    pretty_midi_obj = pretty_midi.PrettyMIDI(file_path)


def test_benchmark_symusiccontainer(benchmark):
    benchmark(call_midiparser)


def test_benchmark_symusiccontainer_from_symusic(benchmark):
    benchmark(call_midiparser_symusic)


def test_benchmark_mido(benchmark):
    benchmark(call_mido)


def test_benchmark_symusic(benchmark):
    benchmark(call_symusic)


def test_benchmark_miditoolkit(benchmark):
    benchmark(call_miditoolkit)


def test_benchmark_pretty_midi(benchmark):
    benchmark(call_pretty_midi)


def call_partitura():
    import partitura

    file_path = "tests/sample/multi-instrumental.mid"
    partitura_obj = partitura.load_score_midi(file_path)


def test_benchmark_partitura(benchmark):
    benchmark(call_partitura)
