def test_check_valid_midi_processing(midi_parser):
    assert (
        len(midi_parser._current_playing_notes) == 0
    ), "Current playing notes should be empty"

    for instrument in midi_parser.sym_music_container.instruments:
        assert instrument.num_notes > 0, "No note events found"

        for note in instrument.notes:
            assert note.end > note.start, "Note end should be greater than start"
            # print(note)

    for tempo_change in midi_parser.sym_music_container.tempo_changes:
        assert tempo_change.tempo > 0, "Tempo should be greater than 0"

    # dump test
    midi_parser.dump("out_midi.mid")


def test_keysignature():
    from midisym.parser.container import KeySignature

    k_a = KeySignature("A", 0)
    k_b = KeySignature("B", 0)
    k_a2 = KeySignature("A", 0)

    assert k_a == k_a2
    assert k_a != k_b


def test_note():
    from midisym.parser.container import Note

    n_a = Note(60, 100, 0, 100)
    n_b = Note(61, 100, 0, 100)
    n_a2 = Note(60, 100, 0, 100)

    assert n_a == n_a2
    assert n_a != n_b


def test_instruments():
    # load midi file
    from midisym.parser.midi import MidiParser

    midi_fn1 = "../2024_ismir/anticipatory_transformer/output_100k/Aqua/Barbie Girl/infilling_input.mid"
    midi_fn2 = "../2024_ismir/anticipatory_transformer/output_100k/Aqua/Barbie Girl/original.mid"
    parser = MidiParser()
    midi_obj1 = parser.parse(midi_fn1)
    midi_obj2 = parser.parse(midi_fn2)

    assert midi_obj1.instruments[0] == midi_obj1.instruments[0]
    assert midi_obj1.instruments[0] != midi_obj1.instruments[1]


def test_instruments_2():
    from midisym.parser.container import Note, Instrument
    from midisym.analysis.utils import is_same_inst

    n1 = Note(60, 100, 0, 100)
    n2 = Note(61, 100, 0, 100)
    n3 = Note(62, 100, 0, 100)
    n4 = Note(63, 100, 0, 100)
    n5 = Note(64, 100, 0, 100)

    i1 = Instrument(0)
    i1.notes = [n1, n2, n3]

    i2 = Instrument(0)
    i2.notes = [n1, n2, n3]

    i3 = Instrument(0)
    i3.notes = [n1, n2, n3, n4]

    i4 = Instrument(0)
    i4.notes = [n1, n2, n5]

    assert i1 == i2
    assert i1 != i3
    assert i1 != i4
    assert i3 != i4

    assert is_same_inst(i1, i3)
    assert not is_same_inst(i2, i4)


def test_note_2():
    from midisym.parser.container import Note

    n1 = Note(60, 100, 0, 100)
    n2 = Note(61, 100, 0, 100)
    n3 = Note(62, 100, 0, 100)
    n4 = Note(63, 100, 0, 100)
    n5 = Note(64, 100, 0, 100)
    n6 = Note(65, 100, 0, 100)

    notes_a = [n1, n2, n5, n6]
    notes_b = [n1, n2, n3, n4]
    notes_c = [n1, n2, n3, n4, n5, n6]

    notes_a_set = set(notes_a)
    notes_b_set = set(notes_b)
    notes_c_set = set(notes_c)

    notes_inter = notes_a_set.intersection(notes_b_set)
    # 차집합도 고려하자
    notes_diff = notes_a_set - notes_b_set

    assert notes_diff == notes_c_set - notes_b_set


def test_symusiccontainer_from_symusic():
    from midisym.parser.midi import MidiParser

    # file_path = "tests/sample/multi-instrumental.mid"
    file_path = "./tests/sample/pop909_001_including_chord_marker.mid"
    midi_parser = MidiParser(file_path)
    sym_obj = midi_parser.sym_music_container

    midi_parser_symusic = MidiParser(file_path, use_symusic=True)
    sym_obj2 = midi_parser_symusic.sym_music_container

    assert sym_obj.instruments == sym_obj.instruments

    assert sym_obj.tempo_changes == sym_obj2.tempo_changes
    assert sym_obj.time_signature_changes == sym_obj2.time_signature_changes
    assert sym_obj.markers == sym_obj2.markers
    print(sym_obj.markers)
    print(sym_obj2.markers)
    # TODO:: key_signature_changes

def test_piast_sample():
    from midisym.parser.midi import MidiParser
    file_path = './tests/sample/piast_example__asc-c5M1k.mid'
    midi_parser = MidiParser(file_path)
    sym_obj = midi_parser.sym_music_container

    for inst in sym_obj.instruments:
        print(inst.program)