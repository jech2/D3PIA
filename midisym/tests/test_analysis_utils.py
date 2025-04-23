def test_find_matching_inst():

    # load midi file
    midi_infilling_fn = "../2024_ismir/anticipatory_transformer/output_100k/Aqua/Barbie Girl/infilling_input.mid"
    midi_ori_fn = "../2024_ismir/anticipatory_transformer/output_100k/Aqua/Barbie Girl/original.mid"
    midi_inpainted_fn = "../2024_ismir/anticipatory_transformer/output_100k/Aqua/Barbie Girl/inpainted_large.mid"

    from midisym.analysis.utils import check_exact_match_note_rate_fn

    ret = check_exact_match_note_rate_fn(
        midi_infilling_fn, midi_ori_fn, midi_inpainted_fn
    )

    # print(ret)


def test_get_all_notes(midi_parser):
    from midisym.analysis.utils import get_all_notes

    # load midi file
    all_notes = get_all_notes(midi_parser.sym_music_container)

    # print("all_notes", all_notes)
    # print("len(all_notes)", len(all_notes))


def test_get_inpainted_pos_notes(midi_parser):
    from midisym.analysis.utils import get_inpainted_pos_notes
    from midisym.parser.midi import MidiParser

    parser = MidiParser()
    # load midi file
    midi_infilling_fn = "../2024_ismir/anticipatory_transformer/output_100k/Aqua/Barbie Girl/infilling_input.mid"
    midi_ori_fn = "../2024_ismir/anticipatory_transformer/output_100k/Aqua/Barbie Girl/original.mid"
    midi_inpainted_fn = "../2024_ismir/anticipatory_transformer/output_100k/Aqua/Barbie Girl/inpainted_large.mid"

    midi_obj_infilling = parser.parse(midi_infilling_fn)
    midi_obj_ori = parser.parse(midi_ori_fn)
    midi_obj_inpainted = parser.parse(midi_inpainted_fn)
    notes = get_inpainted_pos_notes(midi_obj_ori, midi_obj_infilling)
    # print(notes)
