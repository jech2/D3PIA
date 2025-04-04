from midisym.parser.midi import MidiParser
from midisym.converter.matrix import make_grid_quantized_notes, make_grid_quantized_note_prmat

path = './reference_arrangement_style/8bars/332.mid'

midi_parser = MidiParser(path, use_symusic=True)
sym_obj = midi_parser.sym_music_container
# piano_rolls, piano_roll_xs, note_infos = get_absolute_time_mat(sym_obj, pr_res=self.pr_res, chord_style=self.chord_style)

print(sym_obj)

# sym_obj, grid = make_grid_quantized_notes(
#     sym_obj=sym_obj,
#     sym_data_type="analyzed performance MIDI -- grid from ticks",
# )
# prmat = make_grid_quantized_note_prmat(sym_obj, grid, value='duration', do_slicing=False)
