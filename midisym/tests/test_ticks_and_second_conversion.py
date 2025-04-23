from midisym.parser.utils import get_ticks_to_seconds_grid

def test_ticks_to_seconds_grid(midi_parser):
    grid = get_ticks_to_seconds_grid(midi_parser.sym_music_container)
    print(grid[-50:])
    
def test_ticks_to_seconds_grid_2(analyzed_performance_midi_parser):
    grid = get_ticks_to_seconds_grid(analyzed_performance_midi_parser.sym_music_container)
    print(grid[-50:])