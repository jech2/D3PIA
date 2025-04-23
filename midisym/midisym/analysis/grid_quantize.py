from ..parser.container import SymMusicContainer, TempoChange
import numpy as np
import copy

def get_grid_from_constant_tempo(
    sym_obj: SymMusicContainer, quantize_resolution: int = 4
) -> np.array:
    # get the grid
    grid_res = sym_obj.ticks_per_beat // quantize_resolution
    
    # 한 마디 단위로 반올림한 grid
    bar_res = sym_obj.ticks_per_beat * 4
    last_grid_time = np.ceil(sym_obj.max_tick / bar_res) * bar_res
    grid = np.arange(0, last_grid_time, sym_obj.ticks_per_beat // quantize_resolution)
    # grid = np.append(grid, last_grid_time)
    grid = np.unique(grid)

    return grid


def get_grid_from_tempo_changes(
    sym_obj: SymMusicContainer, quantize_resolution: int = 4
) -> np.array:
    if len(sym_obj.tempo_changes) == 1:
        raise ValueError("It seems the tempo is constant. Try get_grid_constant_tempo()")
    # get all tempo changes event and make total grid of variable time length
    db_grid = [tc.time for tc in sym_obj.tempo_changes]
    # 각각의 db_grid 아이템 사이를 quantize_resolution으로 나누어서 grid를 만든다.
    grid = []
    for i in range(len(db_grid) - 1):
        step = (db_grid[i + 1] - db_grid[i]) / quantize_resolution

        grid.extend(
            np.arange(
                db_grid[i],
                db_grid[i + 1],
                step,
                dtype=int,
            )
        )
    grid.append(db_grid[-1])
    # for the last tempo change, add the rest of the grid
    # use the last step as the step
    # step = (sym_obj.max_tick - db_grid[-1]) / quantize_resolution
    grid.extend(
        np.arange(
            db_grid[-1],
            sym_obj.max_tick,
            step,
            dtype=int,
        )
    )
    grid.append(sym_obj.max_tick)
    # make sure that there is no duplicate
    grid = np.unique(grid)

    return grid


def make_grid_quantized_notes(
    sym_obj: SymMusicContainer,
    sym_data_type: str = "constant tempo MIDI",
    quantize_resolution: int = 4,
    time_signature: tuple[int, int] = (4, 4),
    sum_grid: bool = False,
):

    if time_signature != (4, 4):
        raise NotImplementedError("Only 4/4 time signature is supported now")

    # get the ticks per beat
    if sym_data_type == "constant tempo MIDI":
        # ticks_per_beat / 4 as the quantize resolution
        grid = get_grid_from_constant_tempo(sym_obj, quantize_resolution=4)
        print("Currently constant tempo is supported with 4/4 time signature only")
    elif sym_data_type == "analyzed performance MIDI":
        # get the grid
        if sum_grid:
            grid = get_grid_from_tempo_changes(sym_obj, quantize_resolution=4)
            grid2 = get_grid_from_tempo_changes(sym_obj, quantize_resolution=3)
            # grid and grid2 sum
            grid = np.unique(np.concatenate((grid, grid2)))
        else:    
            grid = get_grid_from_tempo_changes(sym_obj, quantize_resolution=4)

        padding_length = (-len(grid)) % (quantize_resolution * 4)
        if padding_length > 0:
            padding_ticks = np.arange(grid[-1] + sym_obj.ticks_per_beat, grid[-1] + (padding_length + 1) * sym_obj.ticks_per_beat, sym_obj.ticks_per_beat)
            grid = np.concatenate((grid, padding_ticks))
            
    elif sym_data_type == "analyzed performance MIDI -- grid from ticks":
        # Calculate step size and initial grid
        step_size = sym_obj.ticks_per_beat // quantize_resolution
        grid = np.arange(0, sym_obj.max_tick, step_size)

        # Pad the grid to ensure the length is a multiple of quantize_resolution * 4
        padding_length = (-len(grid)) % (quantize_resolution * 4)
        if padding_length > 0:
            padding_ticks = np.arange(grid[-1] + step_size, grid[-1] + (padding_length + 1) * step_size, step_size)
            grid = np.concatenate((grid, padding_ticks))
            
    else:
        raise NotImplementedError(
            f"Currently only constant tempo MIDI and analyzed performance MIDI are supported, got {sym_data_type}"
        )
        
    q_sym_obj = copy.deepcopy(sym_obj)
    # change the note start and end to the nearest grid
    for inst in q_sym_obj.instruments:
        new_notes = []
        for note in inst.notes:
            # print('before', note.start, note.end)
            note.start = int(min(grid, key=lambda x: abs(x - note.start)))
            note.end = int(min(grid, key=lambda x: abs(x - note.end)))
            if note.start == note.end:
                if note.start == grid[-1]:
                    note.start = grid[-2]
                else:
                    note.end = grid[np.where(grid == note.start)[0][0] + 1]
            new_notes.append(note)
            # print('after', note.start, note.end)
        inst.notes = new_notes
    
    # change the chord events into the nearest grid
    new_markers = []
    for marker in q_sym_obj.markers:
        marker.time = int(min(grid, key=lambda x: abs(x - marker.time)))
        new_markers.append(marker)
    q_sym_obj.markers = new_markers
    
    # tempo
    new_tempo_changes = []
    for tc in q_sym_obj.tempo_changes:
        tc.time = int(min(grid, key=lambda x: abs(x - tc.time)))
        new_tempo_changes.append(tc)
    q_sym_obj.tempo_changes = new_tempo_changes
    
    return q_sym_obj, grid