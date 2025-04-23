from ..parser.container import SymMusicContainer
from ..analysis.utils import get_all_notes
import numpy as np
import pandas as pd
import midisym.csamplers as csamplers


def get_edges(
    sym_music_container: SymMusicContainer,
    exclude_drum: bool = True,
    window_size: float = 0.0,  # in seconds
):

    edg_src = []
    edg_dst = []
    edg_type = []

    all_notes = get_all_notes(sym_music_container, exclude_drum=exclude_drum)
    all_notes = sorted(all_notes, key=lambda x: x.start)

    for i, note in enumerate(all_notes):
        # nbs = neighbors
        for j, nb in enumerate(all_notes):
            # convert seconds to ticks
            from midisym.analysis.utils import time_to_ticks_perf_midi

            window_size_tick = time_to_ticks_perf_midi(
                note,
                window_size,
                sym_music_container.ticks_per_beat,
                sym_music_container.tempo_changes,
            )
            # if i == j:
            #     continue
            if abs(note.start - nb.start) <= window_size_tick:
                # print(f"Note {i} and {j} have the same onset")
                edg_src.append(i)
                edg_dst.append(j)
                edg_type.append("onset")
            if abs(note.start - nb.end) <= window_size_tick:
                # print(f"Note {i} and {j} have the consecutive onset offset")
                edg_src.append(i)
                edg_dst.append(j)
                edg_type.append("consecutive")
            if note.start < nb.start and note.end > nb.start:
                # print(f"Note {i} and {j} have the sustained relationship")
                edg_src.append(i)
                edg_dst.append(j)
                edg_type.append("sustain")

        # adding self-loop
        # edg_src.extend([i] * 3)
        # edg_dst.extend([i] * 3)
        # edg_type.extend(["onset", "sustain"])

    edges = np.array([edg_src, edg_dst])
    edge_etypes = {"onset": 0, "consecutive": 1, "sustain": 2}
    edge_types = np.array([edge_etypes[x] for x in edg_type])
    return edges, edge_types


# ref from the symbolic music graph classification paper, which seems to be slower due to using pandas
# https://github.com/anusfoil/SymRep
def get_edges_np(sym_music_container: SymMusicContainer, exclude_drum: bool = True):

    edg_src = []
    edg_dst = []
    edg_type = []

    all_notes = get_all_notes(sym_music_container, exclude_drum=exclude_drum)
    all_notes = sorted(all_notes, key=lambda x: x.start)

    all_notes_df = pd.DataFrame(
        [(n.start, n.end, n.pitch, n.velocity) for n in all_notes],
        columns=["start", "end", "pitch", "velocity"],
    )

    for i, note in all_notes_df.iterrows():
        # nbs = neighbors
        onset_nbs = all_notes_df[all_notes_df["start"] == note["start"]]
        consec_nbs = all_notes_df[all_notes_df["start"] == note["end"]]
        sustain_nbs = all_notes_df[
            (all_notes_df["start"] < note["start"])
            & (all_notes_df["end"] > note["start"])
        ]

        if not len(onset_nbs):
            # if no onset neighbors, silence edge
            # find the smallest silence gap between current note end and other note start
            silence_gap = pd.Series(all_notes_df["start"] - note["end"])
            silence_gap_min = silence_gap.loc[silence_gap > 0].min()
            silence_nbs = all_notes_df[silence_gap == silence_gap_min]
            edg_src.extend([i] * len(silence_nbs))
            edg_dst.extend(silence_nbs.index.tolist())
            edg_type.extend(["silence"] * len(silence_nbs))

        edg_src.extend(
            [i] * len(onset_nbs) + [i] * len(consec_nbs) + [i] * len(sustain_nbs)
        )
        edg_dst.extend(
            onset_nbs.index.tolist()
            + consec_nbs.index.tolist()
            + sustain_nbs.index.tolist()
        )
        edg_type.extend(
            ["onset"] * len(onset_nbs)
            + ["consecutive"] * len(consec_nbs)
            + ["sustain"] * len(sustain_nbs)
        )

    edges = np.array([edg_src, edg_dst])
    edge_etypes = {"onset": 0, "consecutive": 1, "sustain": 2}
    edge_types = np.array([edge_etypes[x] for x in edg_type])
    return edges, edge_types


def edges_from_note_array(note_array):
    """Turn note_array to list of edges.

    Parameters
    ----------
    note_array : structured array
        The partitura note_array object. Every entry has 5 attributes, i.e. onset_time, note duration, note velocity, voice, id.

    Returns
    -------
    edg_src : np.array
        The edges in the shape of (2, num_edges).
    """

    edg_src = list()
    edg_dst = list()
    edge_type = list()
    for i, x in enumerate(note_array):
        for j in np.where((note_array["onset_div"] == x["onset_div"]))[
            0
        ]:  # & (note_array["id"] != x["id"]))[0]:
            edg_src.append(i)
            edg_dst.append(j)
            edge_type.append(0)

        for j in np.where(
            note_array["onset_div"] == x["onset_div"] + x["duration_div"]
        )[0]:
            edg_src.append(i)
            edg_dst.append(j)
            edge_type.append(1)

        for j in np.where(
            (x["onset_div"] < note_array["onset_div"])
            & (x["onset_div"] + x["duration_div"] > note_array["onset_div"])
        )[0]:
            edg_src.append(i)
            edg_dst.append(j)
            edge_type.append(2)

    end_times = note_array["onset_div"] + note_array["duration_div"]
    for et in np.sort(np.unique(end_times))[:-1]:
        if et not in note_array["onset_div"]:
            scr = np.where(end_times == et)[0]
            diffs = note_array["onset_div"] - et
            tmp = np.where(diffs > 0, diffs, np.inf)
            dst = np.where(tmp == tmp.min())[0]
            for i in scr:
                for j in dst:
                    edg_src.append(i)
                    edg_dst.append(j)
                    edge_type.append(3)

    edges = np.array([edg_src, edg_dst])
    edge_etypes = {0: "onset", 1: "consecutive", 2: "sustain"}
    edge_types = np.array([edge_etypes[x] for x in edge_type])
    return edges, edge_types


def edges_from_graphmuse(
    sym_music_container: SymMusicContainer, exclude_drum: bool = True
):
    all_notes = get_all_notes(sym_music_container, exclude_drum=exclude_drum)
    all_notes,
    note_array = np.array(
        [(n.start, n.end - n.start) for n in all_notes],
        dtype=[("onset_div", "i4"), ("duration_div", "i4")],
    )
    return edges_from_note_array(note_array)


def get_note_array(sym_music_container: SymMusicContainer, exclude_drum: bool = True):
    all_notes = get_all_notes(sym_music_container, exclude_drum=exclude_drum)
    all_notes = sorted(all_notes, key=lambda x: x.start)
    note_array = np.array(
        [(n.start, n.end - n.start, n.pitch, n.velocity) for n in all_notes],
        dtype=[
            ("onset_div", "i4"),
            ("duration_div", "i4"),
            ("pitch", "i4"),
            ("velocity", "i4"),
        ],
    )
    return note_array


def edges_from_csamplers(
    sym_music_container: SymMusicContainer, exclude_drum: bool = True
):
    note_array = get_note_array(sym_music_container, exclude_drum=exclude_drum)
    edges, edge_types = csamplers.compute_edge_list(
        note_array["onset_div"].astype(np.int32),
        note_array["duration_div"].astype(np.int32),
    )
    return edges, edge_types
