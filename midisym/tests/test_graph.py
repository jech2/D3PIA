from midisym.converter.graph import get_edges, get_edges_np
from midisym.parser.midi import MidiParser
from midisym.analysis.utils import get_all_notes
from collections import Counter
import midisym.csamplers as csamplers
import numpy as np
import pandas as pd
from midisym.converter.graph import edges_from_note_array


def test_edges(midi_parser):

    sym_obj = midi_parser.sym_music_container

    edges = get_edges(sym_obj)
    edges_np = get_edges_np(sym_obj)

    # print(edges)
    # sort each edge
    edges[0] = sorted(edges[0])
    edges[1] = sorted(edges[1])
    edges[2] = sorted(edges[2])

    print("---")

    # print(edges_np)
    # sort each edge
    edges_np[0] = sorted(edges_np[0])
    edges_np[1] = sorted(edges_np[1])
    edges_np[2] = sorted(edges_np[2])

    assert edges.shape == edges_np.shape
    assert Counter(edges[2]) == Counter(edges_np[2])


def test_edges_np():
    midi_parser = MidiParser("tests/sample/multi-instrumental.mid", use_symusic=True)
    sym_obj = midi_parser.sym_music_container
    edges_np = get_edges_np(sym_obj)


def test_edges():
    midi_parser = MidiParser("tests/sample/multi-instrumental.mid", use_symusic=True)
    sym_obj = midi_parser.sym_music_container
    edges = get_edges(sym_obj)


def test_edges_csamplers():
    midi_parser = MidiParser("tests/sample/multi-instrumental.mid", use_symusic=True)

    sym_obj = midi_parser.sym_music_container
    from midisym.converter.graph import edges_from_csamplers

    edges, edge_types = edges_from_csamplers(sym_obj)


def test_edges_graphmuse():
    midi_parser = MidiParser("tests/sample/multi-instrumental.mid", use_symusic=True)

    sym_obj = midi_parser.sym_music_container

    from midisym.converter.graph import edges_from_graphmuse

    edges, edge_types = edges_from_graphmuse(sym_obj)


def test_benchmark_edges(benchmark):
    benchmark(test_edges)


def test_benchmark_edges_np(benchmark):
    benchmark(test_edges_np)


def test_benchmark_edges_csamplers(benchmark):
    benchmark(test_edges_csamplers)


def test_benchmark_edges_graphmuse(benchmark):
    benchmark(test_edges_graphmuse)


def test_edges_csampler():
    import midisym.csamplers as csamplers
    from midisym.parser.midi import MidiParser
    from midisym.analysis.utils import get_all_notes
    import numpy as np
    import pandas as pd
    from collections import Counter
    from midisym.converter.graph import edges_from_note_array

    midi_parser = MidiParser("tests/sample/multi-instrumental.mid", use_symusic=True)

    sym_obj = midi_parser.sym_music_container

    edges = get_edges(sym_obj)

    print(Counter(edges[2]))

    all_notes = get_all_notes(sym_obj, exclude_drum=True)
    all_notes = sorted(all_notes, key=lambda x: x.start)

    note_array = np.array(
        [(n.start, n.end - n.start) for n in all_notes],
        dtype=[("onset_div", "i4"), ("duration_div", "i4")],
    )

    edges, edge_types = csamplers.compute_edge_list(
        note_array["onset_div"].astype(np.int32),
        note_array["duration_div"].astype(np.int32),
    )
    edge_etypes = {0: "onset", 1: "consecutive", 2: "sustain"}

    print(Counter(edge_types))

    edges, edge_types = edges_from_note_array(note_array)
    print(Counter(edge_types))
