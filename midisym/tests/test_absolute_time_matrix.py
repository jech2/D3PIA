import numpy as np

from midisym.parser.utils import get_ticks_to_seconds_grid
from midisym.analysis.utils import get_all_marker_start_end_time
from midisym.analysis.chord.chord_event import ChordEvent
from midisym.converter.matrix import get_absolute_time_mat

from midisym.converter.constants import N_PITCH, PITCH_OFFSET, PR_RES, ONSET, SUSTAIN, CHORD_OFFSET, MELODY, ARRANGEMENT

def test_absolute_time_mat(analyzed_performance_midi_parser_pop1k7):
    print(analyzed_performance_midi_parser_pop1k7)
    
    sym_obj = analyzed_performance_midi_parser_pop1k7.sym_music_container
    
    piano_rolls, piano_roll_xs, note_infos = get_absolute_time_mat(sym_obj)
                        
    random_idx = None
    n_frames = 313
    
    random_idx = np.random.randint(0, len(piano_roll_xs))

    
    for i, piano_roll in enumerate(piano_rolls):
        import matplotlib.pyplot as plt

        # 전체 피아노 롤
        plt.figure(figsize=(12, 6))  # 캔버스 크기
        plt.imshow(piano_roll.T, aspect="auto", origin="lower", interpolation="nearest")
        plt.xlabel("Time (frame)", fontsize=12)
        plt.ylabel("Pitch", fontsize=12)
        plt.title("Full Piano Roll", fontsize=14)
        plt.savefig(f"tests/sample/absolute_time_mat_{i}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)

        # 랜덤 프레임 피아노 롤
        plt.figure(figsize=(12, 6))  # 캔버스 크기
        plt.imshow(piano_roll[random_idx:random_idx+n_frames].T, aspect="auto", origin="lower", interpolation="nearest")
        plt.xlabel("Time (frame)", fontsize=12)
        plt.ylabel("Pitch", fontsize=12)
        plt.title("Random Frame Piano Roll", fontsize=14)
        plt.savefig(f"tests/sample/absolute_time_mat_{i}_random.png", dpi=300, bbox_inches="tight", pad_inches=0.1)

        
def test_absolute_time_mat_pop2piano(quantized_midi_parser_pop2piano):
    print(quantized_midi_parser_pop2piano)
    
    sym_obj = quantized_midi_parser_pop2piano.sym_music_container
    
    piano_rolls, piano_roll_xs, note_infos = get_absolute_time_mat(sym_obj, add_chord_labels_to_pr=False)
                        
    random_idx = None
    n_frames = 313
    
    random_idx = np.random.randint(0, len(piano_roll_xs))

    
    for i, piano_roll in enumerate(piano_rolls):
        import matplotlib.pyplot as plt

        # 전체 피아노 롤
        plt.figure(figsize=(12, 6))  # 캔버스 크기
        plt.imshow(piano_roll.T, aspect="auto", origin="lower", interpolation="nearest")
        plt.xlabel("Time (frame)", fontsize=12)
        plt.ylabel("Pitch", fontsize=12)
        plt.title("Full Piano Roll", fontsize=14)
        plt.savefig(f"tests/sample/absolute_time_mat_{i}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)

        # 랜덤 프레임 피아노 롤
        plt.figure(figsize=(12, 6))  # 캔버스 크기
        plt.imshow(piano_roll[random_idx:random_idx+n_frames].T, aspect="auto", origin="lower", interpolation="nearest")
        plt.xlabel("Time (frame)", fontsize=12)
        plt.ylabel("Pitch", fontsize=12)
        plt.title("Random Frame Piano Roll", fontsize=14)
        plt.savefig(f"tests/sample/absolute_time_mat_{i}_random.png", dpi=300, bbox_inches="tight", pad_inches=0.1)

        
