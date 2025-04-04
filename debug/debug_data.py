import numpy as np
from pathlib import Path

data_dir = './data/pop1k7/midi_analyzed_melody'
pr_res = 16

piano_roll_fps = Path(data_dir).rglob(f'*{pr_res}.npy')

out_dir = './debug/pr_fig'
out_dir = Path(out_dir)
out_dir.mkdir(exist_ok=True, parents=True)

for pr_fp in piano_roll_fps:
    # import os
    # if os.path.exists(pr_fp):
    #     os.remove(pr_fp)
    
    piano_rolls = np.load(pr_fp)
    piano_rolls_32 = np.load(pr_fp.parent / pr_fp.name.replace(f"_{pr_res}.npy","_32.npy"))
    
    import sys
    sys.path.append('.')
    from pr_mat_to_midi import pianoroll2notes, pianoroll2midi
    print(piano_rolls.dtype, piano_rolls.shape)

    # pianoroll2midi(piano_rolls[0], leadsheet=piano_rolls[1], out_fp=out_dir / f"{pr_fp.stem}.mid", pr_res=pr_res)
    
    # for i, piano_roll in enumerate(piano_rolls):
    #     import matplotlib.pyplot as plt
    #     # 전체 피아노 롤
    #     plt.figure(figsize=(12, 6))  # 캔버스 크기
    #     plt.imshow(piano_roll.T, aspect="auto", origin="lower", interpolation="nearest")
    #     plt.xlabel("Time (frame)", fontsize=12)
    #     plt.ylabel("Pitch", fontsize=12)
    #     plt.title("Full Piano Roll", fontsize=14)
    #     plt.savefig(out_dir / f"absolute_time_mat_{pr_fp.stem}_{i}_{pr_res}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    #     plt.close()
        
    # for i, piano_roll in enumerate(piano_rolls_32):
    #     import matplotlib.pyplot as plt
    #     # 전체 피아노 롤
    #     plt.figure(figsize=(12, 6))
    #     plt.imshow(piano_roll.T, aspect="auto", origin="lower", interpolation="nearest")
    #     plt.xlabel("Time (frame)", fontsize=12)
    #     plt.ylabel("Pitch", fontsize=12)
    #     plt.title("Full Piano Roll", fontsize=14)
    #     plt.savefig(out_dir / f"absolute_time_mat_{pr_fp.stem}_{i}_32.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
        
    #     plt.close()
        
    # # random 313 crop
    # random_pos = np.random.randint(0, len(piano_rolls[0])-313)
    
    # for i, piano_roll in enumerate(piano_rolls):
    #     piano_roll = piano_roll[random_pos:random_pos+313]
    #     import matplotlib.pyplot as plt
    #     # 전체 피아노 롤
    #     plt.figure(figsize=(12, 6))
    #     plt.imshow(piano_roll.T, aspect="auto", origin="lower", interpolation="nearest")
    #     plt.xlabel("Time (frame)", fontsize=12)
    #     plt.ylabel("Pitch", fontsize=12)
    #     plt.title("Full Piano Roll", fontsize=14)
    #     plt.savefig(out_dir / f"absolute_time_mat_{pr_fp.stem}_{i}_{pr_res}_crop.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
    #     plt.close()