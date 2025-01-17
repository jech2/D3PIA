import numpy as np
import os

# save as midi
from midisym.parser.midi import MidiParser
from midisym.parser.container import Note
from midisym.parser.container import Instrument
from pathlib import Path

def pianoroll2notes(piano_rolls, ticks_per_beat, pr_res=32):
        # piano roll to midi
        onset_times = [ -1 for _ in range(88) ]
        offset_times = [ -1 for _ in range(88) ]
        
        notes = []
        for t in range(piano_rolls.shape[0]):
            for p in range(piano_rolls.shape[1]):
                if piano_rolls[t, p] == 1:
                    if onset_times[p] == -1:
                        onset_times[p] = t
                    elif offset_times[p] != -1:
                        # seconds to ticks
                        start = int(onset_times[p] * ticks_per_beat * 2 / pr_res)
                        end = int(offset_times[p] * ticks_per_beat * 2 / pr_res)
                        
                        notes.append(Note(
                            pitch=p+21, 
                            velocity=64,
                            start=start,
                            end=end
                        ))
                        onset_times[p] = t
                        offset_times[p] = -1
                        # print(f'pitch: {p+21}, start: {start}, end: {end}')
                elif piano_rolls[t, p] == 2:
                    offset_times[p] = t
                elif piano_rolls[t, p] == 0:
                    if onset_times[p] != -1 and offset_times[p] != -1:
                        # seconds to ticks
                        start = int(onset_times[p] * ticks_per_beat * 2 / pr_res)
                        end = int(offset_times[p] * ticks_per_beat * 2 / pr_res)
                        
                        notes.append(Note(
                            pitch=p+21, 
                            velocity=64,
                            start=start,
                            end=end
                        ))
                        onset_times[p] = -1
                        offset_times[p] = -1
                        # print(f'pitch: {p+21}, start: {start}, end: {end}')
                            
            inst = Instrument()    
            inst.notes = notes
            
        return notes, inst
    
def pianoroll2midi(piano_rolls, leadsheet=None, arrangement=None, out_fp='output.mid', pr_res=32):

    midi = MidiParser()

    print(midi.sym_music_container.ticks_per_beat)

    ticks_per_beat = midi.sym_music_container.ticks_per_beat

    # onset_time_idxs = np.where(piano_rolls == 1)
    # print(onset_time_idxs)

    notes, inst = pianoroll2notes(piano_rolls, ticks_per_beat, pr_res=pr_res)
    midi.sym_music_container.instruments.append(inst)

    if leadsheet is not None:
        _, inst_lead = pianoroll2notes(leadsheet, ticks_per_beat, pr_res=pr_res)
        midi.sym_music_container.instruments.append(inst_lead)
    
    if arrangement is not None:
        _, inst_arr = pianoroll2notes(arrangement, ticks_per_beat, pr_res=pr_res)
        midi.sym_music_container.instruments.append(inst_arr)
    
    midi.dump(out_fp)
    print('midi saved')

def draw_pianoroll(piano_rolls):
    # padding
    n_frames = 313
    n_pitches = 88

    seq_len = piano_rolls.shape[0]
    total_frames_ub = (seq_len // n_frames) * n_frames
    if seq_len % n_frames != 0:
        total_frames_ub += n_frames

        # 패딩 길이 계산
    pad_len = total_frames_ub - seq_len

    # Zero padding 적용 (앞쪽에 0 추가)
    piano_rolls = np.pad(
        piano_rolls, 
        ((0, pad_len), (0, 0)),  # (seq_len, pitch) 방향으로 패딩
        mode='constant',
        constant_values=0  # 패딩 값은 0
    )

    piano_rolls = piano_rolls.reshape(-1, n_frames, n_pitches)

    print(piano_rolls.shape)

    for i, piano_roll in enumerate(piano_rolls):
        import matplotlib.pyplot as plt
        import os
        os.makedirs('tests/samplet', exist_ok=True)
        # 전체 피아노 롤
        plt.figure(figsize=(12, 6))  # 캔버스 크기
        plt.imshow(piano_roll.T, aspect="auto", origin="lower", interpolation="nearest")
        plt.xlabel("Time (frame)", fontsize=12)
        plt.ylabel("Pitch", fontsize=12)
        plt.title("Full Piano Roll", fontsize=14)
        plt.savefig(f"tests/samplet/absolute_time_mat_{i}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='piano roll to midi')
    parser.add_argument('--wandb_id', type=str, help='wandb id')
    parser.add_argument('--pr_res', type=int, default=32, help='piano roll resolution')
    args = parser.parse_args()

    wandb_id = args.wandb_id
    pr_res = args.pr_res

    pr_dir = f'./results/{wandb_id}/'

    pr_dir = Path(pr_dir)
    npz_fps = sorted(list(pr_dir.glob('*.npz')))

    output_dir = pr_dir / 'midi'
    output_dir.mkdir(exist_ok=True)

    for npz_fp in npz_fps:
        if 'test_metrics' in npz_fp.stem:
            continue
        if os.path.exists(output_dir / f'{npz_fp.stem}.mid'):
            print('already exists:', output_dir / f'{npz_fp.stem}.mid')
            continue
        # npz file load
        
        print(f'open from {npz_fp.stem}')
        prs = np.load(npz_fp)
    
        piano_rolls = prs['pred']
        arrangement = prs['arrangement']
        leadsheet = prs['leadsheet']

        print('loaded piano rolls:', piano_rolls.shape)
        

        pianoroll2midi(piano_rolls, leadsheet, arrangement, output_dir / f'{npz_fp.stem}.mid', pr_res=pr_res)