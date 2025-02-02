import numpy as np
import os

# save as midi
from midisym.parser.midi import MidiParser
from midisym.parser.container import Note
from midisym.parser.container import Instrument
from pathlib import Path

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='piano roll to midi')
    parser.add_argument('--wandb_id', type=str, help='wandb id')
    parser.add_argument('--pr_res', type=int, default=32, help='piano roll resolution')
    parser.add_argument('--unit', type=str, default='quantize_grid', help='unit of time')
    args = parser.parse_args()

    wandb_id = args.wandb_id
    pr_res = args.pr_res
    unit = args.unit

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
        
        from midisym.converter.matrix import pianoroll2midi
        pianoroll2midi(piano_rolls, leadsheet, arrangement, output_dir / f'{npz_fp.stem}.mid', pr_res=pr_res, unit=unit)