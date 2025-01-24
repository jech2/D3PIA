"""
modified from JongWook Kim's repository
https://github.com/jongwook/onsets-and-frames/blob/master/onsets_and_frames/dataset.py
"""
import os
from abc import abstractmethod
from pathlib import Path
import json
import csv
import numpy as np
import math
import torch as th
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import soundfile

from transcription.constants import HOP, SR, MAX_MIDI, MIN_MIDI
from transcription.midi import parse_midi, parse_pedal

import numpy as np

from midisym.parser.utils import get_ticks_to_seconds_grid
from midisym.analysis.utils import get_all_marker_start_end_time
from midisym.analysis.chord.chord_event import ChordEvent
from midisym.converter.matrix import get_absolute_time_mat, get_grid_quantized_time_mat

from midisym.converter.constants import N_PITCH, PITCH_OFFSET, PR_RES, ONSET, SUSTAIN, CHORD_OFFSET, POP1k7_MELODY, POP1k7_ARRANGEMENT
from midisym.parser.midi import MidiParser
import random

def uniform_augmentation(arr, width, prob):
    mask = arr > 0
    idx = th.nonzero(mask, as_tuple=True)
    n_change = int(len(idx[0])*prob)
    perm = th.randperm(len(idx[0]))[:n_change]
    idx = [el[perm] for el in idx]
    rand_arr = th.rand((n_change,))
    out_tensor = arr.clone()
    out_tensor[idx] = out_tensor[idx]*(1+(rand_arr*2-1)*width)
    return out_tensor

def onset_time_transform(arr, std, prob, zero_prob):
    mask = arr > 0
    prob_arr = th.rand_like(arr)
    indicator = (prob_arr < prob)
    prob_arr2 = th.randn_like(arr)
    transformed = ~indicator * arr + mask*indicator*arr*(prob_arr2*std + 1)
    zero_prob_arr = th.rand_like(arr)
    zero_indicator = (zero_prob_arr < zero_prob)
    transformed = transformed * ~zero_indicator
    return transformed.to(arr.device)

def vel_transform(arr, var, prob, zero_prob):
    mask = arr > 0
    prob_arr = th.rand_like(arr)
    indicator = (prob_arr < prob)
    prob_arr2 = th.randn_like(arr)
    transformed = ~indicator * arr + mask*indicator*arr*(prob_arr2*var + 1)
    zero_prob_arr = th.rand_like(arr)
    zero_indicator = (zero_prob_arr < zero_prob)
    transformed = transformed * ~zero_indicator
    return transformed.to(arr.device)

class PianoSampleDataset(Dataset):
    def __init__(self, path, groups=None, sample_length=16000*5, seed=1, 
                 random_sample=True, transform=None, delay=1, load_mode='lazy'):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.random_sample = random_sample
        assert all(group in self.available_groups() for group in self.groups)
        self.sample_length = sample_length
        if self.sample_length is not None:
            assert sample_length % HOP == 0
        self.delay = delay
        self.random = np.random.RandomState(seed)
        self.transform = transform

        self.n_keys = MAX_MIDI - MIN_MIDI + 1
        self.data_path = []
        self.max_last = 313

        self.file_list = dict()
        
        self.load_mode=load_mode
        
        # outputs
        self.frame_features = ['label', 'pedal_label', 'velocity', 
                               'last_onset_time', 'last_onset_vel']
        # aggregate files in all groups
        if load_mode == 'ram':
            self.data = []
        for group in groups:
            self.file_list[group] = self.files(group)
            for input_pair in tqdm(self.file_list[group], desc='load files'):
                self.data_path.append(input_pair)

    def __getitem__(self, index):
        '''
        audio-aligned  : - - - - - -|1 1 1 1 1 - - - - 1 1 1 1 1 - - 
                            delay-> | |
        shifted_label  : 0 0 0 0 0 0 0|1 1 1 1 1 0 0 0 0 1 1 1 1 1 0 
        shifted_pedal  : ...
        shifted_vel    : ...
        last_onset_time: 0 0 0 0 0 0 0|1 2 3 4 5 0 0 0 0 1 2 3 4 5 6 
        last_onset_vel : 0 0 0 0 0 0 0|v v v v v 0 0 0 0 V V V V V V 
        '''

        audio_path = self.data_path[index][0]
        result = dict(path=audio_path)
        def th_load_from_memmap(path, dtype, offset=None, shape=None, cast_type=None):
            if cast_type is not None:
                tensor = th.from_numpy(
                    np.memmap(path, dtype=dtype, offset=offset, shape=shape, mode='c').astype(cast_type))
            else:   
                tensor = th.from_numpy(
                    np.memmap(path, dtype=dtype, offset=offset, shape=shape, mode='c'))
            return tensor
            
    
        meta = th.load(audio_path.replace('.flac', '_meta.pt'))
        total_audio_length, total_steps = meta['audio_length'], meta['n_steps']

        if self.sample_length is not None:  # fixed length segmentation
            if self.random_sample: # select from random position
              step_begin = self.random.randint(total_audio_length - self.sample_length) // HOP
            else:
              step_begin = 0
            n_steps = self.sample_length // HOP
            step_end = step_begin + n_steps

            begin = step_begin * HOP
            end = begin + self.sample_length

            result['audio'] = th_load_from_memmap(
                audio_path.replace('.flac', '_audio.npy'), 'int16', 
                begin*np.dtype(np.int16).itemsize, self.sample_length, np.float32)

            for el in self.frame_features:
                if el == 'pedal_label':
                    n_feature = 2
                else:
                    n_feature = 88
                if 'last_onset' in el:
                    cast_type = np.float32
                    if el == 'last_onset_time':
                        save_dtype = np.uint16
                    else:
                        save_dtype = np.uint8
                else:
                    cast_type = np.int64
                    save_dtype = np.uint8
                if step_begin > self.delay - 1:
                    result[el] = th_load_from_memmap(
                        audio_path.replace('.flac', '_{}.npy'.format(el)), save_dtype,
                        (step_begin-self.delay)*n_feature*np.dtype(save_dtype).itemsize, (n_steps+1, n_feature), cast_type)
                else:  # if no previous frames exist
                    result[el] = F.pad(
                        th_load_from_memmap(
                            audio_path.replace('.flac', '_{}.npy'.format(el)), save_dtype,
                            step_begin*n_feature*np.dtype(save_dtype).itemsize, (n_steps, n_feature), cast_type),
                       (0,0,self.delay,0))
            result['time'] = begin / SR 

        else: # use whole sequence at ones; padding
            audio = th_load_from_memmap(
                audio_path.replace('.flac', '_audio.npy'), 'int16', 0, total_audio_length, cast_type=np.float32)
            pad_len = math.ceil(total_audio_length / HOP) * HOP - total_audio_length
            result['audio'] = F.pad(audio, (0, pad_len))
            for el in self.frame_features:
                if el == 'pedal_label':
                    n_feature = 2
                else:
                    n_feature = 88
                if 'last_onset' in el:
                    cast_type = np.float32
                    if el == 'last_onset_time':
                        save_dtype = np.uint16
                    else:
                        save_dtype = np.uint8
                else:
                    cast_type = np.int64
                    save_dtype = np.uint8
                result[el] = F.pad(
                    th_load_from_memmap(
                        audio_path.replace('.flac', '_{}.npy'.format(el)), save_dtype, 0,
                        (total_steps, n_feature), cast_type),
                (0,0,self.delay,0))

        result['audio'] = result['audio'].float().div_(32768.0)

        # make 'last onset features'
        frame_mask = result['label'] > 0
        
        # last_onset_time = last_onset_time.float()
        # last_onset_vel = last_onset_vel.float()
        # last_onset_vel = th.clamp(result['last_onset_vel'], 0, 128) * frame_mask

        last_onset_time = th.clamp(result['last_onset_time'], 0, self.max_last) * frame_mask
        last_onset_vel = result['last_onset_vel'] * frame_mask
        if self.transform:
            last_onset_time = uniform_augmentation(last_onset_time, 0.2, 0.3)
            last_onset_vel = uniform_augmentation(last_onset_vel, 0.2, 0.3)
        
        # result['label'] = result['label'].long()
        # result['pedal_label'] = result['pedal_label'].long()
        # result['velocity'] = result['velocity'].long()
        result['last_onset_time'] = last_onset_time.div_(self.max_last)
        result['last_onset_vel'] = last_onset_vel.div_(128)

        return result

    def sort_by_length(self):
        step_lens = []
        for n in range(len(self)):
            audio_path = self.data_path[n][0]
            meta_path = audio_path.replace('.flac', '_meta.pt')
            step_len = th.load(meta_path)['n_steps']
            step_lens.append(step_len)
        self.data_path = [x for _, x in sorted(zip(step_lens, self.data_path),
                          key=lambda pair: pair[0], reverse=True)]
        

    def __len__(self):
        return len(self.data_path)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """return the names of all available groups"""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """return the list of input files (audio_filename, tsv_filename) for this group"""
        raise NotImplementedError

    def initialize(self):
        for input_pair in tqdm(self.data_path, desc='initialize files:', ncols=100):
            self.load(*input_pair) 
        
    def load(self, audio_path, tsv_path):
        """
        load an audio track and the corresponding labels

        Ex:
        Label          : 0 0 0 0 0|3 4 4 4 4 4 4 1 0 0  0  ..
        last_onset_time: 0 0 0 0 0|1 2 3 4 5 6 7 8 9 10 11 ..
        lase_onset_vel : 0 0 0 0 0|v v v v v v v v v v  v  .. 
        """
        # saved_data_path = audio_path.replace('.flac', '_parsed.pt').replace('.wav', '_parsed.pt')
        # if Path(saved_data_path).exists():
        #     return 
        meta_path = audio_path.replace('.flac', '_meta.pt')
        if Path(meta_path).exists():
            return 

        try:
            audio, sr = soundfile.read(audio_path, dtype='int16')
        except:
            print(audio_path)
        assert sr == SR

        if len(audio.shape) == 2:
            audio = np.mean(audio, axis=-1)

        audio = th.ShortTensor(audio)
        audio_length = len(audio)

        n_keys = MAX_MIDI - MIN_MIDI + 1
        n_steps = (audio_length - 1) // HOP + 1

        label = th.zeros(n_steps, n_keys, dtype=th.uint8)
        velocity = th.zeros(n_steps, n_keys, dtype=th.uint8)
        last_onset_vel = th.zeros(n_steps, n_keys, dtype=th.uint8)
        last_onset_time = th.zeros(n_steps, n_keys, dtype=th.int32)

        midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)
        last_onset_loc = -th.ones(88, dtype=th.int32)
        last_onset_vel_save = -th.ones(88, dtype=th.int32)

        for onset, offset, note, vel in midi:
            left = int(round(onset * SR / HOP))
            onset_right = left + 1
            frame_right = int(round(offset * SR / HOP))
            frame_right = min(n_steps, frame_right)
            offset_right = min(n_steps, frame_right + 1)

            if left > n_steps:
                print(f'Warning: onset after audio ends {audio_length//SR}. {audio_path}. \
                      {onset},{offset},{note},{vel}')
                break

            # off->off :0, on -> off :1, off->onset :2, on -> on :3, on -> onset :4,
            f = int(note) - MIN_MIDI

            if last_onset_loc[f] != -1:
                last_onset_time[last_onset_loc[f]:left, f] = th.arange(1, int(left - last_onset_loc[f] + 1))
                last_onset_vel[last_onset_loc[f]:left, f] = last_onset_vel_save[f] * th.ones(left - last_onset_loc[f])
            last_onset_loc[f] = left
            last_onset_vel_save[f] = vel
            if left > 0 and label[left-1, f] <= 1:
              label[left:onset_right, f] = 2
            elif left == 0:
              label[:onset_right, f] = 2
            else:
              label[left:onset_right, f] = 4
            label[onset_right:frame_right, f] = 3
            label[frame_right:offset_right, f] = 1
            velocity[left:frame_right, f] = vel
        
        for f in range(88):
            if last_onset_loc[f] == -1:
                continue
            last_onset_time[last_onset_loc[f]:, f] = th.arange(1, int(n_steps - last_onset_loc[f] + 1))
            last_onset_vel[last_onset_loc[f]:, f] = last_onset_vel_save[f] * th.ones(n_steps - last_onset_loc[f])
        pedal_tsv_path = tsv_path.replace('.tsv', '_pedal.tsv')
        pedal_label = th.zeros(n_steps, 2, dtype=th.uint8)
        pedal = np.loadtxt(pedal_tsv_path, delimiter='\t', skiprows=1, ndmin=2)

        for onset, offset, pedal_type in pedal:
            if onset >= audio_length / SR:
                break
            left = int(round(onset * SR / HOP))
            onset_right = left + 1
            frame_right = int(round(offset * SR / HOP))
            frame_right = min(n_steps, frame_right)
            offset_right = min(n_steps, frame_right + 1)
            if pedal_type == 0.0:
                type_idx = 0
            elif pedal_type == 2.0:
                type_idx = 1
            # off->off :0, on -> off :1, off->onset :2, on -> on :3, on -> onset :4,
            if left > 0 and pedal_label[left-1, type_idx] <= 1:
              pedal_label[left:onset_right, type_idx] = 2
            elif left == 0:
              pedal_label[:onset_right, type_idx] = 2
            else:
              pedal_label[left:onset_right, type_idx] = 4
            pedal_label[onset_right:frame_right, type_idx] = 3
            pedal_label[frame_right:offset_right, type_idx] = 1

        meta_path = audio_path.replace('.flac', '_meta.pt')
        meta = dict(audio_length=audio_length, n_steps=n_steps)
        th.save(meta, meta_path)
        def save_to_memmap(arr, shape, dtype, path):
            mem = np.memmap(path, dtype=dtype, mode='w+', shape=shape)
            mem[:] = arr[:]
            mem.flush()
            return
        save_to_memmap(audio.numpy(), (audio_length,), 'int16', 
                       audio_path.replace('.flac', '_audio.npy'))
        save_to_memmap(label.numpy(), (n_steps, n_keys), 'uint8',
                       audio_path.replace('.flac', '_label.npy'))
        save_to_memmap(pedal_label.numpy(), (n_steps, 2), 'uint8',
                       audio_path.replace('.flac', '_pedal_label.npy'))
        save_to_memmap(velocity.numpy(), (n_steps, n_keys), 'uint8',
                       audio_path.replace('.flac', '_velocity.npy'))
        save_to_memmap(last_onset_time.numpy().astype(np.uint16), (n_steps, n_keys), 'uint16',
                       audio_path.replace('.flac', '_last_onset_time.npy'))
        save_to_memmap(last_onset_vel.numpy(), (n_steps, n_keys), 'uint8',
                       audio_path.replace('.flac', '_last_onset_vel.npy'))


class MAESTRO(PianoSampleDataset):
    def __init__(self, path='data/MAESTRO', json_file='maestro-v1.0.0.json', groups=None, sequence_length=None, seed=1, 
                 random_sample=True, transform=None):
        self.json_file = json_file
        self.path = Path(path)
        super().__init__(path, groups if groups is not None else ['train'], sequence_length, seed, 
                         random_sample=random_sample, transform=transform)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test', 'debug']

    def files(self, group):
        metadata = json.load(open(os.path.join(self.path, self.json_file)))

        files = sorted([(os.path.join(self.path, row['audio_filename'].replace('.wav', '.flac')),
                            os.path.join(self.path, row['midi_filename'])) for row in metadata if row['split'] == group])

        files = [(Path(audio) if os.path.exists(audio) else audio.replace('.flac', '.wav'), Path(midi)) for audio, midi in files]

        result = []

        for audio_path, midi_path in tqdm(files, desc='Converting midi to tsv group %s' % group, ncols=100):
            tsv_filename = midi_path.with_suffix('.tsv')
            if not tsv_filename.exists():
                midi = parse_midi(midi_path)
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
                pedal = parse_pedal(midi_path)
                np.savetxt(tsv_filename.parent / (tsv_filename.stem + '_pedal.tsv'), pedal, fmt='%.6f', delimiter='\t', header='onset,offset,type')
            else:
                pass
            result.append((str(audio_path), str(tsv_filename)))
        return result


class MAESTRO_V3(PianoSampleDataset):
    def __init__(self, path='data/maestro-v3.0.0', meta_file='maestro-v3.0.0.csv', groups=None, sequence_length=None, seed=1, 
                 random_sample=True, transform=None, load_mode='lazy'):
        self.meta_file = meta_file
        self.path = Path(path)
        super().__init__(self.path, groups if groups is not None else ['test'], sequence_length, seed, random_sample, transform, load_mode=load_mode)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test', 'debug']

    def files(self, group):
        metadata = csv.reader(open(self.path / self.meta_file, 'r',))

        files = sorted([((self.path /row[5]).with_suffix('.flac'), self.path/row[4]) for row in metadata if
                row[2] == group])
        result = []

        for audio_path, midi_path in tqdm(files, desc='Converting midi to tsv group %s' % group, ncols=100):
            tsv_filename = midi_path.with_suffix('.tsv')
            if not tsv_filename.exists():
                midi = parse_midi(midi_path)
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
                pedal = parse_pedal(midi_path)
                np.savetxt(tsv_filename.parent / (tsv_filename.stem + '_pedal.tsv'), pedal, fmt='%.6f', delimiter='\t', header='onset,offset,type')
            else:
                pass
            result.append((str(audio_path), str(tsv_filename)))
        return result
        

class MAPS(PianoSampleDataset):
    def __init__(self, path='data/MAPS', groups=None, sequence_length=None, seed=42, load_mode='ram', random_sample=True, transform=None):
        super().__init__(path, ['ENSTDkAm', 'ENSTDkCl'] if groups in [None, ['test']] else groups, sequence_length, seed, random_sample, transform=transform, load_mode=load_mode)

    @classmethod
    def available_groups(cls):
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    def files(self, group):
        print(group)
        flacs = list((Path(self.path) / 'flac').glob(f'*_{group}.flac'))
        midis = [(Path(self.path) / 'midi') / el.with_suffix('.mid').name for el in flacs]
        tsvs = [(Path(self.path) / 'midi') / el.with_suffix('.tsv').name for el in flacs]

        result = []
        first_tsv = tsvs[0]
        if not os.path.exists(first_tsv):
          for audio_path, midi_path in tqdm(zip(flacs, midis), desc='Converting midi to tsv group %s' % group, ncols=100):
              tsv_filename = midi_path.with_suffix('.tsv')
              midi = parse_midi(midi_path)
              np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
              pedal = parse_pedal(midi_path)
              np.savetxt(str(tsv_filename).replace('.tsv', '_pedal.tsv'), pedal, fmt='%.6f', delimiter='\t', header='onset,offset,type')
              result.append((str(audio_path), str(tsv_filename)))
        else:
          for audio_path, midi_path in zip(flacs, midis):
              tsv_filename = midi_path.with_suffix('.tsv')
              result.append((str(audio_path), str(tsv_filename)))
        return result

class EmotionDataset(PianoSampleDataset):
    def __init__(self, path='data/Emotion', json_file='meta.json', groups=None, sequence_length=None, seed=42, load_mode='ram', random_sample=True, 
                 transform=None):
        self.json_file = json_file
        super().__init__(path, groups if groups is not None else ['test'], sequence_length, seed, random_sample,
                        transform=transform,load_mode=load_mode)
        
    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']

    def files(self, group):
        metadata = json.load(open(os.path.join(self.path, self.json_file)))

        if group == 'debug':
            files = sorted([(os.path.join(self.path, row['flac']),
                             os.path.join(self.path, 'midi', Path(row['flac']).with_suffix('.mid').name)) for row in metadata if
                            row['group'] == 'train'])
            files = files[:50]
        else:
            files = sorted([(os.path.join(self.path, row['flac']),
                             os.path.join(self.path, 'midi', Path(row['flac']).with_suffix('.mid').name)) for row in metadata if
                            row['group'] == group]) 

        result = []

        first_tsv = files[0][1].replace('.midi', '.tsv').replace('.mid', '.tsv')
        if not os.path.exists(first_tsv):
          for audio_path, midi_path in tqdm(files, desc='Converting midi to tsv group %s' % group, ncols=100):
              tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
              midi = parse_midi(midi_path)
              np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
              pedal = parse_pedal(midi_path)
              np.savetxt(tsv_filename.replace('.tsv', '_pedal.tsv'), pedal, fmt='%.6f', delimiter='\t', header='onset,offset,type')
              result.append((audio_path, tsv_filename))
        else:
          for audio_path, midi_path in files:
              tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
              if not os.path.exists(tsv_filename):
                  continue

              result.append((audio_path, tsv_filename))
        return result


class SMD(PianoSampleDataset):
    def __init__(self, path='data/SMD', groups=None, sequence_length=None, seed=42, load_mode='ram', random_sample=True, transform=None):
        super().__init__(path, groups if groups is not None else ['test'], sequence_length,  seed, random_sample,
                        transform=transform,load_mode=load_mode )

    @classmethod
    def available_groups(cls):
        return ['test']

    def files(self, group):
        flacs = list((Path(self.path) / 'midi_audio').glob('*flac'))
        midis = [(Path(self.path) / 'align_midi') / el.with_suffix('.mid').name for el in flacs]
        tsvs = [(Path(self.path) / 'align_midi') / el.with_suffix('.tsv').name for el in flacs]

        result = []
        first_tsv = tsvs[0]
        if not os.path.exists(first_tsv):
          for audio_path, midi_path in tqdm(zip(flacs, midis), desc='Converting midi to tsv group %s' % group, ncols=100):
              tsv_filename = midi_path.with_suffix('.tsv')
              midi = parse_midi(midi_path)
              np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
              pedal = parse_pedal(midi_path)
              np.savetxt(str(tsv_filename).replace('.tsv', '_pedal.tsv'), pedal, fmt='%.6f', delimiter='\t', header='onset,offset,type')
              result.append((str(audio_path), str(tsv_filename)))
        else:
          for audio_path, midi_path in zip(flacs, midis):
              tsv_filename = midi_path.with_suffix('.tsv')
              result.append((str(audio_path), str(tsv_filename)))
        return result

class ViennaCorpus(PianoSampleDataset):
    def __init__(self, path='data/vienna_align', groups=None, sequence_length=None, seed=42, load_mode='ram', random_sample=True, transform=None):
        super().__init__(path, groups if groups is not None else ['test'], sequence_length,  seed, random_sample,
                        transform=transform,load_mode=load_mode )
    @classmethod
    def available_groups(cls):
        return ['test']

    def files(self, group):
        midis = list(Path(self.path).glob('**/*_original.mid'))
        midis = [el for el in midis if 'average' not in el.name]
        files = sorted([(str(el.parent / el.name.replace('_original.mid', '.flac')), str(el)) for el in midis])
        first_onset = dict()
        with open(Path(self.path) / 'first_onset.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                filename = line.split(',')[0]
                onset = float(line.split(',')[1])
                if onset < 0.03:
                    onset = 0.0
                first_onset[filename] = onset
        '''
        midis = list(Path(self.path).glob('**/*_sync.mid'))
        midis = [el for el in midis if 'average' not in el.name and 'original' not in el.name] 
        files = sorted([(str(el.parent / el.name.replace('_sync.mid', '.flac')), str(el)) for el in midis])
        '''

        result = []

        for audio_path, midi_path in tqdm(files, desc='Converting midi to tsv group %s' % group, ncols=100):
            tsv_filename = midi_path.replace('.mid', '.tsv')
            if not Path(tsv_filename).exists():
                midi = parse_midi(midi_path, threshold=21) # special threshold for vienna corpus
                midi_onset = midi[0][0]
                diff = midi_onset - first_onset[Path(audio_path).name] 
                for i in range(len(midi)):
                    midi[i][0] -= diff
                    midi[i][1] -= diff
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
                pedal = parse_pedal(midi_path)
                np.savetxt(tsv_filename.replace('.tsv', '_pedal.tsv'), pedal, fmt='%.6f', delimiter='\t', header='onset,offset,type')
            else:
                pass
            result.append((audio_path, tsv_filename))
        return result
    
class Pop1k7(Dataset):
    def __init__(self, path='data/pop1k7', groups=None, sequence_length=313, seed=1, 
                 random_sample=True, transform=None, load_mode='lazy', pr_res=32, transpose=False, chord_style='chorder'):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.random_sample = random_sample
        print('random sample:', random_sample)
        assert all(group in self.available_groups() for group in self.groups)
        self.sample_length = sequence_length

        self.random = np.random.RandomState(seed)
        self.transform = transform
        self.transpose = transpose

        self.n_keys = MAX_MIDI - MIN_MIDI + 1
        self.data_path = []
        self.max_last = 313

        self.file_list = dict()
        
        self.load_mode=load_mode
        self.pr_res = pr_res
        self.chord_style = chord_style
        # outputs
        self.frame_features = ['label', 'pedal_label', 'velocity', 
                               'last_onset_time', 'last_onset_vel']
        # aggregate files in all groups
        if load_mode == 'ram':
            self.data = []
        if groups == None:
            groups = ['test']
        for group in groups:
            self.file_list[group] = self.files(group)
            for input_pair in tqdm(self.file_list[group], desc='load files'):
                self.data_path.append(input_pair)
        self.initialize()
    
    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test']
    
    def files(self, group):
        split = Path(self.path) / 'custom_split.json'
        metadata = json.load(open(split))
        if group == 'validation':
            group = 'val'
            
        metadata[group] = [item.replace("Data", "data") for item in metadata[group]]
        
        return metadata[group]
    
    def initialize(self):
        for input_path in tqdm(self.data_path, desc='initialize files:', ncols=100):
            self.load(input_path)
        
    def load(self, input_path):
        if isinstance(input_path, Path):
            input_path = str(input_path)
        npy_path = input_path.replace('.mid', f'_piano_rolls_{self.pr_res}.npy')
        # # remove prev npy file
        # if os.path.exists(npy_path):
        #     os.remove(npy_path)
            
        if not os.path.exists(npy_path):
            print('converting midi to piano roll npy')
            midi_parser = MidiParser(input_path, use_symusic=False)
            sym_obj = midi_parser.sym_music_container
            piano_rolls, piano_roll_xs, note_infos = get_absolute_time_mat(sym_obj, pr_res=self.pr_res, chord_style=self.chord_style)
 
            np.save(npy_path, piano_rolls)
            
    def __len__(self):
        return len(self.data_path)

    def transpose_pianoroll(self, np_array, transpose_value):
        if transpose_value == 0:
            return np_array

        # Debugging shapes
        # print("Original shape:", np_array.shape)

        assert np_array.shape[-1] == 88, "The input must have the last dimension as 88 (pitches)."

        # Padding zeros for extension
        zeros_shape = list(np_array.shape)
        zeros_shape[-1] = abs(transpose_value)  # Adjust last dimension for padding
        zeros = np.zeros(zeros_shape)

        if transpose_value > 0:
            # Add zeros before for upward transposition
            extended = np.concatenate((zeros, np_array), axis=-1)
        else:
            # Add zeros after for downward transposition
            extended = np.concatenate((np_array, zeros), axis=-1)

        # Extract the transposed section (88 pitches)
        start_idx = max(-transpose_value, 0)
        extracted = extended[:, :, start_idx : start_idx + 88]

        # print("Extended shape:", extended.shape)
        # print("Extracted shape:", extracted.shape)

        return extracted

    def get_min_max_pitch(self, piano_roll):
        """
        Calculate the minimum and maximum pitch indices in a piano roll.

        Args:
            piano_roll (numpy.ndarray): Piano roll with shape (batch, time, pitch).

        Returns:
            tuple: (min_pitch, max_pitch)
        """
        # Find indices where the piano roll is non-zero (active notes)
        active_pitches = np.where(piano_roll > 0)[-1]

        if active_pitches.size == 0:
            # If no notes are active, return None
            return None, None

        min_pitch = active_pitches.min()
        max_pitch = active_pitches.max()

        return min_pitch, max_pitch

    def __getitem__(self, index):
        piano_roll_fp = self.data_path[index].replace('.mid', f'_piano_rolls_{self.pr_res}.npy')
        
        piano_rolls = np.load(piano_roll_fp)
        
        if self.sample_length is not None:
            random_idx = np.random.randint(0, (piano_rolls.shape[1]-self.sample_length + 1))
            
            cropped_piano_rolls = piano_rolls[:, random_idx:random_idx+self.sample_length]
            
            assert cropped_piano_rolls.shape[1] == self.sample_length
            # assert (cropped_piano_rolls != 0).any()
            
        else:
            cropped_piano_rolls = piano_rolls

        if self.groups == ['train'] and self.transpose and random.random() < 0.5:
            # considering piano roll pitch min max, reset the transpose value
            min_pitch, max_pitch = self.get_min_max_pitch(cropped_piano_rolls)
            if min_pitch is not None:
                min_transpose = max(-min_pitch, -6)
            else:
                min_transpose = -6
            
            if max_pitch is not None:
                max_transpose = min(87 - max_pitch, 6)
            else:
                max_transpose = 6

            transpose_val = random.randint(min_transpose, max_transpose + 1)            

            cropped_piano_rolls = self.transpose_pianoroll(cropped_piano_rolls, transpose_val)


            # for i, piano_roll in enumerate(piano_rolls):
            #     import matplotlib.pyplot as plt
            #     os.makedirs('tests/sample', exist_ok=True)
            #     # 전체 피아노 롤
            #     plt.figure(figsize=(12, 6))  # 캔버스 크기
            #     plt.imshow(piano_roll.T, aspect="auto", origin="lower", interpolation="nearest")
            #     plt.xlabel("Time (frame)", fontsize=12)
            #     plt.ylabel("Pitch", fontsize=12)
            #     plt.title("Full Piano Roll", fontsize=14)
            #     plt.savefig(f"tests/sample/absolute_time_mat_{i}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)

            #     # 랜덤 프레임 피아노 롤
            #     plt.figure(figsize=(12, 6))  # 캔버스 크기
            #     plt.imshow(cropped_piano_rolls[0].T, aspect="auto", origin="lower", interpolation="nearest")
            #     plt.xlabel("Time (frame)", fontsize=12)
            #     plt.ylabel("Pitch", fontsize=12)
            #     plt.title("Random Frame Piano Roll", fontsize=14)
            #     plt.savefig(f"tests/sample/absolute_time_mat_{index}_random.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
            #     plt.close()
            #     # 랜덤 프레임 피아노 롤
            #     plt.figure(figsize=(12, 6))  # 캔버스 크기
            #     plt.imshow(cropped_piano_rolls[1].T, aspect="auto", origin="lower", interpolation="nearest")
            #     plt.xlabel("Time (frame)", fontsize=12)
            #     plt.ylabel("Pitch", fontsize=12)
            #     plt.title("Random Frame Piano Roll", fontsize=14)
            #     plt.savefig(f"tests/sample/absolute_time_mat_{index}_random_ls.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
            #     plt.close()

        return {
            'arrangement': cropped_piano_rolls[0],
            'leadsheet': cropped_piano_rolls[1],
        }
                    
    def sort_by_length(self):
        step_lens = []
        for n in range(len(self)):
            midi_path = self.data_path[n]
            npy_path = midi_path.replace('.mid', f'_piano_rolls_{self.pr_res}.npy')
            step_len = np.load(npy_path).shape[1]
            step_lens.append(step_len)
        self.data_path = [x for _, x in sorted(zip(step_lens, self.data_path),
                          key=lambda pair: pair[0], reverse=True)]

class POP909(Pop1k7):
    def __init__(self, path='data/POP909-Dataset', groups=None, sequence_length=313, seed=1, 
                 random_sample=True, transform=None, load_mode='lazy', pr_res=32, transpose=False, chord_style='pop909'):
        super().__init__(path, groups, sequence_length, seed, random_sample, transform, load_mode, pr_res, transpose, chord_style) 

    @classmethod
    def available_groups(cls):
        return ['train', 'valid', 'test']
    
    def files(self, group):
        split = Path(self.path) / group
        
        mid_files = list(split.glob('**/*.mid'))
        mid_files = [str(el) for el in mid_files]

        return mid_files
    
    def load(self, input_path):
        if isinstance(input_path, Path):
            input_path = str(input_path)
        npy_path = input_path.replace('.mid', f'_piano_rolls_{self.pr_res}.npy')
        # # remove prev npy file
        # if os.path.exists(npy_path):
        #     os.remove(npy_path)
            
        if not os.path.exists(npy_path):
            print('converting midi to piano roll npy')
            midi_parser = MidiParser(input_path, use_symusic=False)
            sym_obj = midi_parser.sym_music_container
            # piano_rolls, piano_roll_xs, note_infos = get_absolute_time_mat(sym_obj, pr_res=self.pr_res, chord_style=self.chord_style)
 
            piano_rolls, _ = get_grid_quantized_time_mat(sym_obj, chord_style='pop909', add_chord_labels_to_pr=True)

            np.save(npy_path, piano_rolls)

    def __getitem__(self, index):
        piano_roll_fp = self.data_path[index].replace('.mid', f'_piano_rolls_{self.pr_res}.npy')
        
        piano_rolls = np.load(piano_roll_fp)
        
        if self.sample_length is not None:
            random_idx = np.random.randint(0, (piano_rolls.shape[1]-self.sample_length + 1) / self.pr_res)
            
            random_idx = random_idx * self.pr_res

            cropped_piano_rolls = piano_rolls[:, random_idx:random_idx+self.sample_length]
            
            assert cropped_piano_rolls.shape[1] == self.sample_length
            # assert (cropped_piano_rolls != 0).any()
            
        else:
            cropped_piano_rolls = piano_rolls

        if self.groups == ['train'] and self.transpose and random.random() < 0.5:
            # considering piano roll pitch min max, reset the transpose value
            min_pitch, max_pitch = self.get_min_max_pitch(cropped_piano_rolls)
            if min_pitch is not None:
                min_transpose = max(-min_pitch, -6)
            else:
                min_transpose = -6
            
            if max_pitch is not None:
                max_transpose = min(87 - max_pitch, 6)
            else:
                max_transpose = 6

            transpose_val = random.randint(min_transpose, max_transpose + 1)            

            cropped_piano_rolls = self.transpose_pianoroll(cropped_piano_rolls, transpose_val)

            # cropped_piano_rolls = cropped_piano_rolls.astype(np.float32)

            # for i, piano_roll in enumerate(piano_rolls):
            #     import matplotlib.pyplot as plt
            #     os.makedirs('tests/sample', exist_ok=True)
            #     # 전체 피아노 롤
            #     plt.figure(figsize=(12, 6))  # 캔버스 크기
            #     plt.imshow(piano_roll.T, aspect="auto", origin="lower", interpolation="nearest")
            #     plt.xlabel("Time (frame)", fontsize=12)
            #     plt.ylabel("Pitch", fontsize=12)
            #     plt.title("Full Piano Roll", fontsize=14)
            #     plt.savefig(f"tests/sample/absolute_time_mat_{i}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)

            #     # 랜덤 프레임 피아노 롤
            #     plt.figure(figsize=(12, 6))  # 캔버스 크기
            #     plt.imshow(cropped_piano_rolls[0].T, aspect="auto", origin="lower", interpolation="nearest")
            #     plt.xlabel("Time (frame)", fontsize=12)
            #     plt.ylabel("Pitch", fontsize=12)
            #     plt.title("Random Frame Piano Roll", fontsize=14)
            #     plt.savefig(f"tests/sample/absolute_time_mat_{index}_random.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
            #     plt.close()
            #     # 랜덤 프레임 피아노 롤
            #     plt.figure(figsize=(12, 6))  # 캔버스 크기
            #     plt.imshow(cropped_piano_rolls[1].T, aspect="auto", origin="lower", interpolation="nearest")
            #     plt.xlabel("Time (frame)", fontsize=12)
            #     plt.ylabel("Pitch", fontsize=12)
            #     plt.title("Random Frame Piano Roll", fontsize=14)
            #     plt.savefig(f"tests/sample/absolute_time_mat_{index}_random_ls.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
            #     plt.close()

        cropped_piano_rolls = th.tensor(cropped_piano_rolls).long()


        return {
            'arrangement': cropped_piano_rolls[0],
            'leadsheet': cropped_piano_rolls[1],
        }