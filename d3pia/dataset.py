import random
import os
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from midisym.parser.midi import MidiParser
from midisym.converter.matrix import get_grid_quantized_time_mat, make_grid_quantized_note_prmat

    
class POP909(Dataset):
    def __init__(self, path='data/POP909-Dataset', groups=None, sequence_length=128, 
                 random_sample=True, pr_res=16, transpose=False, chord_style='pop909', bridge_in_arrangement=False, no_chord_in_lead=False):
        self.path = path
        self.groups = groups if groups is not None else self.available_groups()
        self.random_sample = random_sample
        print('random sample:', random_sample)
        assert all(group in self.available_groups() for group in self.groups)
        self.sample_length = sequence_length

        self.transpose = transpose

        self.data_path = []

        self.file_list = dict()
        
        self.pr_res = pr_res
        self.chord_style = chord_style
        
        self.bridge_in_arrangement = bridge_in_arrangement 
        self.no_chord_in_lead = no_chord_in_lead
        
        if groups == None:
            groups = ['test']
        
        for group in groups:
            self.file_list[group] = self.files(group)
            for input_pair in tqdm(self.file_list[group], desc='load files'):
                self.data_path.append(input_pair)
        
        self.initialize()
        

    @classmethod
    def available_groups(cls):
        return ['train', 'valid', 'test']
    
    def files(self, group):
        split = Path(self.path) / group
        
        mid_files = list(split.glob('**/*.mid'))
        mid_files = [str(el) for el in mid_files]

        return mid_files
    
    def initialize(self):
        for input_path in tqdm(self.data_path, desc='initialize files:', ncols=100):
            self.load(input_path)
    
    def load(self, input_path):
        if isinstance(input_path, Path):
            input_path = str(input_path)
        npy_path = input_path.replace('.mid', f'_piano_rolls_{self.pr_res}_with_style_input_arr.npy')

        # # remove prev npy file
        # if os.path.exists(npy_path):
            # os.remove(npy_path)
            
        if not os.path.exists(npy_path):
            print('converting midi to piano roll npy')
            midi_parser = MidiParser(input_path, use_symusic=False)
            sym_obj = midi_parser.sym_music_container
            
            piano_rolls, grid = get_grid_quantized_time_mat(sym_obj, chord_style=self.chord_style, add_chord_labels_to_pr=True)
            prmat = make_grid_quantized_note_prmat(sym_obj, grid, value='duration', do_slicing=False, inst_ids=[2]) # arr only for texture input

            track_mat = piano_rolls['track_mat']
            chord_mat = piano_rolls['chord_mat']
            polydis_chord_mat = piano_rolls['polydis_chord_mat']

            print(len(track_mat), track_mat[0].shape, chord_mat.shape, polydis_chord_mat.shape, prmat.shape)
            assert track_mat[0].shape[0] == prmat.shape[0]

            ret = {
                'track_mat': track_mat, # for each instrument's track
                'chord_mat': chord_mat, # for chord conditioning
                'polydis_chord_mat': polydis_chord_mat, # for polyphonic chord conditioning
                'prmat': prmat, # for style conditioning. time x pitch , value is quantized duration index(16th note unit) prmat[o, p] = d
            }

            np.save(npy_path, ret)
            
    def transpose_pianoroll(self, np_array, transpose_value):
        if transpose_value == 0:
            return np_array

        assert np_array.shape[-1] == 88, "The input must have the last dimension as 88 (pitches)."

        zeros_shape = list(np_array.shape)
        zeros_shape[-1] = abs(transpose_value)
        zeros = np.zeros(zeros_shape)

        if transpose_value > 0:
            extended = np.concatenate((zeros, np_array), axis=-1)
        else:
            extended = np.concatenate((np_array, zeros), axis=-1)

        start_idx = max(-transpose_value, 0)
        extracted = extended[:, :, start_idx : start_idx + 88]

        return extracted

    def get_min_max_pitch(self, piano_roll):
        """
        Calculate the minimum and maximum pitch indices in a piano roll.

        Args:
            piano_roll (numpy.ndarray): Piano roll with shape (batch, time, pitch).

        Returns:
            tuple: (min_pitch, max_pitch)
        """
        active_pitches = np.where(piano_roll > 0)[-1]

        if active_pitches.size == 0:
            return None, None

        min_pitch = active_pitches.min()
        max_pitch = active_pitches.max()

        return min_pitch, max_pitch


    def transpose_polydis_chord_prmat(self, chord_matrix, semitone_shift):
        """
        Transpose the chord_matrix by semitone_shift semitones.
        
        chord_matrix shape: (T, 36)
        - Each row is [root_one_hot(12), chroma(12), bass_one_hot(12)]
        
        semitone_shift: int
        - Number of semitones to transpose (can be positive or negative)
        
        Returns:
        transposed_matrix: np.ndarray of shape (T, 36)
        """
        shift = semitone_shift % 12
        
        transposed_matrix = np.zeros_like(chord_matrix)

        for t in range(chord_matrix.shape[0]):
            row = chord_matrix[t, :]
            
            root_vector = row[0:12]
            chroma_vector = row[12:24]
            bass_vector = row[24:36]
            
            root_shifted = np.roll(root_vector, shift)
            chroma_shifted = np.roll(chroma_vector, shift)
            bass_shifted = np.roll(bass_vector, shift)
            
            transposed_matrix[t, 0:12] = root_shifted
            transposed_matrix[t, 12:24] = chroma_shifted
            transposed_matrix[t, 24:36] = bass_shifted

        return transposed_matrix

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        piano_roll_fp = self.data_path[index].replace('.mid', f'_piano_rolls_{self.pr_res}_with_style_input_arr.npy')
        
        data = np.load(piano_roll_fp, allow_pickle=True).item()

        piano_rolls = np.array(data['track_mat'])
        chord_only_piano_rolls = np.array(data['chord_mat'])
        polydis_chord_prmat = np.array(data['polydis_chord_mat'])
        
        prmat = data['prmat']

        chord_only_piano_rolls = np.expand_dims(chord_only_piano_rolls, axis=0)

        from midisym.constants import QUANTIZE_RESOLUTION
        if self.sample_length is not None and self.sample_length > piano_rolls.shape[1]:
            padding = self.sample_length - piano_rolls.shape[1]
            padding_beats = padding // QUANTIZE_RESOLUTION
            
            piano_rolls = np.pad(piano_rolls, ((0, 0), (0, padding), (0, 0)), mode='constant')
            chord_only_piano_rolls = np.pad(chord_only_piano_rolls, ((0, padding), (0, 0)), mode='constant')
            polydis_chord_prmat = np.pad(polydis_chord_prmat, ((0, padding_beats), (0, 0)), mode='constant')
            prmat = np.pad(prmat, ((0, padding), (0, 0)), mode='constant')
            
            cropped_piano_rolls = piano_rolls
            cropped_chord_only_piano_rolls = chord_only_piano_rolls
            cropped_polydis_chord_prmat = polydis_chord_prmat
        else:
            if self.sample_length is not None:
                random_idx = np.random.randint(0, (piano_rolls.shape[1]-self.sample_length + 1) / self.pr_res) # 16th note unit                
                random_idx = random_idx * self.pr_res
                random_idx_beats = random_idx // QUANTIZE_RESOLUTION
                random_idx_beats_end = (random_idx + self.sample_length) // QUANTIZE_RESOLUTION

                cropped_piano_rolls = piano_rolls[:, random_idx:random_idx+self.sample_length]
                cropped_chord_only_piano_rolls = chord_only_piano_rolls[:, random_idx:random_idx+self.sample_length]
                cropped_polydis_chord_prmat = polydis_chord_prmat[random_idx_beats:random_idx_beats_end]
                
                prmat = prmat[random_idx:random_idx+self.sample_length]

                assert cropped_piano_rolls.shape[1] == cropped_polydis_chord_prmat.shape[0] * QUANTIZE_RESOLUTION
                assert cropped_piano_rolls.shape[1] == cropped_chord_only_piano_rolls.shape[1]
                assert cropped_piano_rolls.shape[1] == self.sample_length
                assert cropped_piano_rolls.shape[1] == prmat.shape[0]
                
            else:
                cropped_piano_rolls = piano_rolls
                cropped_chord_only_piano_rolls = chord_only_piano_rolls
                cropped_polydis_chord_prmat = polydis_chord_prmat

        if self.groups == ['train'] and self.transpose and random.random() < 0.5:
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
            cropped_chord_only_piano_rolls = self.transpose_pianoroll(cropped_chord_only_piano_rolls, transpose_val)
            
            cropped_polydis_chord_prmat = self.transpose_polydis_chord_prmat(cropped_polydis_chord_prmat, transpose_val)
        else:
            transpose_val = 0

        cropped_piano_rolls = torch.tensor(cropped_piano_rolls).long()
        cropped_chord_only_piano_rolls = torch.tensor(cropped_chord_only_piano_rolls).long()
        cropped_polydis_chord_prmat = torch.tensor(cropped_polydis_chord_prmat).float() # to float32
        
        prmat = torch.tensor(prmat).float() # to float32

        assert cropped_piano_rolls.shape[1] == prmat.shape[0]

        cropped_chord_only_piano_rolls = cropped_chord_only_piano_rolls.squeeze()

        if not self.no_chord_in_lead:        
            leadsheet = torch.maximum(cropped_piano_rolls[0], cropped_chord_only_piano_rolls)
        else:
            leadsheet = cropped_piano_rolls[0]
            
        arrangement = torch.maximum(cropped_piano_rolls[0], cropped_piano_rolls[2])
        if self.bridge_in_arrangement:
            arrangement = torch.maximum(arrangement, cropped_piano_rolls[1])

        return {
            'arrangement': arrangement,
            'leadsheet': leadsheet,
            'chord': cropped_chord_only_piano_rolls,
            'polydis_chord': cropped_polydis_chord_prmat,
            'prmat': prmat,
            'fname': self.data_path[index]
        }