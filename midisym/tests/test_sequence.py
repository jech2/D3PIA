from midisym.analysis.grid_quantize import (
    make_grid_quantized_notes,
)
import numpy as np
from midisym.parser.container import TempoChange, Marker, Note

def test_make_sequence(const_tempo_piano_chord_midi_parser):
    sym_obj = const_tempo_piano_chord_midi_parser.sym_music_container
    sym_data_type = "constant tempo MIDI"
    

    all_event_seq = []
    prev_bar = -1
    
    from midisym.converter.sequence.vocabulary import REMILikeCNE
    vocab = REMILikeCNE()
    
    # all_event_seq = vocab.tokenize(sym_obj, sym_data_type, chord_style='pop909')
            
    # print(all_event_seq)

def test_c_and_e_vocabulary():
    import pickle
    with open('../PianoArrangement/Compose_and_Embellish/stage02_embellish/vocab/skyline2midi_vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
        
    _vocab = vocab[0]
    
    # for key, value in _vocab.items():
    #     print(key, value)
    
    from midisym.converter.sequence.vocabulary import REMILikeCNE, find_nearest_bin
    vocab = REMILikeCNE().vocab
    
    # save txt and check the diff
    with open('skyline2midi_vocab.txt', 'w') as f:
        for key, value in _vocab.items():
            f.write(f'{key} {value}\n')
    
    with open('test_vocab.txt', 'w') as f:
        for key, value in vocab.items():
            f.write(f'{key} {value}\n')

    # assert vocab == _vocab

def test_tokenization_analyzed(analyzed_performance_midi_parser):
    sym_obj = analyzed_performance_midi_parser.sym_music_container
    sym_data_type = "analyzed performance MIDI"
    
    from midisym.converter.sequence.vocabulary import REMILikeCNE
    vocab = REMILikeCNE()
    
    q_sym_obj, grid = make_grid_quantized_notes(
    sym_obj=sym_obj,
    sym_data_type=sym_data_type,
    )

    from midisym.parser.midi import MidiParser
    new_parser = MidiParser(sym_music_container=q_sym_obj)
    new_parser.dump('test_out.mid')
    for inst in q_sym_obj.instruments:
        print(len(inst.notes))
        for note in inst.notes:
            print(note)
        
        break
    
    for marker in q_sym_obj.markers:
        print(marker)

    print(q_sym_obj.ticks_per_beat)

    melody_events = vocab.make_inst_events(q_sym_obj, grid, 1, use_tempo_changes=False) # assume melody is 1
    arrangement_events = vocab.make_inst_events(q_sym_obj, grid, 0, use_tempo_changes=True) # assume arrangement is 0
    
    print(len(melody_events) // 16)
    
    print(melody_events)
    
    tokenized_melody, melody_bar_idxs = vocab.tokenize_inst_events(melody_events, 0, chord_style='chorder')
    tokenized_arrangement, arrangement_bar_idxs = vocab.tokenize_inst_events(arrangement_events, 1, chord_style='chorder')
    
    # print()
    
    
    word_melody = [vocab.idx_to_token(tok) for tok in tokenized_melody]
    word_arrangement = [vocab.idx_to_token(tok) for tok in tokenized_arrangement]
    print('-------')
    sym_obj = vocab.word_to_sym_obj(word_arrangement)
    for inst in sym_obj.instruments:
        print(len(inst.notes))
        for note in inst.notes:
            print(note)
        
        break
    
    for marker in sym_obj.markers:
        print(marker)
    
    print(sym_obj.ticks_per_beat)
    
    vocab.word_to_midi(word_melody, out_fp='test_melody_out.mid')
    vocab.word_to_midi(word_arrangement, out_fp='test_arrangement_out.mid')
    
def test_tokenization_analyzed_join(analyzed_performance_midi_parser):
    sym_obj = analyzed_performance_midi_parser.sym_music_container
    sym_data_type = "analyzed performance MIDI -- grid from ticks"
    
    from midisym.converter.sequence.vocabulary import REMILikeCNE
    vocab = REMILikeCNE()
    
    q_sym_obj, grid = make_grid_quantized_notes(
    sym_obj=sym_obj,
    sym_data_type=sym_data_type,
    )

    melody_events = vocab.make_inst_events(q_sym_obj, grid, 1, use_tempo_changes=False) # assume melody is 1
    arrangement_events = vocab.make_inst_events(q_sym_obj, grid, 0, use_tempo_changes=True) # assume arrangement is 0
    
    tokenized_melody, melody_bar_idxs = vocab.tokenize_inst_events(melody_events, 1, chord_style='chorder', use_velocity=False)
    tokenized_arrangement, arrangement_bar_idxs = vocab.tokenize_inst_events(arrangement_events, 0, chord_style='chorder', use_velocity=True)
    
    data = vocab.tokenize_piece(sym_obj, sym_data_type=sym_data_type, chord_style='chorder')
    print(data)
    
    for i, d in enumerate(data['tokenized_piece']):
        if i > 100:
            break
        print(vocab.idx_to_token(d))
    
    
    tokenized_piece, mel_bar_idxs, arr_bar_idxs = data['tokenized_piece'], data['melody_bar_idxs'], data['arrangement_bar_idxs']
    
    vocab.mel_arr_joined_tokens_to_midi(tokenized_piece, mel_bar_idxs=mel_bar_idxs, arr_bar_idxs=arr_bar_idxs, out_fp='test_joined_out.mid')    
    