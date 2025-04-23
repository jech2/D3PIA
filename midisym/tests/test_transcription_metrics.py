from midisym.analysis.utils import get_all_notes, time_to_ticks
from midisym.analysis.transcription_metrics import calculate_transcription_measure

from midisym.parser.midi import MidiParser


def test_calculate_transcription_measure():

    parser = MidiParser()
    # load midi file
    midi_infilling_fn = "../2024_ismir/anticipatory_transformer/output_100k/Aqua/Barbie Girl/infilling_input.mid"
    midi_ori_fn = "../2024_ismir/anticipatory_transformer/output_100k/Aqua/Barbie Girl/original.mid"
    midi_inpainted_fn = "../2024_ismir/anticipatory_transformer/output_100k/Aqua/Barbie Girl/inpainted_large.mid"

    midi_obj_ori = parser.parse(midi_ori_fn)
    midi_obj_inpainted = parser.parse(midi_inpainted_fn)

    all_notes_ori = get_all_notes(midi_obj_ori)
    all_notes_inpainted = get_all_notes(midi_obj_inpainted)

    time_seconds = 0.05
    offset_tolerance_ticks = time_to_ticks(
        time_seconds, midi_obj_ori.ticks_per_beat, midi_obj_ori.tempo_changes[0].tempo
    )
    metrics = calculate_transcription_measure(
        all_notes_ori, all_notes_inpainted, offset_tolerance_ticks
    )
    print(metrics)
    metrics2 = calculate_transcription_measure(
        all_notes_ori, all_notes_inpainted, offset_tolerance_ticks
    )
    print(metrics2)
