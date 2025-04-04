#!/bin/bash

folders=(
    # "2025-02-21T00-01-12_temperature_1.5_200k_no_inpainting"
    # "2025-03-04T19-05-17_temperature_1.5_no_inpainting"
    # "2025-02-21T00-01-12_temperature_1.5"
    # "2025-02-21T00-01-12_temperature_1.5_200k"
    # "2025-02-21T00-01-12_temperature_2.0"
    # "2025-03-04T19-05-17"
    # "2025-03-04T19-05-17_temperature_1.5"
    # "2025-03-14T04-20-46_no_chord_in_lead_temp_1.5_no_inpaint"
    # "2025-03-15T17-56-41_w_bridge_temp_1.5_no_inpaint_w_size_5"
    # "2025-03-15T17-57-51_w_bridge_temp_1.5_no_inpaint_w_size_9"
    "2025-03-17T23-39-28_w_bridge_temp_1.5_no_inpaint_w_size_17"
    "2025-03-19T18-08-36_w_bridge_temp_1.5_no_inpaint_w_size_3"
)

for folder in "${folders[@]}"; do
    # uv run python pr_mat_to_midi.py --wandb_id $folder --pr_res 16 
    find results/$folder/midi/ -type f | wc -l 
    scp -P 30000 -r "results/$folder" eunjin@mac-herbie3.kaist.ac.kr:/workspace/eunjin/d3rm/results
    scp -P 30000 -r "results/$folder/midi" eunjin@mac-herbie3.kaist.ac.kr:/workspace/eunjin/PianoArrSamples/POP909/d3rm/$folder
done
