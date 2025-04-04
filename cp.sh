#!/bin/bash

dataset=POP909
folders=(
    "2025-03-06T22-33-08_window_15_batch_16_brg_in_arr"
    "2025-03-14T04-15-08_no_chord_in_lead_use_chord_enc_temp_1.5_no_inpaint"
    "2025-03-15T17-53-17_w_bridge_temp_1.5_no_inpaint_w_size_13"
)

for input_id in "${folders[@]}"
do
    cp -r ./results/${input_id}/midi ../PianoArrSamples/${dataset}/d3rm/${input_id}
done