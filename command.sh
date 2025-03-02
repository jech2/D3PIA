
# for pop1k7 
uv run python main_cli.py fit -c ./configs/D3RM_large_cli.yaml

# for pop909
uv run python main_cli_pop909.py fit -c ./configs/D3RM_large_cli_pop909.yaml

## config 파일 내 ckpt_path 수정 후 실행
uv run python main_cli_pop909.py test -c ./logs/2025-01-25T19-24-30/config.yaml

## 인퍼런스 midi 파일로 변환
uv run python pr_mat_to_midi.py --wandb_id 2025-01-25T19-24-30 --pr_res 16 --unit quantize_grid


scp -P 30000 -r ./logs/2025-01-25T19-24-30 eunjin@mac-herbie3.kaist.ac.kr:/workspace/eunjin/d3rm/logs/2025-01-25T19-24-30

scp -P 30000 -r eunjin@mac-herbie3.kaist.ac.kr:/workspace/eunjin/d3rm/data ./data

scp -P 30000 -r eunjin@mac-herbie3.kaist.ac.kr:/workspace/eunjin/Dataset/POP909-Dataset/POP909_processed_polyffusion_split ./POP909_processed_polyffusion_split

scp -P 30000 -r eunjin@mac-herbie3.kaist.ac.kr:/workspace/eunjin/Dataset/POP909-Dataset/POP909_valid_for_ls_inference_polyffusion_split ./POP909_valid_for_ls_inference_polyffusion_split