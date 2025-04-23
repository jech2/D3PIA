
uv run python main_cli.py fit -c ./configs/D3PIA_default.yaml

## config 파일 내 ckpt_path 수정 후 실행
uv run python main_cli.py test -c ./logs/2025-01-25T19-24-30/config.yaml

## 인퍼런스 midi 파일로 변환
uv run python pr_mat_to_midi.py --wandb_id 2025-01-25T19-24-30
