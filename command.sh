
# for pop1k7 
uv run python main_cli.py fit -c ./configs/D3RM_large_cli.yaml

# for pop909
uv run python main_cli_pop909.py fit -c ./configs/D3RM_large_cli_pop909.yaml

uv run python main_cli_pop909.py test -c ./logs/2025-01-25T19-24-30/config.yaml

uv run python pr_mat_to_midi.py --wandb_id 2025-01-25T19-24-30 --pr_res 16


scp -P 30000 -r ./logs/2025-01-25T19-24-30 eunjin@mac-herbie3.kaist.ac.kr:/workspace/eunjin/d3rm/logs/2025-01-25T19-24-30