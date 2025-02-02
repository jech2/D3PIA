
# for pop1k7 
uv run python main_cli.py fit -c ./configs/D3RM_large_cli.yaml

# for pop909
uv run python main_cli_pop909.py fit -c ./configs/D3RM_large_cli_pop909.yaml

uv run python main_cli_pop909.py test -c ./logs/2025-01-25T19-23-52/config.yaml