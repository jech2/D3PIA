uv remove natten
uv remove midisym
uv add ../midisym --no-cache-dir
uv add natten==0.15.1+torch210cu121 -f https://shi-labs.com/natten/wheels
uv pip install -U 'jsonargparse[signatures]>=4.27.7'