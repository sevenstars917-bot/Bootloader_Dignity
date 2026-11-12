import inspect
from mlx_lm.models import gemma4_text

lines = inspect.getsourcelines(gemma4_text.DecoderLayer.__call__)[0]
for line in lines[35:]:
    print(line, end="")
