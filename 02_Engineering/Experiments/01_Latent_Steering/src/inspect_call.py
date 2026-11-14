from mlx_lm import load
import inspect

model_id = "mlx-community/gemma-4-26b-a4b-it-4bit"
model, _ = load(model_id)

text_model = model.language_model.model
lines = inspect.getsourcelines(type(text_model).__call__)[0]
for line in lines[38:65]:
    print(line, end="")
