from mlx_lm import load

model_id = "mlx-community/gemma-4-26b-a4b-it-4bit"
model, tokenizer = load(model_id)

print("model attributes:", dir(model))
# model.model があるか
submodel = getattr(model, "model", model)
print("submodel type:", type(submodel))
print("submodel attributes:", [k for k in dir(submodel) if not k.startswith("_")])

layer14 = submodel.layers[14]
print("layer 14 type:", type(layer14))
print("layer 14 attributes:", [k for k in dir(layer14) if not k.startswith("_")])

