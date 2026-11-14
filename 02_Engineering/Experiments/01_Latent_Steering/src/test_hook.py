from mlx_lm import load
from mlx_lm.models import gemma4_text
import mlx.core as mx

model_id = "mlx-community/gemma-4-26b-a4b-it-4bit"
model, tokenizer = load(model_id)

captured_act = None
original_call = gemma4_text.DecoderLayer.__call__

def hook_call(self, *args, **kwargs):
    res = original_call(self, *args, **kwargs)
    if getattr(self, "layer_idx", None) == 14:
        global captured_act
        h, kvs, offset = res
        captured_act = h
    return res

gemma4_text.DecoderLayer.__call__ = hook_call

# 推論を1回フォワード
tokens = tokenizer.encode("こんにちは")
_ = model(mx.array([tokens]))
mx.eval(captured_act)

print("Captured successfully!")
print("Act shape:", captured_act.shape)
print("Act norm:", mx.linalg.norm(captured_act).item())

