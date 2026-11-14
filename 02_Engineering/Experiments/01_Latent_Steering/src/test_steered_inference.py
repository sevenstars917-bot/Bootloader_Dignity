#!/usr/bin/env python3
"""
test_steered_inference.py
Gemma 4 26B-A4B において、第26層 SVD固有ベクトルによるステアリング
（α=0.0 vs α=1.0 vs α=2.0）の思考プロセスへの影響を検証する。
"""
import time
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from steered_model_wrapper import SteeredGemmaModel

MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
VECTOR_PATH = "clarity_eigenvector_layer26.npz"

QUESTION = "バットとボールの合計は110円。バットはボールより100円高い。ボールの値段はいくら？論理的に考えて答えて。"

def run_test(steered_wrapper, model, tokenizer, alpha, prompt):
    print(f"\n=======================================================")
    print(f"🚀 Testing with Layer 26 SVD Eigenvector: Alpha = {alpha:.1f}")
    print(f"=======================================================")
    
    if alpha > 0.0:
        steered_wrapper.set_alpha(alpha)
        steered_wrapper.activate()
    else:
        steered_wrapper.deactivate()
        
    sampler = make_sampler(temp=0.7)
    
    t0 = time.time()
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=250,
        sampler=sampler,
        verbose=True
    )
    elapsed = time.time() - t0
    print(f"\n[Alpha {alpha:.1f}] Time: {elapsed:.2f}s")
    return response

def main():
    print("Loading Gemma 4 26B-A4B...")
    model, tokenizer = load(MODEL_ID)
    steered_wrapper = SteeredGemmaModel(model, vector_path=VECTOR_PATH)
    
    messages = [{"role": "user", "content": QUESTION}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 1. 素の推論 (alpha = 0.0)
    run_test(steered_wrapper, model, tokenizer, 0.0, prompt)
    
    # 2. SVD固有ベクトル ステアリング (alpha = 1.0)
    run_test(steered_wrapper, model, tokenizer, 1.0, prompt)
    
    # 3. SVD固有ベクトル 強ステアリング (alpha = 2.0)
    run_test(steered_wrapper, model, tokenizer, 2.0, prompt)

if __name__ == "__main__":
    main()
