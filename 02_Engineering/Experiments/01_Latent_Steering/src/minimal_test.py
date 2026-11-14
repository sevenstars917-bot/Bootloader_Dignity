import time
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model_id = "mlx-community/gemma-4-26b-a4b-it-4bit"
print(f"Loading {model_id}...")
t0 = time.time()
model, tokenizer = load(model_id)
print(f"Model loaded in {time.time() - t0:.2f}s")

messages = [
    {"role": "user", "content": "こんにちは！自己紹介を1行でお願いします。"}
]

# tokenize=False で文字列プロンプトとして取得
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"Formatted prompt:\n{repr(prompt)}")

sampler = make_sampler(temp=0.7)

print("\n--- Generating Response ---")
t0 = time.time()
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=100,
    sampler=sampler,
    verbose=True
)
print(f"\nGeneration completed in {time.time() - t0:.2f}s")
