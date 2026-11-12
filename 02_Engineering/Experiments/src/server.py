#!/usr/bin/env python3
"""
server.py
MLX Steered Inference Server (OpenAI Compatible API)
Apple M6 / 32GB Mac mini 専用
"""
import os
import sys
import time
import json
import uuid
import argparse
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
from steered_model_wrapper import SteeredGemmaModel

DEFAULT_MODEL_ID = "mlx-community/gemma-4-26b-a4b-it-4bit"
DEFAULT_VECTOR_PATH = "clarity_eigenvector_layer26.npz"

app = FastAPI(title="MLX Steered Inference Server", version="1.0.0")

# グローバル状態
model = None
tokenizer = None
steered_wrapper = None
server_model_name = "gemma-4-26b-a4b-it-4bit-steered"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    alpha: Optional[float] = None  # 動的ステアリング強度オーバーライド

@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": server_model_name,
        "target_layer": steered_wrapper.target_layer if steered_wrapper else None,
        "default_alpha": steered_wrapper.alpha if steered_wrapper else None,
        "is_steered": steered_wrapper.is_active if steered_wrapper else False
    }

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": server_model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "antigravity-m6",
                "permission": [],
                "root": server_model_name,
                "parent": None
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    global model, tokenizer, steered_wrapper
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    # リクエスト個別の alpha があれば一時適用
    req_alpha = req.alpha if req.alpha is not None else steered_wrapper.alpha
    original_alpha = steered_wrapper.alpha
    if req_alpha != original_alpha:
        steered_wrapper.set_alpha(req_alpha)

    # チャットテンプレート適用
    raw_messages = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt = tokenizer.apply_chat_template(raw_messages, tokenize=False, add_generation_prompt=True)
    
    sampler = make_sampler(temp=req.temperature, top_p=req.top_p)
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())
    
    if req.stream:
        # ストリーミング (SSE)
        def event_generator():
            try:
                for chunk in stream_generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=req.max_tokens,
                    sampler=sampler
                ):
                    delta_text = chunk.text
                    chunk_data = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": server_model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": delta_text},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                    
                # 終了チャンク
                final_data = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": server_model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                if req_alpha != original_alpha:
                    steered_wrapper.set_alpha(original_alpha)

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        # 通常一括生成
        try:
            generated_text = ""
            completion_tokens = 0
            for chunk in stream_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=req.max_tokens,
                sampler=sampler
            ):
                generated_text += chunk.text
                completion_tokens += 1
                
            return {
                "id": req_id,
                "object": "chat.completion",
                "created": created_time,
                "model": server_model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(tokenizer.encode(prompt)),
                    "completion_tokens": completion_tokens,
                    "total_tokens": len(tokenizer.encode(prompt)) + completion_tokens
                }
            }
        finally:
            if req_alpha != original_alpha:
                steered_wrapper.set_alpha(original_alpha)

def main():
    global model, tokenizer, steered_wrapper
    
    parser = argparse.ArgumentParser(description="MLX Steered Inference Server")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face Model ID or local path")
    parser.add_argument("--vector", type=str, default=DEFAULT_VECTOR_PATH, help="Path to clarity eigenvector npz")
    parser.add_argument("--alpha", type=float, default=1.0, help="Default steering strength (alpha)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    args = parser.parse_args()
    
    print(f"================================================================")
    print(f"⚡ Starting MLX Steered Server on {args.host}:{args.port}")
    print(f"   Model: {args.model}")
    print(f"   Vector: {args.vector}")
    print(f"   Alpha: {args.alpha}")
    print(f"================================================================")
    
    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"[✓] Model loaded in {time.time() - t0:.2f}s")
    
    steered_wrapper = SteeredGemmaModel(model, vector_path=args.vector, alpha=args.alpha)
    steered_wrapper.activate()
    print(f"[✓] J-Space Steering ACTIVATED on Layer {steered_wrapper.target_layer} (Alpha={steered_wrapper.alpha})")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()

