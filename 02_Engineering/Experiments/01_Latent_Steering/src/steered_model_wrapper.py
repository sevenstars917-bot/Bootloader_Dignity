#!/usr/bin/env python3
"""
steered_model_wrapper.py
MLX (Gemma 4 26B-A4B) の第26層（J空間特異点）残差ストリームへ
SVD第一主成分固有ベクトル（Clarity）を注入する高精度ラッパー。
"""
import os
import numpy as np
import mlx.core as mx
from mlx_lm.models import gemma4_text

DEFAULT_VECTOR_PATH = "clarity_eigenvector_layer26.npz"

class SteeredGemmaModel:
    def __init__(self, model, vector_path=DEFAULT_VECTOR_PATH, alpha=1.0):
        self.model = model
        
        # 相対パスまたは絶対パスの解決
        if not os.path.exists(vector_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(script_dir, vector_path)
            if os.path.exists(candidate):
                vector_path = candidate
            else:
                raise FileNotFoundError(f"Vector file not found: {vector_path}")
                
        data = np.load(vector_path)
        self.vector_np = data["vector"]
        self.target_layer = int(data.get("layer", 26))
        self.alpha = float(alpha)
        
        # MLXテンソルに変換 (bfloat16)
        self.vector_mx = mx.array(self.vector_np).astype(mx.bfloat16)
        
        self.is_active = False
        self._original_call = gemma4_text.DecoderLayer.__call__
        
    def activate(self):
        """ステアリングを有効化（フォワードパス時にテンソル加算）"""
        if self.is_active:
            return
            
        original_call = self._original_call
        vector = self.vector_mx
        target_layer = self.target_layer
        
        def steered_call(self_layer, x, *args, **kwargs):
            res = original_call(self_layer, x, *args, **kwargs)
            if getattr(self_layer, "layer_idx", None) == target_layer:
                h, kvs, offset = res
                # h: [batch, seq_len, 2816]
                if self.alpha != 0.0:
                    h = h + (self.alpha * vector)
                return (h, kvs, offset)
            return res
            
        gemma4_text.DecoderLayer.__call__ = steered_call
        self.is_active = True
        
    def deactivate(self):
        """ステアリングを無効化（素の推論へ復元）"""
        if not self.is_active:
            return
        gemma4_text.DecoderLayer.__call__ = self._original_call
        self.is_active = False
        
    def set_alpha(self, alpha: float):
        """ステアリング強度を変更"""
        self.alpha = float(alpha)

    def __enter__(self):
        self.activate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.deactivate()
