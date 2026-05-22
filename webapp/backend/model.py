import os
import threading
import time

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Resolve paths relative to repo root (one level above webapp/)
HERE = os.path.dirname(os.path.abspath(__file__))            # webapp/backend
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # repo root
ONNX_PATH = os.path.join(REPO_ROOT, "onnx_export", "multioutput_phobert.onnx")
TOKENIZER_DIR = os.path.join(REPO_ROOT, "onnx_export", "tokenizer_phobert")

MAX_LEN = 256
ID2LABEL = {0: "None", 1: "Positive", 2: "Negative", 3: "Neutral"}
ASPECT_NAMES = ["Chất Liệu", "Kích Cỡ/Form", "Thiết Kế", "Gia Công", "Giá Trị Thực Tế"]
ASPECT_KEYS = ["chat_lieu", "kich_co", "thiet_ke", "gia_cong", "gia_tri"]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class ModelService:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.session: ort.InferenceSession | None = None
        self.tokenizer = None
        self._ready = False
        self._load_seconds: float | None = None
        self._load_error: str | None = None

    @classmethod
    def get(cls) -> "ModelService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def load(self) -> None:
        if self._ready:
            return
        t0 = time.time()
        try:
            if not os.path.exists(ONNX_PATH):
                raise FileNotFoundError(f"ONNX not found: {ONNX_PATH}")
            if not os.path.isdir(TOKENIZER_DIR):
                raise FileNotFoundError(f"Tokenizer dir not found: {TOKENIZER_DIR}")
            self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)
            sess_opts = ort.SessionOptions()
            # FIX: avoid thread oversubscription with uvicorn/anyio threadpool.
            # On a 16-core CPU, 4 ONNX threads is the measured sweet spot.
            sess_opts.intra_op_num_threads = 4
            sess_opts.inter_op_num_threads = 1
            sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                ONNX_PATH,
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            self._ready = True
            self._load_seconds = time.time() - t0
            print(f"[ModelService] Loaded in {self._load_seconds:.1f}s")
        except Exception as e:
            self._load_error = repr(e)
            print(f"[ModelService] Load failed: {e}")

    def is_ready(self) -> bool:
        return self._ready

    def info(self) -> dict:
        return {
            "model": "multioutput-phobert-v2",
            "max_len": MAX_LEN,
            "aspects": ASPECT_NAMES,
            "labels": list(ID2LABEL.values()),
            "load_seconds": self._load_seconds,
            "load_error": self._load_error,
        }

    def predict_one(self, rating: int, content: str) -> dict:
        if not self._ready:
            raise RuntimeError("Model not ready")

        t0 = time.time()
        text = f"Đánh giá {rating} sao. {content}"
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="np",
        )
        logits = self.session.run(
            ["logits"],
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            },
        )[0]  # (1, 5, 4)

        probs = softmax(logits, axis=-1)[0]  # (5, 4)
        pred_ids = probs.argmax(axis=-1)  # (5,)

        predictions = []
        for i in range(5):
            p = probs[i]
            predictions.append({
                "aspect": ASPECT_NAMES[i],
                "aspect_key": ASPECT_KEYS[i],
                "label": ID2LABEL[int(pred_ids[i])],
                "label_id": int(pred_ids[i]),
                "confidence": float(p[pred_ids[i]]),
                "probs": {ID2LABEL[j]: float(p[j]) for j in range(4)},
            })

        return {
            "predictions": predictions,
            "inference_ms": int((time.time() - t0) * 1000),
        }

    def predict_batch(self, ratings: list[int], contents: list[str], batch_size: int = 16):
        """Returns list of dicts: each has {'predictions': [...5 aspects...]}"""
        if not self._ready:
            raise RuntimeError("Model not ready")

        texts = [f"Đánh giá {r} sao. {c}" for r, c in zip(ratings, contents)]
        results = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = self.tokenizer(
                chunk,
                truncation=True,
                max_length=MAX_LEN,
                padding="max_length",
                return_tensors="np",
            )
            logits = self.session.run(
                ["logits"],
                {
                    "input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64),
                },
            )[0]  # (B, 5, 4)
            probs = softmax(logits, axis=-1)  # (B, 5, 4)
            pred_ids = probs.argmax(axis=-1)  # (B, 5)
            for b in range(len(chunk)):
                per_aspect = []
                for i in range(5):
                    per_aspect.append({
                        "aspect": ASPECT_NAMES[i],
                        "label": ID2LABEL[int(pred_ids[b, i])],
                        "confidence": float(probs[b, i, pred_ids[b, i]]),
                    })
                results.append(per_aspect)
        return results
