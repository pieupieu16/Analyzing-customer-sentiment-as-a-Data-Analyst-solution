import time
import statistics
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

print("Loading model...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained("../onnx_export/tokenizer_phobert", use_fast=True)
opts = ort.SessionOptions()
print(f"  threads intra/inter: {opts.intra_op_num_threads}/{opts.inter_op_num_threads}")
sess = ort.InferenceSession(
    "../onnx_export/multioutput_phobert.onnx",
    sess_options=opts,
    providers=["CPUExecutionProvider"],
)
print(f"Loaded in {time.time()-t0:.1f}s\n")

text = "Đánh giá 5 sao. quần kaki chất vải rất tốt, mặc đứng form, đường may chắc chắn"
enc = tok(text, truncation=True, max_length=256, padding="max_length", return_tensors="np")
feed = {
    "input_ids":      enc["input_ids"].astype(np.int64),
    "attention_mask": enc["attention_mask"].astype(np.int64),
}

# Warmup
print("Warmup 3 runs...")
for _ in range(3):
    sess.run(["logits"], feed)

# Đo 20 lần
print("Measure 20 runs...")
times = []
for _ in range(20):
    t = time.time()
    sess.run(["logits"], feed)
    times.append((time.time() - t) * 1000)

print(f"\n=== RESULT ===")
print(f"  Min    = {min(times):.0f} ms")
print(f"  Mean   = {statistics.mean(times):.0f} ms")
print(f"  Median = {statistics.median(times):.0f} ms")
print(f"  Max    = {max(times):.0f} ms")
print(f"  Stdev  = {statistics.stdev(times):.0f} ms")

import os
print(f"\nSystem: cpu_count={os.cpu_count()}, onnxruntime={ort.__version__}")