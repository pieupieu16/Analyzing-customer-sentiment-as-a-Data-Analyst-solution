import time, statistics, numpy as np, onnxruntime as ort
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("../onnx_export/tokenizer_phobert", use_fast=True)
sess = ort.InferenceSession("../onnx_export/multioutput_phobert.onnx",
                            providers=["CPUExecutionProvider"])

text = "Đánh giá 5 sao. quần kaki chất vải rất tốt, mặc đứng form, đường may chắc chắn"

# Warmup
for _ in range(3):
    enc = tok(text, truncation=True, max_length=256, padding="max_length", return_tensors="np")
    sess.run(["logits"], {"input_ids": enc["input_ids"].astype(np.int64),
                          "attention_mask": enc["attention_mask"].astype(np.int64)})

# Đo tokenize+run trong cùng loop (giống /predict thật)
times = []
tok_times = []
run_times = []
for _ in range(20):
    t0 = time.time()
    enc = tok(text, truncation=True, max_length=256, padding="max_length", return_tensors="np")
    t1 = time.time()
    sess.run(["logits"], {"input_ids": enc["input_ids"].astype(np.int64),
                          "attention_mask": enc["attention_mask"].astype(np.int64)})
    t2 = time.time()
    tok_times.append((t1-t0)*1000)
    run_times.append((t2-t1)*1000)
    times.append((t2-t0)*1000)

print(f"Tokenize: Mean={statistics.mean(tok_times):.0f}ms")
print(f"Run:      Mean={statistics.mean(run_times):.0f}ms")
print(f"Total:    Mean={statistics.mean(times):.0f}ms")