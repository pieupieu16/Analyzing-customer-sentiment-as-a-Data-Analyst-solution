import json
import urllib.request

URL = "http://localhost:8000/predict"
PAYLOAD = {
    "rating": 5,
    "content": "quần kaki chất vải rất tốt, mặc đứng form, đường may chắc chắn"
}

# Print bytes actually sent so we can verify UTF-8
body = json.dumps(PAYLOAD, ensure_ascii=False).encode("utf-8")
print(f"--- BYTES SENT ({len(body)} bytes) ---")
print(body)
print(f"--- HEX of 'quần' segment ---")
target = "quần".encode("utf-8")
print(target.hex())
print()

req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json; charset=utf-8"})
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode("utf-8"))

print("--- RESULT ---")
for p in data["predictions"]:
    print(f"  {p['aspect']:<20} {p['label']:<10} {p['confidence']:.3f}")
print(f"  inference_ms = {data['inference_ms']}")
