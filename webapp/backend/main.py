import io
import os
import threading
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .aggregate import build_insights, build_summary
from .model import ModelService
from .schemas import BatchResponse, BatchSummary, PredictRequest, PredictResponse

MAX_ROWS = 5000
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS")
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Kick off model load in background thread so the server is reachable immediately.
    svc = ModelService.get()
    t = threading.Thread(target=svc.load, daemon=True)
    t.start()
    yield


app = FastAPI(title="ABSA Quần áo — PhoBERT", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health():
    svc = ModelService.get()
    info = svc.info()
    return {
        "status": "ok",
        "model_loaded": svc.is_ready(),
        **info,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    svc = ModelService.get()
    if not svc.is_ready():
        raise HTTPException(status_code=503, detail="Model not ready, please retry in a few seconds.")
    result = svc.predict_one(req.rating, req.content)
    return PredictResponse(
        input=req,
        predictions=result["predictions"],
        inference_ms=result["inference_ms"],
    )


@app.post("/predict_batch", response_model=BatchResponse)
async def predict_batch(file: UploadFile = File(...)):
    svc = ModelService.get()
    if not svc.is_ready():
        raise HTTPException(status_code=503, detail="Model not ready.")

    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        max_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"CSV file is too large. Max size is {max_mb:.1f} MB.")

    df = None
    last_err = None
    for enc in ["utf-8-sig", "utf-8", "cp1258", "latin-1"]:
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            break
        except Exception as e:
            last_err = e
    if df is None:
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {last_err!r}")

    if "rating" not in df.columns or "content" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="CSV must contain columns 'rating' and 'content'.",
        )

    original_count = len(df)
    truncated_from = None
    if original_count > MAX_ROWS:
        df = df.head(MAX_ROWS)
        truncated_from = original_count

    before = len(df)
    df = df.dropna(subset=["content"]).copy()
    df["content"] = df["content"].astype(str)
    df = df[df["content"].str.strip().str.len() > 0].copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"]).copy()
    df["rating"] = df["rating"].astype(int).clip(1, 5)
    skipped = before - len(df)

    if len(df) == 0:
        raise HTTPException(status_code=400, detail="No valid rows after cleaning.")

    t0 = time.time()
    preds = svc.predict_batch(df["rating"].tolist(), df["content"].tolist())

    rows = []
    for (_, src_row), aspect_list in zip(df.iterrows(), preds):
        out = {"rating": int(src_row["rating"]), "content": src_row["content"]}
        for ap in aspect_list:
            out[ap["aspect"]] = ap["label"]
            out[f"{ap['aspect']}_conf"] = round(ap["confidence"], 4)
        rows.append(out)

    summary = build_summary(rows, skipped=skipped, truncated_from=truncated_from)
    summary["insights"] = build_insights(summary)

    total_ms = int((time.time() - t0) * 1000)
    return BatchResponse(summary=BatchSummary(**summary), rows=rows, total_ms=total_ms)


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
