from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = FastAPI(title="IoT Solution Finder")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE = os.path.dirname(os.path.dirname(__file__))
_store = {}
_model = None

@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(BASE, "public", "index.html"), encoding="utf-8") as f:
        return f.read()

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def detect_columns(df):
    keywords = (
        ("problem", "issue", "fault", "error", "failure", "alarm"),
        ("symptom", "sign", "indication", "observation"),
        ("solution", "fix", "action", "resolution", "remedy", "corrective"),
    )
    result = []
    for terms, fallback in zip(keywords, range(3)):
        col = next((c for term in terms for c in df.columns if term in c.lower()), None)
        result.append(col or df.columns[min(fallback, len(df.columns)-1)])
    return result

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        return JSONResponse({"error": "Please upload an Excel file (.xlsx or .xls)."}, status_code=400)
    try:
        df = pd.read_excel(io.BytesIO(await file.read()))
        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(how="all").reset_index(drop=True)
        if df.empty or len(df.columns) == 0:
            return JSONResponse({"error": "The workbook has no data rows."}, status_code=400)
        prob_col, sym_col, sol_col = detect_columns(df)
        combined = (df[prob_col].fillna("").astype(str) + " " + df[sym_col].fillna("").astype(str)).str.strip()
        embeddings = get_model().encode(combined.tolist(), show_progress_bar=False)
        _store["current"] = {"df": df, "embeddings": embeddings, "prob_col": prob_col, "sym_col": sym_col, "sol_col": sol_col}
        return {"rows": len(df), "prob_col": prob_col, "sym_col": sym_col, "sol_col": sol_col, "columns": list(df.columns)}
    except Exception as exc:
        return JSONResponse({"error": f"Could not process workbook: {exc}"}, status_code=400)

@app.post("/api/search")
async def search(query: str = Form(...), top_n: int = Form(5), min_score: float = Form(0.20)):
    if "current" not in _store:
        return JSONResponse({"error": "No file uploaded yet."}, status_code=400)
    if not query.strip():
        return JSONResponse({"error": "Enter a search query."}, status_code=400)
    data = _store["current"]
    corrected = query
    try:
        from fuzzywuzzy import process
        domain = ["motor", "pump", "sensor", "relay", "gateway", "battery", "firmware", "conveyor", "pressure", "temperature", "humidity", "flow", "vibration", "overheating", "current", "voltage", "power", "bearing", "inverter", "valve", "compressor", "fan", "communication", "connection", "signal", "reading"]
        corrected = " ".join((process.extractOne(w.lower(), domain)[0] if len(w) > 3 and process.extractOne(w.lower(), domain)[1] >= 80 else w) for w in query.split())
    except ImportError:
        pass
    scores = cosine_similarity(get_model().encode([corrected]), data["embeddings"])[0]
    top_n = max(1, min(int(top_n), 10))
    min_score = max(0.0, min(float(min_score), 1.0))
    results = []
    for rank, idx in enumerate(np.argsort(scores)[::-1][:top_n], 1):
        score = float(scores[idx])
        if score < min_score:
            break
        df = data["df"]
        results.append({"rank": rank, "score": round(score, 3), "problem": str(df.loc[idx, data["prob_col"]]), "symptoms": str(df.loc[idx, data["sym_col"]]), "solution": str(df.loc[idx, data["sol_col"]]), "row": int(idx) + 2})
    return {"results": results, "corrected": corrected if corrected.lower() != query.lower() else None, "query": query}
