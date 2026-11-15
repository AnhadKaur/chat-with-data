import time
import uuid
import json

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

import chat_engine as ce

# ---- settings -------------------------------------------------------------
MAX_UPLOAD_MB = 5          # biggest file we accept
MAX_SESSIONS = 50          # how many visitors' datasets we keep in memory at once
SESSION_MINUTES = 60       # a dataset is forgotten after this long without use

app = FastAPI(title="Chat With Data")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],   # the website we build later
    allow_methods=["*"],
    allow_headers=["*"],
)

# Every visitor gets their own entry here: session id -> their cleaned table and report
SESSIONS = {}


def clean_old_sessions():
    cutoff = time.time() - SESSION_MINUTES * 60
    for sid in [s for s, v in SESSIONS.items() if v["last_used"] < cutoff]:
        del SESSIONS[sid]
    while len(SESSIONS) >= MAX_SESSIONS:                    # still full: drop the oldest
        oldest = min(SESSIONS, key=lambda s: SESSIONS[s]["last_used"])
        del SESSIONS[oldest]


def records(frame, n=50):
    """A table as a plain list of rows that can be sent over the internet (blanks become null)."""
    return json.loads(frame.head(n).to_json(orient="records", date_format="iso"))


def column_kind(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    return "text"


@app.get("/health")
def health():
    return {"status": "ok", "message": "The server is running"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    if not name.endswith((".csv", ".txt", ".tsv")):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    limit = MAX_UPLOAD_MB * 1024 * 1024
    data = await file.read(limit + 1)                       # read one byte more than allowed, to detect big files
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"The file is larger than {MAX_UPLOAD_MB} MB, which is the limit.")

    try:
        frame, report = await run_in_threadpool(ce.load_csv, data, MAX_UPLOAD_MB)
    except ValueError as e:                                 # our own clear messages: empty file, too many rows...
        raise HTTPException(status_code=400, detail=str(e))

    clean_old_sessions()
    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = {"df": frame, "report": report, "last_used": time.time()}

    return {
        "session_id": session_id,
        "rows": len(frame),
        "columns": [{"name": c, "type": column_kind(frame[c])} for c in frame.columns],
        "report": report,
        "preview": records(frame, 5),
    }