import os
import time
import uuid
import json
import threading
import traceback
from contextlib import asynccontextmanager

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

import chat_engine as ce
from sample_data import make_sample_csv_bytes

load_dotenv()          # reads the secret key from the .env file on your computer

# ---- settings -------------------------------------------------------------
MAX_UPLOAD_MB = 5          # biggest file we accept
MAX_SESSIONS = 50          # how many visitors' datasets we keep in memory at once
SESSION_MINUTES = 60       # a dataset is forgotten after this long without use
MAX_QUESTIONS_PER_MINUTE = 6          # per visitor
GLOBAL_QUESTIONS_PER_MINUTE = 20      # for the whole server, to protect the free AI quota

# Websites allowed to talk to this server (comma separated). Add your live website address here later.
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")

ENGINE_LOCK = threading.Lock()      # the engine works on one dataset at a time, so questions take turns


@asynccontextmanager
async def lifespan(app):
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        ce.init_llm(key, test=False)                    # connect to Gemini once, when the server starts
    else:
        print("WARNING: GEMINI_API_KEY is not set, so /ask will not work. Uploading still works.")
    yield


app = FastAPI(title="Chat With Data", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Every visitor gets their own entry here: session id -> their cleaned table and report
SESSIONS = {}

RATE_LOCK = threading.Lock()
ASKED = {}          # session id -> the times of that visitor's recent questions
ASKED_ALL = []      # the times of recent questions from everyone


def check_rate_limit(session_id):
    """Allow a question, or stop with a friendly 'please wait' message."""
    now = time.time()
    with RATE_LOCK:
        mine = [t for t in ASKED.get(session_id, []) if now - t < 60]
        everyone = [t for t in ASKED_ALL if now - t < 60]
        if len(mine) >= MAX_QUESTIONS_PER_MINUTE:
            wait = int(60 - (now - mine[0])) + 1
            raise HTTPException(status_code=429, headers={"Retry-After": str(wait)},
                                detail=f"You're asking too quickly. Please wait {wait} seconds and try again.")
        if len(everyone) >= GLOBAL_QUESTIONS_PER_MINUTE:
            raise HTTPException(status_code=429, headers={"Retry-After": "30"},
                                detail="The demo is very busy right now. Please try again in a moment.")
        mine.append(now)
        everyone.append(now)
        ASKED[session_id] = mine
        ASKED_ALL[:] = everyone


def clean_old_sessions():
    cutoff = time.time() - SESSION_MINUTES * 60
    for sid in [s for s, v in SESSIONS.items() if v["last_used"] < cutoff]:
        del SESSIONS[sid]
        ASKED.pop(sid, None)
    while len(SESSIONS) >= MAX_SESSIONS:                    # still full: drop the oldest
        oldest = min(SESSIONS, key=lambda s: SESSIONS[s]["last_used"])
        del SESSIONS[oldest]
        ASKED.pop(oldest, None)


def records(frame, n=50):
    """A table as a plain list of rows that can be sent over the internet (blanks become null)."""
    return json.loads(frame.head(n).to_json(orient="records", date_format="iso"))


def column_kind(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    return "text"


def new_session(frame, report):
    clean_old_sessions()
    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = {"df": frame, "report": report, "last_used": time.time()}
    return session_id


def session_response(session_id, frame, report):
    return {
        "session_id": session_id,
        "rows": len(frame),
        "columns": [{"name": c, "type": column_kind(frame[c])} for c in frame.columns],
        "report": report,
        "preview": records(frame, 5),
    }


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

    session_id = new_session(frame, report)
    return session_response(session_id, frame, report)


SAMPLE_QUESTIONS = [
    "What is the total revenue?",
    "Which region has the highest total revenue?",
    "How many orders have no region?",
    "Show a bar chart of revenue by category",
    "Plot monthly revenue over time",
    "Predict the revenue of an order of 5 Shoes at unit price 1500 in the West",
]


@app.post("/sample")
def sample():
    """Start a session with the built-in demo data, so visitors can try the app without a file."""
    frame, report = ce.load_csv(make_sample_csv_bytes(), MAX_UPLOAD_MB)
    session_id = new_session(frame, report)
    out = session_response(session_id, frame, report)
    out["suggested_questions"] = SAMPLE_QUESTIONS
    return out


class Question(BaseModel):
    session_id: str
    question: str = Field(min_length=1, max_length=500)


def details_json(d):
    """The prediction details, turned into plain numbers and text."""
    if d is None:
        return None
    best = d["best"]
    return {
        "pattern_found": bool(d["signal"]),
        "target": d["plan"]["target"],
        "features": d["plan"]["features"],
        "best_model": str(best["model"]),
        "best_score": float(best["R² average"]),
        "baseline_score": float(d["baseline"]),
        "models": records(d["table"]),
        "importance": {k: float(v) for k, v in d["importance"].items()},
        "prediction": None if d["prediction"] is None else float(d["prediction"]),
    }


@app.post("/ask")
def ask(q: Question):
    if ce.client is None:
        raise HTTPException(status_code=503, detail="The AI is not set up on this server yet.")

    session = SESSIONS.get(q.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Your data has expired. Please upload your file again.")
    session["last_used"] = time.time()
    check_rate_limit(q.session_id)                          # stops visitors who ask too fast

    try:
        with ENGINE_LOCK:                                   # one question at a time
            ce.set_dataset(session["df"])                   # point the engine at THIS visitor's data
            result = ce.ask_anything(q.question)
    except Exception as e:
        if ce.is_rate_limit(str(e)):
            raise HTTPException(status_code=429, detail="The free AI is busy right now. Please wait a minute and try again.")
        traceback.print_exc()                               # details stay in the server's terminal, not shown to the visitor
        raise HTTPException(status_code=500, detail="Something went wrong while answering. Please try again.")

    table = result["table"]
    return {
        "answer": result["answer"],
        "route": result["route"],
        "sql": result["sql"],
        "table": None if table is None else records(table, 50),
        "table_rows": None if table is None else len(table),
        "chart": result["chart"],
        "notes": result["notes"],
        "details": details_json(result["details"]),
    }