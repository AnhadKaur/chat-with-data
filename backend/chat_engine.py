"""
chat_engine.py  -  the core logic of "Chat With Data"

Everything in this file was built and tested step by step in the Colab notebook:
  - talking to the LLM, with automatic fallback across several models
  - the SQL path:   question -> safe SQL -> DuckDB -> plain-English answer
  - the loader:     messy CSV -> clean table + a short report for the user
  - the router, the prediction path (plan -> checks -> models -> explanation)
  - the chart path  (the AI picks the chart, our code checks it)
  - ask_anything(question): the single front door

Rule that runs through all of it: the AI only writes plans, SQL and sentences.
Our own code does the checking, the maths and the decisions.

How to use it in a notebook (the functions share the notebook's variables):
    exec(open("chat_engine.py").read())
    init_llm(api_key)
    set_dataset(load_csv("file.csv")[0])
    ask_anything("How many rows are there?")
"""
import io
import re
import csv
import json
import time
import textwrap

import numpy as np
import pandas as pd
import duckdb
from google import genai
from google.genai import types
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------------------------
# Shared state (only created if it does not exist yet, so re-running this file
# in the same notebook never wipes the connection or the loaded data)
# ---------------------------------------------------------------------------
for _name, _default in [("client", None), ("WORKING_MODELS", []), ("df", None), ("schema_text", "")]:
    if _name not in globals():
        globals()[_name] = _default
if "con" not in globals():
    con = duckdb.connect()


# ---------------------------------------------------------------------------
# 1. Talking to the LLM
# ---------------------------------------------------------------------------
DEFAULT_MODELS = ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-flash-lite-latest"]


def clean(v):
    """Turn numpy numbers into normal Python numbers."""
    return v.item() if hasattr(v, "item") else v


def is_rate_limit(msg):
    m = str(msg).lower()
    return "429" in m or "resource_exhausted" in m or "all models failed" in m or "quota" in m


def ask_llm(system, user, max_tokens=1000, rounds=2):
    """Ask the AI. If one model is busy, try the next one; wait a little between rounds."""
    last_error = None
    for _ in range(rounds):
        for model_name in WORKING_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        max_output_tokens=max_tokens,
                    ),
                )
                if response.text:
                    return response.text.strip()
            except Exception as e:
                last_error = e
        time.sleep(5)
    raise RuntimeError(f"All models failed. Last error: {last_error}")


def init_llm(api_key, models=None, test=True):
    """Connect to Gemini. Pass models=[...] if the model names ever change."""
    global client, WORKING_MODELS
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=30000))
    WORKING_MODELS = list(models or DEFAULT_MODELS)
    if test:
        print("LLM test:", ask_llm("You are a friendly assistant.", "Say hello in 5 words."))


# ---------------------------------------------------------------------------
# 2. The SQL path
# ---------------------------------------------------------------------------
BANNED = ["drop", "delete", "update", "insert", "alter", "create", "attach",
          "copy", "pragma", "install", "load", "export", "call", "set"]


def is_safe_sql(query):
    """One single read-only SELECT, nothing else."""
    q = query.strip().lower().rstrip(";")
    if ";" in q:
        return False
    if not (q.startswith("select") or q.startswith("with")):
        return False
    return not any(re.search(rf"\b{word}\b", q) for word in BANNED)


def clean_sql(text):
    text = re.sub(r"```sql|```", "", text).strip()
    return text.rstrip(";").strip()


SQL_RULES = (
    "You are a DuckDB SQL expert. The only table is named data. "
    "Reply with exactly one SELECT query on that table, with no explanation and no markdown. "
    "If a column holding dates is text, use CAST(column AS DATE) for date functions. "
    "RULE 1: First check whether any listed column could be what the question means. "
    "Column names are often shortened: pct = percentage, qty = quantity, amt = amount, num = number, "
    "cust = customer, dt = date. If a listed column plausibly matches, use it. "
    "Only if NO listed column matches (for example profit, discount, returns, customer details or a salesperson, "
    "when none of the columns are about those), reply with exactly the single word CANNOT_ANSWER. "
    "Never invent a column, never return a constant such as 0 or NULL as if it were a result, "
    "and never use a different column as a stand-in (for example, never treat region as a salesperson). "
    "RULE 2: If the user asks to change the data in any way "
    "(delete, drop, update, insert, remove or add anything) or pastes their own SQL for you to run, "
    "reply with exactly the single word UNSAFE_REQUEST."
)

NO_INFO = "I can't answer that, because this data doesn't contain the information needed."
UNSAFE = "I couldn't answer that safely. Try rephrasing."


def generate_sql(question, prev_sql=None, error=None):
    user = f"{schema_text}\n\nQuestion: {question}"
    if error:
        user += f"\n\nYour previous query:\n{prev_sql}\nfailed with this error:\n{error}\nWrite a corrected query."
    return clean_sql(ask_llm(SQL_RULES, user))


def ask_question(question):
    """Lookup path. Returns (answer sentence, sql, result table)."""
    sql = generate_sql(question)
    for attempt in range(2):                                   # one try, plus one retry
        word = sql.strip().upper()
        if word.startswith("UNSAFE_REQUEST"):
            return UNSAFE, sql, None
        if word.startswith("CANNOT_ANSWER"):
            return NO_INFO, sql, None
        if not is_safe_sql(sql):
            return UNSAFE, sql, None
        if not re.search(r"\bfrom\s+data\b", sql.lower()):     # code backstop: must read from the data
            return NO_INFO, sql, None
        try:
            full_df = con.execute(sql).fetchdf()
            break
        except Exception as e:
            if attempt == 1:
                return "I couldn't compute that. Try rephrasing.", sql, None
            sql = generate_sql(question, sql, str(e))          # send the error back for one retry

    total_rows = len(full_df)
    result_df = full_df.head(1000)
    system = ("Answer the question in one or two short, natural sentences, using only the result provided. "
              "If the result is only a preview of a longer table, say how many rows there are in total "
              "and describe what the table contains. Never claim a preview is the complete data. "
              "This app can only read data, never change it, so never say that data was deleted, changed or removed.")
    user = (f"Question: {question}\n\nThe query returned {total_rows} rows in total. "
            f"Here are the first {min(total_rows, 20)} rows:\n{result_df.head(20).to_string(index=False)}")
    return ask_llm(system, user), sql, result_df


# ---------------------------------------------------------------------------
# 3. The loader: any CSV -> clean table + report
# ---------------------------------------------------------------------------
def _is_text_col(s):
    """True for text columns in both older and newer versions of pandas."""
    return s.dtype == object or pd.api.types.is_string_dtype(s.dtype)


def clean_column_names(columns):
    new_names, seen = [], {}
    for i, name in enumerate(columns):
        n = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower()).strip("_")
        if n == "":
            n = f"column_{i + 1}"
        if n[0].isdigit():
            n = "col_" + n
        if n in seen:                       # same name twice: add a number
            seen[n] += 1
            n = f"{n}_{seen[n]}"
        else:
            seen[n] = 1
        new_names.append(n)
    return new_names


MISSING_MARKERS = {"", "-", "--", "n/a", "na", "nan", "null", "none", "?", "\u2014"}


def tidy_cells(frame):
    """Trim spaces, turn 'missing' markers into real blanks, drop empty rows."""
    frame = frame.copy()
    for c in frame.columns:
        if _is_text_col(frame[c]):
            frame[c] = frame[c].map(
                lambda v: np.nan if (isinstance(v, str) and v.strip().lower() in MISSING_MARKERS)
                else (v.strip() if isinstance(v, str) else v)
            )
    return frame.dropna(how="all").reset_index(drop=True)


def convert_number_columns(frame):
    """Turn text like '₹1,200' or '10%' into real numbers (only if 80%+ of the values read as numbers)."""
    frame = frame.copy()
    renames = {}
    for c in frame.columns:
        if _is_text_col(frame[c]):
            vals = frame[c].dropna()
            if len(vals) == 0 or not all(isinstance(v, str) for v in vals):
                continue
            is_percent = (vals.str.strip().str.endswith("%")).mean() >= 0.8
            cleaned = frame[c].map(lambda v: re.sub(r"(?i)rs\.?|[\u20b9$\u20ac\u00a3,%\s]", "", v) if isinstance(v, str) else v)
            numeric = pd.to_numeric(cleaned, errors="coerce")
            if numeric.notna().sum() >= 0.8 * len(vals):
                frame[c] = numeric
                if is_percent:
                    renames[c] = f"{c}_pct"
        if pd.api.types.is_float_dtype(frame[c]):
            filled = frame[c].dropna()
            if len(filled) > 0 and (filled % 1 == 0).all():          # all whole numbers
                frame[c] = frame[c].astype("Int64")
    return frame.rename(columns=renames)


def standardize_spelling(frame):
    """'north' / 'North' -> the most common spelling, everywhere."""
    frame = frame.copy()
    for c in frame.columns:
        if _is_text_col(frame[c]):
            vals = frame[c].dropna()
            if len(vals) == 0 or not all(isinstance(v, str) for v in vals) or vals.nunique() > 50:
                continue
            groups = {}
            for v, n in vals.value_counts().items():
                groups.setdefault(v.lower(), []).append((v, n))
            mapping = {}
            for variants in groups.values():
                best = max(variants, key=lambda x: (x[1], x[0].istitle()))[0]
                for v, _ in variants:
                    mapping[v] = best
            frame[c] = frame[c].map(lambda v: mapping.get(v, v))
    return frame


DATE_PATTERN = re.compile(r"^(\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}|\d{1,2}[/\-.]\d{1,2}[/\-.]\d{4})$")


def convert_date_columns(frame):
    """Read date-looking text as real dates, working out day/month order from the evidence."""
    frame = frame.copy()
    notes = []
    for c in frame.columns:
        if not _is_text_col(frame[c]):
            continue
        vals = frame[c].dropna()
        if len(vals) == 0 or not all(isinstance(v, str) for v in vals):
            continue
        is_date = vals.map(lambda v: bool(DATE_PATTERN.match(v)))
        if is_date.mean() < 0.8:
            continue

        text = frame[c].astype(str).str.strip().str.replace(r"[/.]", "-", regex=True)
        parts = vals[is_date].str.replace(r"[/.]", "-", regex=True).str.split("-")

        if parts.map(lambda p: len(p[0]) == 4).mean() >= 0.5:
            fmt, why = "%Y-%m-%d", "year first"
        else:
            first = parts.map(lambda p: int(p[0]))
            second = parts.map(lambda p: int(p[1]))
            if (first > 12).any() and (second > 12).any():
                notes.append(f"{c}: left as text, because the day/month order is unclear (mixed formats)")
                continue
            elif (first > 12).any():
                fmt, why = "%d-%m-%Y", "day/month/year, because a first number above 12 was found"
            elif (second > 12).any():
                fmt, why = "%m-%d-%Y", "month/day/year, because a second number above 12 was found"
            else:
                fmt, why = "%d-%m-%Y", "day/month/year (assumed: every date could be read either way)"

        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if parsed.notna().sum() < 0.8 * len(vals):
            notes.append(f"{c}: left as text, because many values didn't fit one date format")
            continue
        frame[c] = parsed
        notes.append(f"{c}: read as {why}")
    return frame, notes


def load_csv(source, max_mb=20, max_rows=200_000, max_cols=100):
    """source = a file path or raw bytes. Returns (clean table, report lines)."""
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        with open(source, "rb") as f:
            raw = f.read()
    if len(raw.strip()) == 0:
        raise ValueError("The file is empty.")
    if len(raw) > max_mb * 1024 * 1024:
        raise ValueError(f"The file is larger than {max_mb} MB, which is the limit.")

    frame, enc, sep = None, None, ","
    for enc in ["utf-8-sig", "cp1252", "latin-1"]:
        try:
            sample = raw[:20000].decode(enc, errors="replace")
            try:
                sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except csv.Error:
                sep = ","
            frame = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep, dtype=str)
            break
        except UnicodeDecodeError:
            continue
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            raise ValueError(f"I couldn't read this file as a table ({str(e)[:80]}).")
    if frame is None:
        raise ValueError("I couldn't read this file as a CSV.")

    rows_in, cols_in = frame.shape
    frame = frame.dropna(axis=1, how="all")
    if frame.shape[1] == 0 or len(frame) == 0:
        raise ValueError("The file has no readable data.")
    if len(frame) > max_rows:
        raise ValueError(f"The file has more than {max_rows:,} rows, which is the limit.")
    if frame.shape[1] > max_cols:
        raise ValueError(f"The file has more than {max_cols} columns, which is the limit.")

    orig_cols = list(frame.columns)
    frame.columns = clean_column_names(frame.columns)
    frame = tidy_cells(frame)
    removed = rows_in - len(frame)
    frame = convert_number_columns(frame)
    frame, date_notes = convert_date_columns(frame)
    frame = standardize_spelling(frame)
    if len(frame) == 0:
        raise ValueError("The file has no data rows.")

    report = [f"Read {rows_in} rows and {cols_in} columns (encoding {enc}, separator {sep!r})."]
    if removed > 0:
        report.append(f"Removed {removed} blank row(s).")
    renamed = [f"'{o}' -> {n}" for o, n in zip(orig_cols, frame.columns) if o != n]
    if renamed:
        report.append("Renamed columns: " + "; ".join(renamed) + ".")
    nums = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    dates = [c for c in frame.columns if pd.api.types.is_datetime64_any_dtype(frame[c])]
    texts = [c for c in frame.columns if c not in nums and c not in dates]
    report.append(f"Number columns: {', '.join(nums) or 'none'}. Date columns: {', '.join(dates) or 'none'}. "
                  f"Text columns: {', '.join(texts) or 'none'}.")
    report += date_notes
    gaps = {c: int(frame[c].isna().sum()) for c in frame.columns if frame[c].isna().any()}
    if gaps:
        report.append("Columns with missing values: " + ", ".join(f"{c} ({n})" for c, n in gaps.items()) + ".")
    return frame, report


def set_dataset(new_df):
    """Point the whole app at a new table and rebuild the briefing the AI reads."""
    global df, con, schema_text
    df = new_df
    try:
        con.unregister("data")
    except Exception:
        pass
    con.register("data", df)
    schema = con.execute("DESCRIBE data").fetchdf()
    schema_text = (
        "Table name: data\n\nColumns:\n"
        + schema[["column_name", "column_type"]].to_string(index=False)
        + "\n\nSample rows:\n"
        + df.head(3).to_string(index=False) + "\n"
    )
    return schema_text


# ---------------------------------------------------------------------------
# 4. The router
# ---------------------------------------------------------------------------
def classify_question(question):
    system = ("Classify a user's question about a dataset into exactly one word: "
              "lookup (asks for a number, list, comparison or fact from the existing data), "
              "chart (asks to plot, graph, chart or visualize something), "
              "predict (asks to forecast, estimate future values, or predict an outcome using a model). "
              "Reply with only that one word.")
    user = f"{schema_text}\n\nQuestion: {question}"
    out = ask_llm(system, user, max_tokens=200).strip().lower()
    for label in ["predict", "chart", "lookup"]:
        if label in out:
            return label
    return "lookup"        # safe default


# ---------------------------------------------------------------------------
# 5. The prediction path (numbers)
# ---------------------------------------------------------------------------
def get_ml_plan(question):
    system = ("You plan a machine learning task for a dataset. Reply with ONLY a JSON object, no markdown, with these keys: "
              '"task" ("regression" if the target is a number, "classification" if it is a category), '
              '"target" (one column name), '
              '"features" (list of column names to predict from; never include ID columns or the target), '
              '"scenario" (an object mapping column names to values if the user describes one specific case to predict, otherwise null). '
              "Use only column names that exist in the table.")
    raw = ask_llm(system, f"{schema_text}\n\nQuestion: {question}", max_tokens=2000)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    return json.loads(match.group(0))


def validate_plan(plan, frame):
    """Our own check of the AI's plan. Returns (clean plan, list of dropped columns)."""
    cols = list(frame.columns)
    dropped = []

    if plan.get("task") not in ("regression", "classification"):
        raise ValueError("I couldn't tell whether to predict a number or a category")

    target = plan.get("target")
    if target not in cols:
        raise ValueError(f"the column '{target}' doesn't exist in this data")

    features = []
    for c in plan.get("features", []):
        if c not in cols:
            dropped.append((c, "doesn't exist"))
        elif c == target:
            dropped.append((c, "is the column being predicted"))
        elif not pd.api.types.is_numeric_dtype(frame[c]) and frame[c].nunique() > 20:
            dropped.append((c, "text column with too many different values"))
        elif frame[c].nunique() == len(frame):
            dropped.append((c, "looks like an ID (every row is different)"))
        else:
            features.append(c)

    if not features:
        raise ValueError("there are no usable columns to predict from")
    if frame[target].notna().sum() < 30:
        raise ValueError("there are too few rows with a value to learn from (need at least 30)")

    return {**plan, "features": features}, dropped


def find_formula_columns(frame):
    """Finds numeric columns that are exactly the product of two other columns (like revenue = quantity x unit_price)."""
    num_cols = frame.select_dtypes(include="number").columns.tolist()
    if len(num_cols) > 15:            # limit: the search grows quickly with many number columns
        return {}
    formulas = {}
    for i, a in enumerate(num_cols):
        for b in num_cols[i + 1:]:
            for c in num_cols:
                if c in (a, b) or c in formulas:
                    continue
                mask = frame[[a, b, c]].notna().all(axis=1)
                if mask.sum() < 30:
                    continue
                if np.allclose(frame.loc[mask, a].astype(float) * frame.loc[mask, b].astype(float),
                               frame.loc[mask, c].astype(float)):
                    formulas[c] = (a, b)
    return formulas


def check_leakage(plan, frame):
    formulas = find_formula_columns(frame)
    target = plan["target"]
    features = list(plan["features"])
    notes = []
    for derived, (a, b) in formulas.items():
        if target in (a, b) and derived in features:
            features.remove(derived)
            notes.append(f"Removed '{derived}': it is calculated from '{target}' ({derived} = {a} x {b}), so it would give the answer away.")
        if derived == target and a in features and b in features:
            notes.append(f"Note: '{target}' is exactly {a} x {b}, so any model will look perfect. It is just rediscovering a formula, not learning a pattern.")
    if not features:
        raise ValueError("after removing leaky columns, nothing is left to predict from")
    return {**plan, "features": features}, notes


def build_prep(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", Pipeline([("fill", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])


def prepare_X(X):
    X = X.copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(object).where(X[c].notna(), np.nan)    # make blanks a standard "missing" marker
        elif pd.api.types.is_extension_array_dtype(X[c]):
            X[c] = X[c].astype(float)                                  # nullable integers -> plain decimals
    return X


def compare_regression_models(frame, plan):
    data = frame.dropna(subset=[plan["target"]])
    X = prepare_X(data[plan["features"]])
    y = data[plan["target"]].astype(float)

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    models = {
        "Baseline (always the average)": DummyRegressor(),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)       # same splits for everyone
    rows = []
    for name, model in models.items():
        pipe = Pipeline([("prep", build_prep(num_cols, cat_cols)), ("model", model)])
        s = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
        rows.append({"model": name, "R² average": round(s.mean(), 3), "spread (std)": round(s.std(), 3)})

    table = pd.DataFrame(rows)
    real = table[~table["model"].str.startswith("Baseline")]
    best = real.loc[real["R² average"].idxmax()]
    baseline_score = table.iloc[0]["R² average"]
    return table, best, baseline_score


def make_model(name):
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }[name]


def fit_and_predict(frame, plan, best_name):
    data = frame.dropna(subset=[plan["target"]])
    X = prepare_X(data[plan["features"]])
    y = data[plan["target"]].astype(float)
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pipe = Pipeline([("prep", build_prep(num_cols, cat_cols)),
                     ("model", make_model(best_name))]).fit(X, y)

    imp = permutation_importance(pipe, X, y, n_repeats=5, random_state=42, scoring="r2")
    importance = pd.Series(imp.importances_mean, index=plan["features"]).sort_values(ascending=False)

    prediction = None
    sc = plan.get("scenario")
    if sc:
        row = {}
        for c in plan["features"]:
            if c in sc:
                row[c] = sc[c]
            elif c in num_cols:
                row[c] = X[c].median()        # not mentioned: use the typical value
            else:
                row[c] = X[c].mode()[0]
        prediction = pipe.predict(prepare_X(pd.DataFrame([row])))[0]
    return pipe, importance, prediction


def has_signal(best, baseline_score, min_r2=0.10, min_gain=0.05):
    """Our rule for 'is there a real pattern?': the best model must clearly beat the dumb guess."""
    score = best["R² average"]
    return score >= min_r2 and (score - baseline_score) >= min_gain


def explain_prediction(plan, table, best, baseline_score, importance, prediction, notes, signal=True):
    facts = f"Goal: predict '{plan['target']}' from {plan['features']}.\n"
    facts += f"Models tested with 5-fold cross-validation (R² average):\n{table.to_string(index=False)}\n"
    facts += f"Highest scoring model: {best['model']} with R² {best['R² average']}. Baseline (always guessing the average): {baseline_score}.\n"
    if signal:
        facts += f"Most important columns (higher means it mattered more): {importance.round(2).to_dict()}.\n"
        if prediction is not None:
            facts += f"Prediction for the user's case {plan.get('scenario')}: {round(float(prediction))}.\n"
    else:
        facts += ("VERDICT: no reliable pattern was found. No model clearly beats the baseline. "
                  "Do not give a prediction and do not rank columns.\n")
    for n in notes:
        facts += f"Important note: {n}\n"

    system = ("You explain a machine learning result to a non-technical person in 3 to 5 short sentences. "
              "If the facts contain a VERDICT saying no reliable pattern was found: say plainly that these columns do not "
              "reliably predict the target, explain that the models scored near or below the baseline guess, "
              "and suggest the question may not be answerable from this data. Do not describe any model as a success. "
              "Otherwise cover: what was predicted, which model did best versus the baseline, which columns mattered most, "
              "and the prediction if given. If there is an important note about a formula, mention it honestly. "
              "Use only the numbers provided. Do not invent anything.")
    return ask_llm(system, facts, max_tokens=4000)


def answer_prediction(question):
    """Prediction path. Returns (explanation text, details dict or None)."""
    try:
        plan = get_ml_plan(question)
        plan, dropped = validate_plan(plan, df)
        plan, notes = check_leakage(plan, df)
    except Exception as e:
        if is_rate_limit(str(e)):
            raise                     # a busy AI is not the same thing as an impossible question
        return f"I couldn't set up that prediction: {e}", None

    if plan["task"] != "regression" or not pd.api.types.is_numeric_dtype(df[plan["target"]]):
        return "I can predict numbers (like revenue or quantity) right now. Predicting categories is not built yet.", None

    # make sure the user's case uses real numbers for number columns
    sc = plan.get("scenario")
    if sc:
        fixed = {}
        for k, v in sc.items():
            if k in plan["features"]:
                if pd.api.types.is_numeric_dtype(df[k]):
                    try:
                        v = float(v)
                    except (ValueError, TypeError):
                        continue
                fixed[k] = v
        plan["scenario"] = fixed or None

    table, best, baseline = compare_regression_models(df, plan)
    signal = has_signal(best, baseline)

    importance, prediction = pd.Series(dtype=float), None
    if signal:
        _, importance, prediction = fit_and_predict(df, plan, best["model"])

    text = explain_prediction(plan, table, best, baseline, importance, prediction, notes, signal=signal)
    details = {"plan": plan, "table": table, "best": best, "baseline": baseline,
               "signal": signal, "importance": importance, "prediction": prediction}
    return text, details


# ---------------------------------------------------------------------------
# 6. The chart path
# ---------------------------------------------------------------------------
CHART_TYPES = {"bar", "line", "pie"}
DATE_LIKE = re.compile(r"^\d{4}(-\d{2}){0,2}$")


def build_chart(spec, table):
    """Check the AI's chart choice and package the data for the website to draw."""
    notes = []
    if table is None or len(table) == 0:
        raise ValueError("there is no data to draw")

    ctype = str(spec.get("chart", "")).lower()
    if ctype not in CHART_TYPES:
        raise ValueError(f"I can't draw a '{ctype}' chart (I can do bar, line and pie)")

    x, y = spec.get("x"), spec.get("y")
    if x not in table.columns or y not in table.columns or x == y:
        raise ValueError("the chart refers to a column that isn't in the data")
    if not pd.api.types.is_numeric_dtype(table[y]):
        raise ValueError(f"'{y}' is not a number column, so it can't be plotted")

    data = table[[x, y]].copy()
    x_is_time = pd.api.types.is_datetime64_any_dtype(data[x])
    x_is_num = pd.api.types.is_numeric_dtype(data[x])
    if x_is_time:
        data[x] = data[x].dt.strftime("%Y-%m-%d")
    elif not x_is_num:
        data[x] = data[x].map(lambda v: "(blank)" if pd.isna(v) else str(v))

    before = len(data)
    data = data.dropna(subset=[x, y])
    if len(data) < before:
        notes.append(f"Left out {before - len(data)} row(s) that had a missing value.")
    if len(data) == 0:
        raise ValueError("there is no data to draw")

    ordered = x_is_time or x_is_num or data[x].astype(str).map(lambda v: bool(DATE_LIKE.match(v))).all()
    if ctype == "line" and not ordered:
        ctype = "bar"
        notes.append("A line chart needs dates or numbers along the bottom, so I used a bar chart.")
    if ctype == "pie" and ((data[y] < 0).any() or len(data) > 8):
        ctype = "bar"
        notes.append("A pie chart needs 8 or fewer positive values, so I used a bar chart.")
    if ctype == "line":
        data = data.sort_values(x)

    cap = 200 if ctype == "line" else 30
    if len(data) > cap:
        notes.append(f"Showing the first {cap} of {len(data)} rows.")
        data = data.head(cap)

    points = [{"x": clean(a), "y": clean(b)} for a, b in zip(data[x], data[y])]
    return {"type": ctype, "x": x, "y": y, "title": str(spec.get("title", "")),
            "data": points, "notes": notes}


def get_chart_spec(question, table):
    cols = ", ".join(f"{c} ({'number' if pd.api.types.is_numeric_dtype(table[c]) else 'text or date'})"
                     for c in table.columns)
    system = ("You choose how to draw a chart. Reply with ONLY a JSON object, no markdown, with these keys: "
              '"chart" ("bar", "line" or "pie"), "x" (the column for labels or dates), '
              '"y" (the column holding the numbers to plot), "title" (a short title). '
              "Use only the column names listed. Use line only when x is a date or a month, "
              "pie only for parts of a whole with at most 8 categories, and bar otherwise. "
              "If the user names a chart type, use it.")
    user = f"Question: {question}\nTable columns: {cols}\nFirst rows:\n{table.head(5).to_string(index=False)}"
    raw = ask_llm(system, user, max_tokens=2000)
    return json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))


def answer_chart(question):
    out = {"question": question, "answer": None, "sql": None, "table": None, "chart": None, "notes": []}
    data_question = (question + " (Give only the data: one column of labels and one column of numbers. "
                     "Do not draw bars or charts with text.)")
    answer, sql, table = ask_question(data_question)
    out["answer"], out["sql"], out["table"] = answer, sql, table
    if table is None or len(table) == 0:
        return out                                   # refused, impossible, or empty: no chart
    try:
        chart = build_chart(get_chart_spec(question, table), table)
        out["chart"], out["notes"] = chart, chart["notes"]
    except Exception as e:
        if is_rate_limit(str(e)):
            raise
        out["notes"] = [f"I couldn't draw a chart ({e}), so here is the data as a table."]
    return out


def draw_chart(chart):
    """Draw a chart package with matplotlib (for notebooks; the website will draw it with its own library)."""
    import matplotlib.pyplot as plt
    labels = [str(p["x"]) for p in chart["data"]]
    values = [p["y"] for p in chart["data"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    if chart["type"] == "bar":
        ax.bar(labels, values)
    elif chart["type"] == "line":
        ax.plot(labels, values, marker="o")
    else:
        ax.pie(values, labels=labels, autopct="%1.0f%%")
    ax.set_title(chart["title"] or f"{chart['y']} by {chart['x']}")
    if chart["type"] != "pie":
        ax.set_xlabel(chart["x"])
        ax.set_ylabel(chart["y"])
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    for n in chart["notes"]:
        print("note:", n)


# ---------------------------------------------------------------------------
# 7. The single front door
# ---------------------------------------------------------------------------
def ask_anything(question):
    """Every question comes in here. The answer always has the same shape."""
    route = classify_question(question)
    result = {"question": question, "route": route, "answer": None, "sql": None,
              "table": None, "details": None, "chart": None, "notes": []}

    if route == "predict":
        text, details = answer_prediction(question)
        result["answer"], result["details"] = text, details

    elif route == "chart":
        r = answer_chart(question)
        result["answer"], result["sql"], result["table"] = r["answer"], r["sql"], r["table"]
        result["chart"], result["notes"] = r["chart"], r["notes"]

    else:  # lookup
        answer, sql, table = ask_question(question)
        result["answer"], result["sql"], result["table"] = answer, sql, table

    return result
