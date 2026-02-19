"""
EvalAI Backend – FastAPI
AI-powered Assignment Evaluation Platform

Features:
  - REST APIs for assignments & submissions
  - TF-IDF cosine similarity for plagiarism detection
  - LLM-based automated feedback (OpenAI / fallback rule-based)
  - SQLite database (swap to PostgreSQL for production)
  - File upload support (PDF text extraction)
  - CORS enabled for Vercel frontend
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, List
import sqlite3, uuid, os, re, math, json
from datetime import datetime

# PDF support disabled (removed to simplify deployment)
PDF_SUPPORT = False

# ── Optional: OpenAI for LLM feedback ────────────────────────────────────────
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
except ImportError:
    OPENAI_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════════════

DB_PATH = os.getenv("DATABASE_URL", "evalai.db")

def get_db():
    """Dependency – yields a DB connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Users
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            email       TEXT UNIQUE NOT NULL,
            role        TEXT NOT NULL CHECK(role IN ('student','instructor')),
            password_hash TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        )
    """)

    # Assignments
    c.execute("""
        CREATE TABLE IF NOT EXISTS assignments (
            id            TEXT PRIMARY KEY,
            title         TEXT NOT NULL,
            description   TEXT,
            instructor_id TEXT NOT NULL,
            deadline      TEXT,
            created_at    TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (instructor_id) REFERENCES users(id)
        )
    """)

    # Submissions
    c.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id            TEXT PRIMARY KEY,
            assignment_id TEXT NOT NULL,
            student_id    TEXT NOT NULL,
            content       TEXT NOT NULL,
            file_path     TEXT,
            submitted_at  TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (assignment_id) REFERENCES assignments(id),
            FOREIGN KEY (student_id)    REFERENCES users(id)
        )
    """)

    # Feedback
    c.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id               TEXT PRIMARY KEY,
            submission_id    TEXT NOT NULL UNIQUE,
            score            INTEGER,
            plagiarism_risk  TEXT,
            feedback_summary TEXT,
            similarity_scores TEXT,  -- JSON blob
            created_at       TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (submission_id) REFERENCES submissions(id)
        )
    """)

    # Seed demo data
    c.execute("INSERT OR IGNORE INTO users(id,name,email,role) VALUES(?,?,?,?)",
              ("I001","Dr. Sharma","sharma@university.edu","instructor"))
    c.execute("INSERT OR IGNORE INTO users(id,name,email,role) VALUES(?,?,?,?)",
              ("S001","Alice Patel","alice@student.edu","student"))
    c.execute("INSERT OR IGNORE INTO users(id,name,email,role) VALUES(?,?,?,?)",
              ("S002","Bob Kumar","bob@student.edu","student"))
    c.execute("INSERT OR IGNORE INTO users(id,name,email,role) VALUES(?,?,?,?)",
              ("S003","Carol Singh","carol@student.edu","student"))

    c.execute("""INSERT OR IGNORE INTO assignments(id,title,description,instructor_id,deadline)
                 VALUES(?,?,?,?,?)""",
              ("A001","Data Structures Essay",
               "Compare arrays and linked lists. Discuss time/space complexity.",
               "I001","2025-08-01"))
    c.execute("""INSERT OR IGNORE INTO assignments(id,title,description,instructor_id,deadline)
                 VALUES(?,?,?,?,?)""",
              ("A002","Algorithm Analysis",
               "Write a report on Big-O notation with examples.",
               "I001","2025-08-10"))
    c.execute("""INSERT OR IGNORE INTO assignments(id,title,description,instructor_id,deadline)
                 VALUES(?,?,?,?,?)""",
              ("A003","OS Concepts",
               "Compare processes vs threads with use cases.",
               "I001","2025-08-15"))

    conn.commit()
    conn.close()

# ══════════════════════════════════════════════════════════════════════════════
#  TF-IDF PLAGIARISM ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())

def compute_tfidf(doc_tokens: list[str], all_docs: list[list[str]]) -> dict[str, float]:
    tf: dict[str, float] = {}
    for t in doc_tokens:
        tf[t] = tf.get(t, 0) + 1
    for t in tf:
        tf[t] /= len(doc_tokens) if doc_tokens else 1

    idf: dict[str, float] = {}
    N = len(all_docs)
    for term in tf:
        df = sum(1 for d in all_docs if term in d)
        idf[term] = math.log((N + 1) / (df + 1)) + 1

    return {t: tf[t] * idf[t] for t in tf}

def cosine_similarity(v1: dict, v2: dict) -> float:
    common = set(v1) & set(v2)
    if not common:
        return 0.0
    dot = sum(v1[t] * v2.get(t, 0) for t in common)
    mag1 = math.sqrt(sum(x**2 for x in v1.values()))
    mag2 = math.sqrt(sum(x**2 for x in v2.values()))
    return dot / (mag1 * mag2) if mag1 and mag2 else 0.0

def detect_plagiarism(new_text: str, existing_texts: list[str]) -> tuple[float, list[dict]]:
    """
    Returns (max_similarity_pct, detailed_scores_list).
    Uses TF-IDF + cosine similarity across all existing submissions.
    """
    if not existing_texts:
        return 0.0, []

    new_tokens = tokenize(new_text)
    all_tokens = [tokenize(t) for t in existing_texts]
    all_docs = [new_tokens] + all_tokens

    new_vec = compute_tfidf(new_tokens, all_docs)
    scores = []
    for i, tokens in enumerate(all_tokens):
        vec = compute_tfidf(tokens, all_docs)
        sim = cosine_similarity(new_vec, vec)
        scores.append({"submission_index": i, "similarity": round(sim * 100, 2)})

    max_sim = max(s["similarity"] for s in scores) if scores else 0.0
    return max_sim, scores


# ══════════════════════════════════════════════════════════════════════════════
#  FEEDBACK GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def rule_based_feedback(content: str, score: int) -> str:
    """Fallback rule-based feedback when no LLM is available."""
    length = len(content.split())
    feedback_parts = []

    if length < 50:
        feedback_parts.append("The submission is too brief – please elaborate further.")
    elif length < 150:
        feedback_parts.append("The response has a reasonable length but could use more detail.")
    else:
        feedback_parts.append("Good length and effort in the submission.")

    if score >= 80:
        feedback_parts.append("The core concepts are explained well with clear structure.")
    elif score >= 60:
        feedback_parts.append("The explanation covers the basics but lacks depth in key sections.")
    else:
        feedback_parts.append("The submission needs significant improvement. Key concepts are missing or unclear.")

    # Check for keywords
    keywords = ["because", "therefore", "however", "example", "for instance"]
    if not any(k in content.lower() for k in keywords):
        feedback_parts.append("Consider using examples and connective reasoning to strengthen the argument.")

    return " ".join(feedback_parts)


def llm_feedback(content: str, assignment_title: str) -> str:
    """Use OpenAI to generate intelligent feedback."""
    if not OPENAI_AVAILABLE:
        return None
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": (
                    "You are an academic evaluator. Give concise, constructive feedback "
                    "(2-3 sentences) on student assignments. Be specific about strengths and weaknesses."
                )},
                {"role": "user", "content": (
                    f"Assignment: {assignment_title}\n\n"
                    f"Student submission:\n{content[:2000]}\n\n"
                    "Provide 2-3 sentences of constructive feedback."
                )},
            ],
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


def score_submission(content: str, plagiarism_pct: float) -> int:
    """Heuristic scoring: length + complexity - plagiarism penalty."""
    words = len(content.split())
    sentences = max(1, content.count('.') + content.count('!') + content.count('?'))
    avg_word_len = sum(len(w) for w in content.split()) / max(1, words)

    # Base score
    length_score = min(40, words // 5)        # up to 40 pts for length
    complexity_score = min(30, int(avg_word_len * 4))  # up to 30 pts for vocabulary
    structure_score = min(20, sentences * 2)   # up to 20 pts for sentence count
    keyword_bonus = 10 if any(k in content.lower() for k in ["therefore", "however", "example", "conclude"]) else 0

    raw = length_score + complexity_score + structure_score + keyword_bonus
    # Plagiarism penalty
    penalty = int(plagiarism_pct * 0.4)
    return max(20, min(100, raw - penalty))


# ══════════════════════════════════════════════════════════════════════════════
#  PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class AssignmentCreate(BaseModel):
    title: str
    description: str = ""
    instructor_id: str = "I001"
    deadline: str = None

class SubmissionCreate(BaseModel):
    assignment_id: str
    student_id: str
    content: str

class UserCreate(BaseModel):
    name: str
    email: str
    role: str  # 'student' | 'instructor'
    password: str = None


# ══════════════════════════════════════════════════════════════════════════════
#  APP STARTUP
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title="EvalAI API",
    description="Intelligent Assignment Evaluation & Feedback Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: set to your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES – HEALTH
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "ok", "message": "EvalAI API is running 🎓"}

@app.get("/health")
def health():
    return {"status": "healthy", "pdf_support": PDF_SUPPORT, "llm_support": OPENAI_AVAILABLE}


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES – USERS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/users", status_code=201)
def create_user(payload: UserCreate, db: sqlite3.Connection = Depends(get_db)):
    if payload.role not in ("student", "instructor"):
        raise HTTPException(400, "role must be 'student' or 'instructor'")
    uid = payload.role[0].upper() + str(uuid.uuid4())[:6].upper()
    try:
        db.execute(
            "INSERT INTO users(id,name,email,role) VALUES(?,?,?,?)",
            (uid, payload.name, payload.email, payload.role)
        )
        db.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(409, "Email already registered")
    return {"id": uid, "name": payload.name, "email": payload.email, "role": payload.role}

@app.get("/users")
def list_users(role: Optional[str] = None, db: sqlite3.Connection = Depends(get_db)):
    if role:
        rows = db.execute("SELECT id,name,email,role,created_at FROM users WHERE role=?", (role,)).fetchall()
    else:
        rows = db.execute("SELECT id,name,email,role,created_at FROM users").fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES – ASSIGNMENTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/assignments", status_code=201)
def create_assignment(payload: AssignmentCreate, db: sqlite3.Connection = Depends(get_db)):
    if not payload.title.strip():
        raise HTTPException(422, "title cannot be empty")
    aid = "A" + str(uuid.uuid4())[:8].upper()
    db.execute(
        "INSERT INTO assignments(id,title,description,instructor_id,deadline) VALUES(?,?,?,?,?)",
        (aid, payload.title.strip(), payload.description, payload.instructor_id, payload.deadline)
    )
    db.commit()
    return {
        "id": aid,
        "title": payload.title,
        "description": payload.description,
        "instructor_id": payload.instructor_id,
        "deadline": payload.deadline,
    }

@app.get("/assignments")
def list_assignments(instructor_id: Optional[str] = None, db: sqlite3.Connection = Depends(get_db)):
    if instructor_id:
        rows = db.execute("SELECT * FROM assignments WHERE instructor_id=? ORDER BY created_at DESC", (instructor_id,)).fetchall()
    else:
        rows = db.execute("SELECT * FROM assignments ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]

@app.get("/assignments/{assignment_id}")
def get_assignment(assignment_id: str, db: sqlite3.Connection = Depends(get_db)):
    row = db.execute("SELECT * FROM assignments WHERE id=?", (assignment_id,)).fetchone()
    if not row:
        raise HTTPException(404, f"Assignment {assignment_id} not found")
    return dict(row)

@app.delete("/assignments/{assignment_id}", status_code=204)
def delete_assignment(assignment_id: str, db: sqlite3.Connection = Depends(get_db)):
    db.execute("DELETE FROM assignments WHERE id=?", (assignment_id,))
    db.commit()


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES – SUBMISSIONS (core AI/ML logic here)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/submissions", status_code=201)
def create_submission(payload: SubmissionCreate, db: sqlite3.Connection = Depends(get_db)):
    # Validate
    if not payload.content.strip():
        raise HTTPException(422, "content cannot be empty")
    asgn = db.execute("SELECT * FROM assignments WHERE id=?", (payload.assignment_id,)).fetchone()
    if not asgn:
        raise HTTPException(404, f"Assignment {payload.assignment_id} not found")

    # Generate submission ID
    sid = "S" + str(uuid.uuid4())[:8].upper()

    # Save submission
    db.execute(
        "INSERT INTO submissions(id,assignment_id,student_id,content) VALUES(?,?,?,?)",
        (sid, payload.assignment_id, payload.student_id, payload.content)
    )
    db.commit()

    # ── Plagiarism Detection (TF-IDF Cosine Similarity) ────────────────────
    existing_rows = db.execute(
        "SELECT content FROM submissions WHERE assignment_id=? AND id!=?",
        (payload.assignment_id, sid)
    ).fetchall()
    existing_texts = [r["content"] for r in existing_rows]

    max_sim, detail_scores = detect_plagiarism(payload.content, existing_texts)
    plagiarism_pct = round(max_sim, 1)
    plagiarism_label = f"{plagiarism_pct}%"

    # ── Scoring ────────────────────────────────────────────────────────────
    score = score_submission(payload.content, plagiarism_pct)

    # ── Feedback Generation ────────────────────────────────────────────────
    feedback_text = llm_feedback(payload.content, dict(asgn)["title"])
    if not feedback_text:
        feedback_text = rule_based_feedback(payload.content, score)

    # ── Save Feedback ──────────────────────────────────────────────────────
    fid = "F" + str(uuid.uuid4())[:8].upper()
    db.execute(
        """INSERT INTO feedback(id,submission_id,score,plagiarism_risk,feedback_summary,similarity_scores)
           VALUES(?,?,?,?,?,?)""",
        (fid, sid, score, plagiarism_label, feedback_text, json.dumps(detail_scores))
    )
    db.commit()

    return {
        "submission_id": sid,
        "assignment_id": payload.assignment_id,
        "student_id": payload.student_id,
        "plagiarism_risk": plagiarism_label,
        "feedback_summary": feedback_text,
        "score": score,
        "submitted_at": datetime.utcnow().isoformat(),
    }


@app.get("/submissions")
def list_submissions(
    assignment_id: Optional[str] = None,
    student_id: Optional[str] = None,
    db: sqlite3.Connection = Depends(get_db)
):
    query = """
        SELECT s.id as submission_id, s.assignment_id, s.student_id,
               s.submitted_at, f.score, f.plagiarism_risk, f.feedback_summary
        FROM submissions s
        LEFT JOIN feedback f ON f.submission_id = s.id
        WHERE 1=1
    """
    params = []
    if assignment_id:
        query += " AND s.assignment_id=?"
        params.append(assignment_id)
    if student_id:
        query += " AND s.student_id=?"
        params.append(student_id)
    query += " ORDER BY s.submitted_at DESC"
    rows = db.execute(query, params).fetchall()
    return [dict(r) for r in rows]


@app.get("/submissions/{submission_id}")
def get_submission(submission_id: str, db: sqlite3.Connection = Depends(get_db)):
    row = db.execute("""
        SELECT s.*, f.score, f.plagiarism_risk, f.feedback_summary, f.similarity_scores
        FROM submissions s
        LEFT JOIN feedback f ON f.submission_id = s.id
        WHERE s.id=?
    """, (submission_id,)).fetchone()
    if not row:
        raise HTTPException(404, f"Submission {submission_id} not found")
    result = dict(row)
    if result.get("similarity_scores"):
        result["similarity_scores"] = json.loads(result["similarity_scores"])
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES – FILE UPLOAD (Bonus)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    assignment_id: str = "A001",
    student_id: str = "S001",
    db: sqlite3.Connection = Depends(get_db)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")
    if not PDF_SUPPORT:
        raise HTTPException(501, "PDF support not installed. Run: pip install pdfplumber")

    import pdfplumber, io
    content_bytes = await file.read()
    text = ""
    with pdfplumber.open(io.BytesIO(content_bytes)) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"

    if not text.strip():
        raise HTTPException(422, "Could not extract text from PDF")

    # Reuse submission logic
    from fastapi.testclient import TestClient  # lazy import for recursion avoidance
    payload = SubmissionCreate(assignment_id=assignment_id, student_id=student_id, content=text)
    return create_submission(payload, db)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES – ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/analytics/summary")
def analytics_summary(db: sqlite3.Connection = Depends(get_db)):
    total_assignments = db.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
    total_submissions = db.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
    avg_score = db.execute("SELECT AVG(score) FROM feedback").fetchone()[0]
    high_risk = db.execute(
        "SELECT COUNT(*) FROM feedback WHERE CAST(REPLACE(plagiarism_risk,'%','') AS REAL) >= 30"
    ).fetchone()[0]
    return {
        "total_assignments": total_assignments,
        "total_submissions": total_submissions,
        "average_score": round(avg_score or 0, 1),
        "high_plagiarism_count": high_risk,
    }