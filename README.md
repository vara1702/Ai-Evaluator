# AI Evaluator

A web app I built to automate assignment evaluation using machine learning. Students submit their work, and the system scores it, checks for plagiarism, and gives feedback — all without a teacher having to read every single submission manually.

---

## What it does

- Students log in, pick an assignment, write their answer, and hit submit
- The backend runs a TF-IDF similarity check against all previous submissions to catch plagiarism
- A scoring algorithm rates the response based on length, vocabulary, and structure
- Feedback is generated automatically (rule-based, or via GPT if you add an API key)
- Instructors can create assignments and review all submissions with scores and flagged entries

---

## Tech I used

**Frontend** — React 18 + Vite. Plain CSS for styling, no UI library. Just `useState` and `useEffect` for everything.

**Backend** — Python with FastAPI. I chose FastAPI because it auto-generates API docs at `/docs` and handles validation cleanly with Pydantic models.

**Database** — SQLite for local development. Four tables: users, assignments, submissions, and feedback.

**AI/ML** — TF-IDF with cosine similarity, written from scratch using only Python's `math` library. No scikit-learn. The similarity score between a new submission and every existing one for that assignment is computed, and the highest match becomes the plagiarism risk percentage.

---

## Running it locally

You'll need Python 3.11+ and Node.js 18+ installed.

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` to see all the API endpoints.

### Frontend

Open a second terminal:

```bash
cd frontend
echo VITE_API_URL=http://localhost:8000 > .env
npm install
npm run dev
```

Open `http://localhost:5173`.

---

## Project structure

```
project/
├── frontend/
│   ├── src/
│   │   ├── App.jsx       # all UI components
│   │   ├── index.css     # styling
│   │   └── main.jsx      # entry point
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── backend/
│   ├── main.py           # FastAPI routes + AI logic
│   ├── requirements.txt
│   └── Procfile
├── docs/
│   └── er-diagram.svg
└── README.md
```

---

## API endpoints

| Method | Path | What it does |
|--------|------|--------------|
| GET | `/` | Health check |
| POST | `/assignments` | Create a new assignment |
| GET | `/assignments` | List all assignments |
| POST | `/submissions` | Submit + run AI evaluation |
| GET | `/submissions` | List submissions (filter by student or assignment) |
| GET | `/submissions/{id}` | Full detail with feedback |
| GET | `/analytics/summary` | Platform-wide stats |

---

## How the plagiarism detection works

When a student submits, the backend:

1. Tokenizes the text (lowercase, removes punctuation)
2. Computes TF-IDF vectors for the new submission and all existing ones for the same assignment
3. Calculates cosine similarity between the new vector and each existing one
4. Takes the highest similarity value as the plagiarism risk percentage

Anything above 30% gets flagged in the instructor view.

---

## Database schema

Four tables:

- **users** — stores both students and instructors, differentiated by a `role` column
- **assignments** — created by instructors, linked via `instructor_id`
- **submissions** — student responses, linked to both the assignment and the student
- **feedback** — one-to-one with submissions, stores the score, risk %, and feedback text as a separate record

See `docs/er-diagram.svg` for the full diagram.

---

## Deploying

Frontend goes on Vercel — set root directory to `frontend`, add `VITE_API_URL` as an environment variable pointing to your backend URL.

Backend goes on Railway — set root directory to `backend`, it picks up the Procfile automatically.

---

Built by Vara