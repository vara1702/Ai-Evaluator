import { useState, useEffect } from "react";

// where the backend lives — reads from .env or falls back to local
const SERVER = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function callAPI(path, options = {}) {
  const response = await fetch(`${SERVER}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

const Icons = {
  graduate: (
    <svg width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
      <path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/>
    </svg>
  ),
  pencil: (
    <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
    </svg>
  ),
  tick: (
    <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2.2" viewBox="0 0 24 24">
      <polyline points="20 6 9 17 4 12"/>
    </svg>
  ),
  clock: (
    <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
    </svg>
  ),
  flag: (
    <svg width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
      <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" y1="22" x2="4" y2="15"/>
    </svg>
  ),
  plus: (
    <svg width="17" height="17" fill="none" stroke="currentColor" strokeWidth="2.2" viewBox="0 0 24 24">
      <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
    </svg>
  ),
};

function RiskPill({ value }) {
  const num = parseFloat(value);
  const palette =
    num < 20 ? { bg: "#d4f0e8", text: "#1d7a57", label: "Low Risk" }
    : num < 40 ? { bg: "#fef3d0", text: "#92600a", label: "Moderate" }
    : { bg: "#fce4e4", text: "#a32121", label: "High Risk" };
  return (
    <span className="risk-pill" style={{ background: palette.bg, color: palette.text }}>
      {palette.label} · {value}
    </span>
  );
}

function ScoreCircle({ score }) {
  const radius = 26;
  const circumference = 2 * Math.PI * radius;
  const filled = (score / 100) * circumference;
  const color = score >= 70 ? "#4aab8a" : score >= 50 ? "#c9913a" : "#b84040";
  return (
    <div className="score-circle-wrap">
      <svg width="68" height="68" viewBox="0 0 68 68">
        <circle cx="34" cy="34" r={radius} fill="none" stroke="#e8ede9" strokeWidth="5"/>
        <circle cx="34" cy="34" r={radius} fill="none" stroke={color} strokeWidth="5"
          strokeDasharray={`${filled} ${circumference - filled}`}
          strokeLinecap="round" transform="rotate(-90 34 34)"
          style={{ transition: "stroke-dasharray 0.9s ease" }}/>
        <text x="34" y="39" textAnchor="middle" fill={color}
          style={{ fontSize: 15, fontWeight: 700, fontFamily: "'DM Sans', sans-serif" }}>
          {score}
        </text>
      </svg>
      <span className="score-label">score</span>
    </div>
  );
}

function StudentView({ studentId = "S001" }) {
  const [taskList, setTaskList] = useState([]);
  const [mySubmissions, setMySubmissions] = useState([]);
  const [chosenTask, setChosenTask] = useState(null);
  const [answerText, setAnswerText] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [latestResult, setLatestResult] = useState(null);
  const [activeTab, setActiveTab] = useState("write");

  useEffect(() => {
    callAPI("/assignments").then(setTaskList).catch(() =>
      setTaskList([
        { id: "A001", title: "Data Structures Essay", description: "Compare arrays vs linked lists — focus on time and space complexity.", deadline: "2025-08-01" },
        { id: "A002", title: "Algorithm Analysis", description: "Write a report on Big-O notation with real-world examples.", deadline: "2025-08-10" },
        { id: "A003", title: "OS Concepts", description: "Explain the difference between processes and threads with use cases.", deadline: "2025-08-15" },
      ])
    );
    callAPI(`/submissions?student_id=${studentId}`).then(setMySubmissions).catch(() => setMySubmissions([]));
  }, []);

  async function handleSubmit() {
    if (!chosenTask || !answerText.trim()) return;
    setIsSubmitting(true);
    try {
      const result = await callAPI("/submissions", {
        method: "POST",
        body: JSON.stringify({ assignment_id: chosenTask, student_id: studentId, content: answerText }),
      });
      setLatestResult(result);
      setMySubmissions((prev) => [result, ...prev]);
      setAnswerText("");
    } catch {
      const demoResult = {
        submission_id: `S${200 + mySubmissions.length + 1}`,
        plagiarism_risk: `${Math.floor(Math.random() * 30 + 5)}%`,
        feedback_summary: "Your introduction sets a clear direction. Section 2 could benefit from a concrete example to reinforce your argument. The conclusion is concise — consider tying it back to your opening claim.",
        score: Math.floor(Math.random() * 28 + 58),
      };
      setLatestResult(demoResult);
      setMySubmissions((prev) => [{ ...demoResult, assignment_id: chosenTask }, ...prev]);
      setAnswerText("");
    }
    setIsSubmitting(false);
  }

  return (
    <div className="view-wrap">
      <aside className="sidebar">
        <div className="sidebar-header">
          <span className="sidebar-icon teal-icon">{Icons.pencil}</span>
          <div>
            <div className="sidebar-title">Student</div>
            <div className="sidebar-sub">ID · {studentId}</div>
          </div>
        </div>
        <nav className="sidebar-nav">
          <button className={activeTab === "write" ? "nav-item active" : "nav-item"} onClick={() => { setActiveTab("write"); setLatestResult(null); }}>Write & Submit</button>
          <button className={activeTab === "history" ? "nav-item active" : "nav-item"} onClick={() => setActiveTab("history")}>
            My Submissions {mySubmissions.length > 0 && <span className="nav-badge">{mySubmissions.length}</span>}
          </button>
        </nav>
        <div className="sidebar-footer">
          <div className="tip-box">
            <div className="tip-title">Quick tip</div>
            <div className="tip-body">Write at least 3–4 sentences for a better score. Use examples where possible.</div>
          </div>
        </div>
      </aside>

      <main className="main-panel">
        {activeTab === "write" && (
          <div className="panel-content">
            <div className="panel-heading">
              <h2>Choose an assignment</h2>
              <p className="panel-sub">Select the task you want to work on, then write your response below.</p>
            </div>
            <div className="task-list">
              {taskList.map((task) => (
                <div key={task.id} className={`task-card ${chosenTask === task.id ? "task-selected" : ""}`} onClick={() => { setChosenTask(task.id); setLatestResult(null); }}>
                  <div className="task-top">
                    <span className="task-id">{task.id}</span>
                    {task.deadline && <span className="task-due">{Icons.clock} {task.deadline}</span>}
                  </div>
                  <div className="task-name">{task.title}</div>
                  <div className="task-desc">{task.description}</div>
                </div>
              ))}
            </div>
            {chosenTask && (
              <div className="write-section">
                <label className="write-label">Your answer</label>
                <textarea className="write-box" rows={7} placeholder="Write your response here. Try to be clear and use examples where relevant…" value={answerText} onChange={(e) => setAnswerText(e.target.value)}/>
                <div className="write-footer">
                  <span className="word-count">{answerText.trim() ? answerText.trim().split(/\s+/).length : 0} words</span>
                  <button className="submit-btn" onClick={handleSubmit} disabled={isSubmitting || !answerText.trim()}>
                    {isSubmitting ? <span className="spinner-sm" /> : Icons.tick}
                    {isSubmitting ? "Evaluating…" : "Submit for review"}
                  </button>
                </div>
              </div>
            )}
            {latestResult && (
              <div className="result-card">
                <div className="result-header">
                  <span className="result-tag">Evaluation complete</span>
                  <span className="result-id">Ref: {latestResult.submission_id}</span>
                </div>
                <div className="result-body">
                  <ScoreCircle score={latestResult.score} />
                  <div className="result-mid">
                    <RiskPill value={latestResult.plagiarism_risk} />
                    <span className="result-risk-label">similarity check</span>
                  </div>
                  <div className="result-feedback">
                    <div className="feedback-label">Feedback</div>
                    <p className="feedback-text">{latestResult.feedback_summary}</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
        {activeTab === "history" && (
          <div className="panel-content">
            <div className="panel-heading">
              <h2>Your submissions</h2>
              <p className="panel-sub">All work you have submitted so far.</p>
            </div>
            {mySubmissions.length === 0 ? (
              <div className="empty-msg">Nothing submitted yet. Head to Write &amp; Submit to get started.</div>
            ) : (
              <div className="history-list">
                {mySubmissions.map((item, idx) => (
                  <div key={idx} className="history-item">
                    <div className="history-top">
                      <span className="history-ref">{item.submission_id || `Sub #${idx + 1}`}</span>
                      <span className="history-asgn">Assignment {item.assignment_id}</span>
                    </div>
                    <div className="history-scores">
                      {item.score && <ScoreCircle score={item.score} />}
                      {item.plagiarism_risk && <RiskPill value={item.plagiarism_risk} />}
                    </div>
                    {item.feedback_summary && <p className="history-feedback">{item.feedback_summary}</p>}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

function InstructorView({ instructorId = "I001" }) {
  const [assignments, setAssignments] = useState([]);
  const [submissions, setSubmissions] = useState([]);
  const [activeTab, setActiveTab] = useState("overview");
  const [newTask, setNewTask] = useState({ title: "", description: "", deadline: "" });
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [openRow, setOpenRow] = useState(null);

  useEffect(() => {
    callAPI("/assignments").then(setAssignments).catch(() =>
      setAssignments([
        { id: "A001", title: "Data Structures Essay", description: "Compare arrays vs linked lists.", deadline: "2025-08-01" },
        { id: "A002", title: "Algorithm Analysis", description: "Big-O notation report.", deadline: "2025-08-10" },
        { id: "A003", title: "OS Concepts", description: "Processes vs threads.", deadline: "2025-08-15" },
      ])
    );
    callAPI("/submissions").then(setSubmissions).catch(() =>
      setSubmissions([
        { submission_id: "S101", assignment_id: "A001", student_id: "S001", score: 74, plagiarism_risk: "11%", feedback_summary: "Strong introduction with good structure. The conclusion could be expanded to summarise key points more clearly." },
        { submission_id: "S102", assignment_id: "A001", student_id: "S002", score: 58, plagiarism_risk: "41%", feedback_summary: "Several sections closely resemble external sources. Please ensure the work is original and properly cited." },
        { submission_id: "S103", assignment_id: "A002", student_id: "S003", score: 68, plagiarism_risk: "22%", feedback_summary: "Good grasp of the concept overall. Section 2 lacks depth — consider adding worked examples." },
      ])
    );
  }, []);

  async function saveAssignment() {
    if (!newTask.title.trim()) return;
    setSaving(true);
    try {
      const created = await callAPI("/assignments", { method: "POST", body: JSON.stringify({ ...newTask, instructor_id: instructorId }) });
      setAssignments((prev) => [created, ...prev]);
    } catch {
      setAssignments((prev) => [{ id: `A${String(assignments.length + 1).padStart(3, "0")}`, ...newTask }, ...prev]);
    }
    setNewTask({ title: "", description: "", deadline: "" });
    setSaving(false); setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  }

  const avgScore = submissions.length ? Math.round(submissions.reduce((a, s) => a + (s.score || 0), 0) / submissions.length) : 0;
  const flaggedCount = submissions.filter((s) => parseFloat(s.plagiarism_risk) >= 30).length;

  return (
    <div className="view-wrap">
      <aside className="sidebar">
        <div className="sidebar-header">
          <span className="sidebar-icon sage-icon">{Icons.graduate}</span>
          <div>
            <div className="sidebar-title">Instructor</div>
            <div className="sidebar-sub">ID · {instructorId}</div>
          </div>
        </div>
        <nav className="sidebar-nav">
          <button className={activeTab === "overview" ? "nav-item active" : "nav-item"} onClick={() => setActiveTab("overview")}>Overview</button>
          <button className={activeTab === "create" ? "nav-item active" : "nav-item"} onClick={() => setActiveTab("create")}>Add Assignment</button>
          <button className={activeTab === "submissions" ? "nav-item active" : "nav-item"} onClick={() => setActiveTab("submissions")}>
            Submissions {submissions.length > 0 && <span className="nav-badge">{submissions.length}</span>}
          </button>
        </nav>
        <div className="sidebar-footer">
          <div className="stat-mini"><div className="stat-num">{assignments.length}</div><div className="stat-lbl">assignments</div></div>
          <div className="stat-mini"><div className="stat-num" style={{ color: "#b84040" }}>{flaggedCount}</div><div className="stat-lbl">flagged</div></div>
        </div>
      </aside>

      <main className="main-panel">
        {activeTab === "overview" && (
          <div className="panel-content">
            <div className="panel-heading">
              <h2>Hello, welcome back</h2>
              <p className="panel-sub">Here's a snapshot of how things are going.</p>
            </div>
            <div className="stat-row">
              <div className="stat-card"><div className="stat-big">{assignments.length}</div><div className="stat-label">Assignments created</div></div>
              <div className="stat-card"><div className="stat-big">{submissions.length}</div><div className="stat-label">Total submissions</div></div>
              <div className="stat-card"><div className="stat-big">{avgScore}</div><div className="stat-label">Average score</div></div>
              <div className="stat-card flagged"><div className="stat-big">{flaggedCount}</div><div className="stat-label">Flagged for review</div></div>
            </div>
            <div className="panel-heading" style={{ marginTop: 32 }}>
              <h3 style={{ fontSize: 18, fontWeight: 600, color: "#2d4a45" }}>Your assignments</h3>
            </div>
            <div className="task-list">
              {assignments.map((a) => (
                <div key={a.id} className="task-card">
                  <div className="task-top">
                    <span className="task-id">{a.id}</span>
                    {a.deadline && <span className="task-due">{Icons.clock} {a.deadline}</span>}
                  </div>
                  <div className="task-name">{a.title}</div>
                  <div className="task-desc">{a.description}</div>
                </div>
              ))}
            </div>
          </div>
        )}
        {activeTab === "create" && (
          <div className="panel-content">
            <div className="panel-heading">
              <h2>New assignment</h2>
              <p className="panel-sub">Fill in the details below. Students will see this when they log in.</p>
            </div>
            {saved && <div className="saved-banner">{Icons.tick} Assignment saved successfully.</div>}
            <div className="form-block">
              <div className="field">
                <label className="field-label">Title <span className="required">*</span></label>
                <input className="field-input" placeholder="e.g. Introduction to Sorting Algorithms" value={newTask.title} onChange={(e) => setNewTask({ ...newTask, title: e.target.value })}/>
              </div>
              <div className="field">
                <label className="field-label">Submission deadline</label>
                <input className="field-input" type="date" value={newTask.deadline} onChange={(e) => setNewTask({ ...newTask, deadline: e.target.value })}/>
              </div>
              <div className="field full">
                <label className="field-label">Instructions for students</label>
                <textarea className="field-input" rows={4} placeholder="Describe what students need to write about…" value={newTask.description} onChange={(e) => setNewTask({ ...newTask, description: e.target.value })}/>
              </div>
              <button className="submit-btn" onClick={saveAssignment} disabled={saving || !newTask.title.trim()}>
                {saving ? <span className="spinner-sm" /> : Icons.plus}
                {saving ? "Saving…" : "Create assignment"}
              </button>
            </div>
          </div>
        )}
        {activeTab === "submissions" && (
          <div className="panel-content">
            <div className="panel-heading">
              <h2>All submissions</h2>
              <p className="panel-sub">Click any row to read the AI-generated feedback.</p>
            </div>
            <div className="sub-table-wrap">
              <table className="sub-table">
                <thead>
                  <tr><th>Ref</th><th>Assignment</th><th>Student</th><th>Score</th><th>Similarity</th><th></th></tr>
                </thead>
                <tbody>
                  {submissions.map((sub) => (
                    <>
                      <tr key={sub.submission_id} className={openRow === sub.submission_id ? "sub-row open" : "sub-row"} onClick={() => setOpenRow(openRow === sub.submission_id ? null : sub.submission_id)}>
                        <td className="mono-cell">{sub.submission_id}</td>
                        <td>{sub.assignment_id}</td>
                        <td>{sub.student_id}</td>
                        <td>{sub.score && <ScoreCircle score={sub.score} />}</td>
                        <td>{sub.plagiarism_risk && <RiskPill value={sub.plagiarism_risk} />}</td>
                        <td><span className="row-toggle">{openRow === sub.submission_id ? "▲" : "▼"}</span></td>
                      </tr>
                      {openRow === sub.submission_id && (
                        <tr key={`${sub.submission_id}-detail`} className="detail-row">
                          <td colSpan={6}>
                            <div className="detail-content">
                              <span className="detail-label">{Icons.flag} AI feedback</span>
                              <p>{sub.feedback_summary}</p>
                            </div>
                          </td>
                        </tr>
                      )}
                    </>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default function App() {
  const [role, setRole] = useState(null);

  if (!role) {
    return (
      <div className="landing-page">
        <div className="landing-blob blob-1" />
        <div className="landing-blob blob-2" />
        <div className="landing-inner">
          <div className="brand-mark">
            <svg width="44" height="44" viewBox="0 0 40 40" fill="none">
              <circle cx="20" cy="20" r="18" stroke="#4aab8a" strokeWidth="2" fill="#f0faf6"/>
              <path d="M20 10 L28 15 L28 25 L20 30 L12 25 L12 15 Z" stroke="#4aab8a" strokeWidth="1.8" fill="none"/>
              <path d="M20 10 L20 30 M12 15 L28 15 M12 25 L28 25" stroke="#4aab8a" strokeWidth="1.2" opacity="0.5"/>
            </svg>
          </div>
          <h1 className="brand-name">AI Evaluator</h1>
          <p className="brand-tagline">Smarter feedback for every submission —<br/>built by <strong>Vara</strong></p>
          <div className="role-grid">
            <button className="role-btn student-role-btn" onClick={() => setRole("student")}>
              <span className="role-btn-icon">{Icons.pencil}</span>
              <span className="role-btn-title">Student</span>
              <span className="role-btn-sub">Submit work &amp; get instant feedback</span>
            </button>
            <button className="role-btn instructor-role-btn" onClick={() => setRole("instructor")}>
              <span className="role-btn-icon">{Icons.graduate}</span>
              <span className="role-btn-title">Instructor</span>
              <span className="role-btn-sub">Create tasks &amp; review submissions</span>
            </button>
          </div>
          <p className="landing-note">Powered by TF-IDF similarity analysis · Python FastAPI · React</p>
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="topbar-logo" onClick={() => setRole(null)}>
          <svg width="20" height="20" viewBox="0 0 40 40" fill="none">
            <circle cx="20" cy="20" r="18" stroke="#4aab8a" strokeWidth="2.5" fill="none"/>
            <path d="M20 10 L28 15 L28 25 L20 30 L12 25 L12 15 Z" stroke="#4aab8a" strokeWidth="2" fill="none"/>
          </svg>
          <span>AI Evaluator</span>
        </button>
        <span className="topbar-role">{role === "student" ? "Student view" : "Instructor view"}</span>
        <button className="topbar-switch" onClick={() => setRole(null)}>Switch role</button>
      </header>
      <div className="app-body">
        {role === "student" ? <StudentView /> : <InstructorView />}
      </div>
    </div>
  );
}