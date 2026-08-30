# grading-engine-ai

Work in progress.

- AI-assisted grading workflow for exam submissions
- Retrieval-augmented context using local vector search
- Deterministic confidence scoring before auto-approval
- Human review path for low-confidence or edge-case cases
- SQLite-backed persistence for submissions and evaluation logs
- Local LLM integration via Ollama

## 🗺️ Project Roadmap & Status

### 🟢 Completed Milestones (Core Engine & Data Foundation)
- [x] **Vector Database & RAG Pipeline (`src/vector_store.py`)**
  - Integrated ChromaDB with sentence-transformers for local semantic search.
  - Implemented distance thresholding (`0.70` cutoff) to prevent low-relevance noise from entering the prompt.
- [x] **Deterministic Confidence Engine (`src/workflow.py`)**
  - Built custom mathematical confidence calculator evaluating rubric clarity, keyword density, score ratio decisiveness, and LLM self-confidence.
  - Established strict auto-approval threshold ($\ge 0.80$) vs. human review fallback ($< 0.80$).
- [x] **LangGraph State Machine Architecture (`src/workflow.py`)**
  - Designed fully stateful DAG workflow (`retrieve_context` $\rightarrow$ `evaluate_answer` $\rightarrow$ `compute_confidence` $\rightarrow$ conditional router).
  - Configured structured output constraints using Pydantic models for Ollama / Qwen 8B.
- [x] **Relational Schema & DB Persistence (`src/database.py`)**
  - Aligned database structure strictly with `SCHEMA.md` using SQLAlchemy.
  - Configured UUID primary/foreign keys across `exam_questions`, `student_submissions`, `evaluation_results`, and `human_reviews`.
  - Built relational logging pipeline linking execution states directly to PostgreSQL / SQLite tables.

---

## 🚀 Remaining Backlog and probable tasks

### Phase 1: Authentication & User Roles (RBAC)
- [ ] **Task 1: Database Auth Schema**
  - Add `users` table to `src/database.py` (`user_id`, `email`, `password_hash`, `role`).
  - Enforce explicit roles: `STUDENT`, `EXAM_REVIEWER`, `EXAM_CREATOR`, `ADMIN`.
  - Link foreign keys in `student_submissions` and `human_reviews` to `users(user_id)`.

### Phase 2: Security, Anonymization & Guardrails
- [ ] **Task 2: PII Anonymizer & Dynamic Domain Allowlist (`src/sanitizer.py`)**
  - Build dynamic allowlist extractor from question, rubric, and context (keeps terms like *Jonas Salk* intact).
  - Run local NER/regex stripping on student submissions to redact personal names/IDs.
- [ ] **Task 3: Prompt Injection & Plea Detector Node**
  - Add a pre-filter node in `src/workflow.py` to flag system overrides or emotional pleas.
  - Short-circuit flagged runs directly to `NEEDS_HUMAN_REVIEW`.

### Phase 3: Textbook Vector Store Ingestion
- [ ] **Task 4: Automated Textbook Ingestion Script (`src/ingest_textbook.py`)**
  - Build parser for raw OpenStax PDF/Markdown files.
  - Implement semantic chunking (500 tokens / 50 overlap) and bulk load into ChromaDB.

### Phase 4: Production API & Middleware
- [ ] **Task 5: FastAPI Application (`src/api.py`)**
  - Create endpoints for answer submissions, question creation, and human review queues.
- [ ] **Task 6: JWT Auth & Route Protection**
  - Enforce RBAC middleware to restrict reviewer/admin endpoints.

### Phase 5: Evaluation & Benchmarking
- [ ] **Task 7: End-to-End Evaluation Test Suite**
  - Execute batch test suites across edge cases (perfect answers, partial answers, injections, PII attempts).

### Notes
- This project is still evolving.
- Architecture and workflows may change as the grading pipeline is refined.
- **Full disclosure**: This project was developed with **AI-assisted pair programming**. All the project decisions from **Archictecture** to **Schema** were made and refined by the author
