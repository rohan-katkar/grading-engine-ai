# grading-engine-ai

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
- [x] **Security & PII Protection (`src/utils/security.py`, `src/utils/sanitizer.py`)**
  - Added salted PBKDF2-HMAC-SHA256 password hashing and constant-time password verification for RBAC users.
  - Added Presidio-based detection and anonymization for names, email addresses, phone numbers, and student IDs.
  - Added dynamic domain allowlisting for trusted question, rubric, and RAG terms, plus deterministic regex fallbacks.
- [x] **Authentication & User Roles (`src/database.py`, `src/utils/constants.py`)**
  - Added the `users` table with `user_id`, `full_name`, `email`, `password_hash`, `role`, and `created_at` fields.
  - Added a `UserRole` enum that enforces the allowed roles: `STUDENT`, `EXAM_REVIEWER`, `EXAM_CREATOR`, and `ADMIN`.
  - Bound the `users.role` database column to the explicit role enum and seeded default users for each role.
  - Linked foreign keys in `student_submissions` and `human_reviews` to `users(user_id)`.
- [x] **Security Guardrails & Question-Type Routing (`src/workflow.py`, `src/utils/constants.py`)**
  - Added workflow-level PII sanitization before evaluation and prompt-injection scanning with automatic human-review routing.
  - Added deterministic multiple-choice grading with `MCQ`, `LONG_ANSWER`, and `SHORT_ANSWER` question-type definitions.
  - Added final database logging for auto-approved, human-review, security-flagged, and deterministic MCQ evaluations.

---

## 🚀 Remaining Backlog and probable tasks

### Phase 2: Security, Anonymization & Guardrails
- [x] **Task 2: PII Anonymizer & Dynamic Domain Allowlist (`src/utils/sanitizer.py`)**
  - Build dynamic allowlist extractor from question, rubric, and context (keeps terms like *Jonas Salk* intact).
  - Run local Presidio NER and regex stripping on student submissions to redact personal names, emails, phone numbers, and IDs.
  - Added standalone Presidio coverage checks in `src/utils/test_presidio.py`.
- [x] **Task 3: Prompt Injection & Plea Detector Node (`src/workflow.py`)**
  - Added a pre-filter node to flag system overrides, rubric overrides, jailbreak phrases, and requests for maximum marks.
  - Short-circuits flagged runs directly to `NEEDS_HUMAN_REVIEW` and persists the result through the database logger.

### Phase 3: Textbook Vector Store Ingestion
- [x] **Task 4: Automated Textbook Ingestion Script (`src/ingest_textbook.py`)**
  - Build parser for raw OpenStax PDF/Markdown files.
  - Implement word-boundary-safe chunking (400 characters / 50-character overlap) to comply with ChromaDB and embedding limits.
  - Parse and bulk load the OpenStax Biology PDF into ChromaDB; the completed ingestion produced 12,487 persistent chunks.

### Phase 4: Production API & Middleware
- [ ] **Task 5: FastAPI Application (`src/api.py`)**
  - Create endpoints for answer submissions, question creation, and human review queues.
- [ ] **Task 6: JWT Auth & Route Protection**
  - Enforce RBAC middleware to restrict reviewer/admin endpoints.

### Phase 5: Evaluation & Benchmarking
- [ ] **Task 7: End-to-End Evaluation Test Suite**
  - Execute batch test suites across edge cases (perfect answers, partial answers, injections, PII attempts).
- See the [Future Detection Cases Appendix](docs/FUTURE_DETECTION_CASES.md) for the plug-and-play plea and prompt-injection case catalog.

### Notes
- This project is still evolving.
- Architecture and workflows may change as the grading pipeline is refined.
- **Full disclosure**: This project was developed with **AI-assisted pair programming**. All the project decisions from **Archictecture** to **Schema** were made and refined by the author
