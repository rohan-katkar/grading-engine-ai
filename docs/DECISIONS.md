# Architecture Decision Records (ADR)

This document records the key architectural choices, trade-offs, and design rationale for the **EvalAI Grading Engine**.

---

## ADR 01: Transition from Handwritten Paper (OCR) to Typed Digital Inputs

* **Status:** Accepted
* **Context:** Initial designs considered taking optical character recognition (OCR) scans of handwritten student papers.
* **Decision:** Shift to a digital, typed-input Progressive Web App (PWA) client interface.
* **Rationale:**
  * **Elimination of Vision Noise:** OCR introduces character transcription errors, cursive ambiguity, and bad contrast issues that degrade grading accuracy.
  * **Massive Cost & Latency Reduction:** Image tokens are expensive and slow to process (~1,500–3,000 vision tokens per page). Typed text reduces token overhead by 80–90% (~200–400 tokens per answer).
  * **Predictable Evaluation:** Small local LLMs perform significantly better on clean, unambiguous text strings.

---

## ADR 02: RAG-Backed Context Retrieval over LLM Fine-Tuning

* **Status:** Accepted
* **Context:** The system needs to grade student answers against official textbook knowledge and scoring rubrics.
* **Decision:** Keep the textbook outside the model in a local Vector Database (**ChromaDB**) using Retrieval-Augmented Generation (RAG), rather than fine-tuning the LLM.
* **Rationale:**
  * **Hallucination Prevention:** Small local models (2B/8B) can misremember fine details if forced to memorize a entire textbook. Passing retrieved snippets into the prompt forces "open-book" evaluation.
  * **Instant Syllabus Updates:** Updating or swapping a textbook chapter only requires re-indexing vectors in ChromaDB, requiring zero model re-training.
  * **Auditability:** Storing the exact retrieved context snippet (`retrieved_rag_context`) alongside the grade creates an immutable proof trail if a student files a grievance.

---

## ADR 03: Hybrid Deterministic Confidence Engine

* **Status:** Accepted
* **Context:** LLMs suffer from calibration issues and frequently report overconfident scores (e.g., `1.0` or `100%`) even when outputting incomplete or flawed logic.
* **Decision:** Do not rely solely on the LLM's self-reported confidence score. Calculate a composite confidence metric using a **deterministic Python calculation tool**.
* **Rationale:**
  * **Weighted Formulas:** The composite score weighs Rubric Item Coverage (40%), Exact Keyword Matches (30%), Score Boundary Decisiveness (15%), and LLM Self-Report (15%).
  * **Defensive Guardrails:** Includes scale normalization (`/ 100`) and boundary clamping (`max(0.0, min(1.0, val))`) to ensure model hallucinations cannot corrupt downstream routing logic.
  * **Automated Human Escalation:** Any submission with a composite confidence below `0.80` is automatically redirected to a Human-in-the-Loop (HITL) expert review queue.

---

## ADR 04: Local Model Upgrade to Qwen 8B

* **Status:** Accepted
* **Context:** Initial tests used `Qwen3:8b`.
* **Decision:** Standardize on an 8-billion parameter local model (`Qwen 8B`).
* **Rationale:**
  * **Instruction & JSON Adherence:** An 8B model strictly adheres to complex Pydantic JSON schemas without dropping fields or wrapping outputs in conversational text.
  * **Nuanced Logic Extraction:** Significantly better at awarding fair partial credit and evaluating complex student reasoning without hallucination.