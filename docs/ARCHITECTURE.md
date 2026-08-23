# System Architecture: EvalAI Grading Engine

## Overview
EvalAI is a deterministic, high-throughput exam evaluation and security engine. It utilizes an **Offline-First Exam Delivery Model**, a **Vector-RAG Context Retrieval Pipeline**, and a **LangGraph State Machine** driven by local open-weight LLMs (`Qwen2.5:2b`) to evaluate typed student responses while routing edge cases to human experts.

---

## High-Level Architecture

```mermaid
graph TD
    classDef edge fill:#ebf8ff,stroke:#3182ce,stroke-width:2px,color:#2c5282;
    classDef cloud fill:#fffaf0,stroke:#dd6b20,stroke-width:2px,color:#9c4221;
    classDef core fill:#f0fff4,stroke:#38a169,stroke-width:2px,color:#22543d;
    classDef human fill:#faf5ff,stroke:#805ad5,stroke-width:2px,color:#553c9a;

    subgraph Client ["1. Client Layer (Exam Terminal)"]
        A1["Student Typed Input"] --> A2["Client PWA Local Cache"]
        A2 --> A3["Asymmetric Encrypted Payload"]
    end

    subgraph Ingestion ["2. Ingestion & RAG Pipeline"]
        B1["Cloud Ingestion Queue"] --> B2["RAG Context Matcher"]
        B3[("ChromaDB Vector Store<br/>(Textbooks & Rubrics)")] --> B2
    end

    subgraph LangGraph ["3. Core Evaluation Engine"]
        C1["Qwen2.5:2b Rubric Evaluator"] --> C2{"Grading Confidence >= 80%?"}
        C2 -- Yes --> C3["Auto-Finalize Score"]
    end

    subgraph HITL ["4. Human-in-the-Loop (HITL) Queue"]
        D1["Flag: Low Confidence / Ambiguity"] --> D2[("PostgreSQL Checkpoints DB")]
        D2 --> D3["Expert Review Portal"]
        D3 --> D4["Human Override & Final Score"]
    end

    A3 --> B1
    B2 --> C1
    C2 -- No --> D1

    class A1,A2,A3 edge;
    class B1,B2,B3 cloud;
    class C1,C2,C3 core;
    class D1,D2,D3,D4 human;
```
## Core Operational Phases

1. **Offline-First Client** 
    - **LayerTerminal**: Runs a Progressive Web App (PWA) on local exam center LANs.
    - **Resilience**: Caches state locally so power or network drops do not affect active student exams.
    - **Data Security**: Answers are encrypted locally using asymmetric keys before cloud transmission.

2. **Retrieval-Augmented Generation (RAG) Pipeline**
    - **Storage**: Textbooks and official grading rubrics are split into 500–1000 token chunks and indexed in a local ChromaDB instance using Qwen Text Embeddings.
    - **Runtime**: When a student answer is received, the RAG matcher retrieves the top 2 matching textbook snippets and official question rubrics to form the prompt context.

3. **LangGraph Evaluation Engine**
    - **Execution**: A state machine invokes a local Qwen2.5:2b model via structured JSON output mode.
    - **Routing**: 
        - **Confidence $\ge$ 80%**: The score, rationale, and retrieved context are written directly to the database.
        - **Confidence $<$ 80%**: Execution state pauses and persists to PostgreSQL as REQUIRES_HUMAN_REVIEW.

4. **Human-in-the-Loop (HITL) Exception Queue**
    - Human reviewers evaluate low-confidence or flagged submissions via an administrative dashboard.
    - Submitting an override updates the grade and resumes the thread state in LangGraph.
    - Review logs serve as continuous fine-tuning data for future prompt calibration.