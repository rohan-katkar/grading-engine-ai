# Database Schema Specification

## Schema Design

The system uses PostgreSQL for relational data persistence and audit tracking, paired with ChromaDB for vector-based document retrieval.

[exam_questions] ──► [student_submissions] ──► [evaluation_results] ──► [human_reviews]

---

## PostgreSQL DDL Specifications

```sql
-- 1. QUESTIONS & RUBRICS
CREATE TABLE exam_questions (
    question_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject VARCHAR(100) NOT NULL,
    topic VARCHAR(150) NOT NULL,
    question_text TEXT NOT NULL,
    max_marks INT NOT NULL DEFAULT 10,
    official_rubric JSONB NOT NULL,
    vector_chunk_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. STUDENT SUBMISSIONS
CREATE TABLE student_submissions (
    submission_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID NOT NULL,
    question_id UUID REFERENCES exam_questions(question_id) ON DELETE CASCADE,
    typed_answer TEXT NOT NULL,
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processing_status VARCHAR(50) DEFAULT 'PENDING'
    -- Status options: 'PENDING', 'AUTO_GRADED', 'REQUIRES_HUMAN_REVIEW', 'HUMAN_FINALIZED'
);

-- 3. AI EVALUATION LOGS
CREATE TABLE evaluation_results (
    evaluation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id UUID REFERENCES student_submissions(submission_id) ON DELETE CASCADE,
    ai_score NUMERIC(4, 2),
    ai_confidence NUMERIC(3, 2),
    ai_reasoning_feedback TEXT,
    model_version VARCHAR(50) DEFAULT 'qwen3:8b',
    retrieved_rag_context TEXT,
    evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. HUMAN-IN-THE-LOOP OVERRIDE QUEUE
CREATE TABLE human_reviews (
    review_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id UUID REFERENCES student_submissions(submission_id) ON DELETE CASCADE,
    evaluation_id UUID REFERENCES evaluation_results(evaluation_id),
    flag_reason VARCHAR(100) NOT NULL,
    reviewer_id UUID,
    final_human_score NUMERIC(4, 2),
    reviewer_comments TEXT,
    review_status VARCHAR(30) DEFAULT 'UNASSIGNED',
    reviewed_at TIMESTAMP WITH TIME ZONE
);
```