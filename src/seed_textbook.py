# scripts/seed_textbook.py
import sys
from pathlib import Path

# Ensure project root is in path for standalone/interactive runs
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest_textbook import TextbookIngestor
from src.vector_store import TextbookVectorStore

def seed_database():
    pdf_path = Path("data/Biology-2e_-_WEB.pdf")
    
    print("🚀 Starting ONE-TIME textbook ingestion...")
    ingestor = TextbookIngestor(chunk_size=400, chunk_overlap=50)
    store = TextbookVectorStore()
    
    # 1. Extract & Chunk PDF
    documents = ingestor.ingest_pdf(
        pdf_path=pdf_path,
        subject="Biology",
        topic="General Biology"
    )
    
    # 2. Batch upsert to ChromaDB (persists to ./chroma_db)
    store.add_documents(documents, batch_size=500)
    print("🎉 Ingestion complete! The vector database is saved to disk.")

if __name__ == "__main__":
    seed_database()