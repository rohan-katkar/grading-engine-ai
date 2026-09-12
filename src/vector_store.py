# %% [imports]
import os
import sys
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is in path for standalone/interactive runs
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from chromadb.utils import embedding_functions
from src.ingest_textbook import TextbookIngestor

# Define persistent storage directory for ChromaDB
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")

# Explicitly assign GPU device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚡ [VectorStore] Initializing embeddings on device: {device.upper()}")


# %% [vector_store_class]
class TextbookVectorStore:
    def __init__(self, collection_name: str = "biology_textbook"):
        self.client = chromadb.PersistentClient(path=DB_DIR)
        
        # Load embedding model explicitly on GPU
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device=device
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 500) -> int:
        """
        Ingests structured document chunks into ChromaDB in batches to respect 
        Chroma's maximum batch size limit (5,461) and optimize performance.
        """
        if not documents:
            return 0
            
        total_docs = len(documents)
        print(f"📦 [VectorStore] Starting batch upsert of {total_docs} total chunks (Batch size: {batch_size})...")
        
        # Loop through documents in increments of batch_size
        for i in range(0, total_docs, batch_size):
            batch = documents[i : i + batch_size]
            ids = [doc["id"] for doc in batch]
            texts = [doc["text"] for doc in batch]
            metadatas = [doc["metadata"] for doc in batch]
            
            self.collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            current_batch_num = (i // batch_size) + 1
            total_batches = (total_docs + batch_size - 1) // batch_size
            print(f"  ↳ Upserted batch {current_batch_num}/{total_batches} ({i + len(batch)}/{total_docs} chunks)")
            
        print(f"✅ [VectorStore] Fully indexed all {total_docs} chunks into '{self.collection.name}'.")
        return total_docs

    def seed_data(self):
        """Indexes sample textbook excerpts and rubrics into ChromaDB."""
        documents = [
            (
                "Mitochondria are double-membrane-bound organelles found in eukaryotic cells. "
                "Known as the powerhouse of the cell, they generate ATP through oxidative "
                "phosphorylation and cellular respiration during glucose breakdown."
            ),
            (
                "Photosynthesis occurs in the chloroplasts of plant cells. It absorbs sunlight "
                "to convert carbon dioxide and water into glucose and oxygen."
            ),
            (
                "Ribosomes are microscopic cellular structures made of RNA and proteins. "
                "Their primary function is protein synthesis via translation of mRNA."
            )
        ]
        
        metadatas = [
            {"subject": "Biology", "topic": "Mitochondria", "chapter": "Cell Biology", "question_id": "Q101"},
            {"subject": "Biology", "topic": "Photosynthesis", "chapter": "Plant Physiology", "question_id": "Q102"},
            {"subject": "Biology", "topic": "Ribosomes", "chapter": "Cell Structure", "question_id": "Q103"}
        ]
        
        doc_ids = ["doc_q101_mitochondria", "doc_q102_photosynthesis", "doc_q103_ribosomes"]

        # Add documents to collection
        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=doc_ids
        )
        print(f"✅ Indexed {len(documents)} document chunks into ChromaDB ('{self.collection.name}').")

    def query_context(
        self, 
        question_text: str, 
        subject: Optional[str] = None, 
        topic: Optional[str] = None, 
        n_results: int = 4,        # Request extra candidates to account for potential duplicates
        max_distance: float = 0.45  # Tuned threshold: <= 0.45 keeps true matches, drops noise
    ) -> str:
        """
        Retrieves top matching textbook context.
        Filters out matches exceeding max_distance threshold and removes duplicates.
        """
        where_clause = {}
        if subject and subject != "UNSPECIFIED":
            where_clause["subject"] = subject
        if topic and topic != "UNSPECIFIED":
            where_clause["topic"] = topic

        results = self.collection.query(
            query_texts=[question_text],
            n_results=n_results,
            where=where_clause if where_clause else None,
            include=["documents", "distances"]
        )
        
        if not results or not results.get("documents") or not results["documents"][0]:
            return "No relevant textbook context found."

        retrieved_docs = results["documents"][0]
        distances = results["distances"][0]

        valid_chunks = []
        seen_texts = set()

        for doc, distance in zip(retrieved_docs, distances):
            print(f"DEBUG - Document Distance: {round(distance, 4)}")
            
            # 1. Enforce strict distance threshold
            if distance <= max_distance:
                cleaned_doc = doc.strip()
                # 2. Deduplicate text chunks
                if cleaned_doc not in seen_texts:
                    seen_texts.add(cleaned_doc)
                    valid_chunks.append(cleaned_doc)
            else:
                print(f"⚠️ Chunk excluded (Distance {round(distance, 4)} > Threshold {max_distance})")

        if not valid_chunks:
            closest = round(distances[0], 4)
            return f"No relevant textbook context found (Closest match distance: {closest} exceeded threshold {max_distance})."

        # Return top 2 unique context chunks
        return "\n\n---\n\n".join(valid_chunks[:2])

    def seed_from_pdf(
        self, 
        pdf_path: str, 
        subject: str = "Biology", 
        topic: str = "Cell Biology"
    ):
        """Extracts, chunks, and indexes a PDF textbook into ChromaDB."""
        ingestor = TextbookIngestor(chunk_size=400, chunk_overlap=50)
        
        # Parse PDF into structured chunks
        documents = ingestor.ingest_pdf(
            pdf_path=Path(pdf_path),
            subject=subject,
            topic=topic
        )
        
        # Store in ChromaDB
        self.add_documents(documents)

# %% [unit_test]
if __name__ == "__main__":
    print("Initializing Vector Store...")
    vector_store = TextbookVectorStore()
    
    # Seed sample biology data
    # vector_store.seed_data()

    # Seed directly from PDF
    # pdf_file = "data/Biology-2e_-_WEB.pdf"
    # vector_store.seed_from_pdf(
    #     pdf_path=pdf_file,
    #     subject="Biology",
    #     topic="Cellular Respiration"
    # )
    
    # Test Query 1: Relevant Query
    test_question_1 = "What is the function of mitochondria and ATP?"
    print(f"\n🔍 Querying Vector Store for: '{test_question_1}'")
    retrieved_text_1 = vector_store.query_context(test_question_1)
    print("\n=== RETRIEVED RAG CONTEXT 1 ===")
    print(retrieved_text_1)

    # Test Query 2: Irrelevant / High Distance Query
    test_question_2 = "What is affect labeling concept in CBT?"
    print(f"\n🔍 Querying Vector Store for: '{test_question_2}'")
    retrieved_text_2 = vector_store.query_context(test_question_2)
    print("\n=== RETRIEVED RAG CONTEXT 2 ===")
    print(retrieved_text_2)