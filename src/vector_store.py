import os
import chromadb
from chromadb.utils import embedding_functions

# 1. Define persistent storage directory for ChromaDB
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")

class TextbookVectorStore:
    """
    Manages vector storage and similarity retrieval for textbook excerpts 
    and official exam rubrics using ChromaDB.
    """
    def __init__(self, collection_name: str = "biology_textbook"):
        # Initialize persistent disk-backed client
        self.client = chromadb.PersistentClient(path=DB_DIR)
        
        # Default lightweight local sentence transformer for vector embeddings
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        # Initialize or retrieve collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"} # Cosine similarity search
        )

    def seed_data(self):
        """Indexes textbook excerpts and rubrics into ChromaDB."""
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
            {"chapter": "Cell Biology", "topic": "Mitochondria", "question_id": "Q101"},
            {"chapter": "Plant Physiology", "topic": "Photosynthesis", "question_id": "Q102"},
            {"chapter": "Cell Structure", "topic": "Ribosomes", "question_id": "Q103"}
        ]
        
        doc_ids = ["doc_q101_mitochondria", "doc_q102_photosynthesis", "doc_q103_ribosomes"]

        # Add documents to collection
        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=doc_ids
        )
        print(f"✅ Indexed {len(documents)} document chunks into ChromaDB ('{self.collection.name}').")

    def query_context(self, question_text: str, n_results: int = 1, max_distance: float = 0.7) -> str:
        """
        Retrieves top matching textbook context.
        Filters out matches if cosine distance exceeds max_distance threshold.
        """
        results = self.collection.query(
            query_texts=[question_text],
            n_results=n_results,
            include=["documents", "distances"] # Request distance metrics
        )
        
        if results and results.get("documents") and results["documents"][0]:
            doc = results["documents"][0][0]
            distance = results["distances"][0][0]
            
            # Low distance = high similarity. High distance = poor match.
            if distance <= max_distance:
                return doc
            else:
                return f"No relevant textbook context found (Closest match distance: {round(distance, 2)} exceeded threshold {max_distance})."
            
        return "No relevant textbook context found."


# Standalone runner for testing and seeding
if __name__ == "__main__":
    print("Initializing Vector Store...")
    vector_store = TextbookVectorStore()
    
    # Seed data
    vector_store.seed_data()
    
    # Test Query
    test_question = "What is affect labeling concept in CBT?"
    print(f"\n🔍 Querying Vector Store for: '{test_question}'")
    
    retrieved_text = vector_store.query_context(test_question)
    print("\n=== RETRIEVED RAG CONTEXT ===")
    print(retrieved_text)