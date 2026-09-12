# %% [imports]
import sys
import os
import uuid
import re
from pathlib import Path
from pypdf import PdfReader
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Optional LangChain fallback for clean chunking
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    HAS_LANGCHAIN_SPLITTER = True
except ImportError:
    HAS_LANGCHAIN_SPLITTER = False


def safe_word_boundary_splitter(
    text: str, 
    chunk_size: int = 300, 
    chunk_overlap: int = 40
) -> List[str]:
    """Word-boundary safe text splitter that prevents word slicing and bad joins."""
    if HAS_LANGCHAIN_SPLITTER:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(text)
    
    # Custom word-aware splitter fallback
    words = text.split()
    chunks = []
    current_words: List[str] = []
    current_length = 0

    for word in words:
        # Check if adding word exceeds size limit
        if current_length + len(word) + 1 > chunk_size and current_words:
            chunk_text = " ".join(current_words)
            chunks.append(chunk_text)
            
            # Retain words for overlap without cutting words in half
            overlap_words: List[str] = []
            overlap_len = 0
            for w in reversed(current_words):
                if overlap_len + len(w) + 1 <= chunk_overlap:
                    overlap_words.insert(0, w)
                    overlap_len += len(w) + 1
                else:
                    break
            
            current_words = overlap_words
            current_length = overlap_len

        current_words.append(word)
        current_length += len(word) + 1

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


# %% [textbook_ingestion_class]
class TextbookIngestor:
    """Handles parsing, chunking, and metadata tagging for educational reference materials."""
    
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 40):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_raw_text(
        self, 
        text_content: str, 
        subject: str = "UNSPECIFIED", 
        topic: str = "UNSPECIFIED",
        source_doc: str = "manual_input"
    ) -> List[Dict[str, Any]]:
        """Transforms raw text into structured, metadata-tagged document chunks."""
        clean_text = re.sub(r'\s+', ' ', text_content).strip()
        chunks = safe_word_boundary_splitter(
            clean_text, 
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        
        documents = []
        for idx, chunk in enumerate(chunks):
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source_doc}_{topic}_{idx}"))
            metadata = {
                "chunk_id": doc_id,
                "subject": subject,
                "topic": topic,
                "source_doc": source_doc,
                "chunk_index": idx,
                "total_chunks": len(chunks)
            }
            documents.append({
                "id": doc_id,
                "text": chunk,
                "metadata": metadata
            })
            
        print(f"📖 [Ingestion] Processed '{source_doc}' ({topic}) -> Generated {len(documents)} clean chunk(s).")
        return documents

    def ingest_file(self, file_path: Path, subject: str = "UNSPECIFIED", topic: str = "UNSPECIFIED") -> List[Dict[str, Any]]:
        """Ingests content directly from a plain text file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")
            
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            
        return self.process_raw_text(
            text_content=content, 
            subject=subject, 
            topic=topic, 
            source_doc=path.name
        )

    def ingest_pdf(
        self, 
        pdf_path: Path, 
        subject: str = "UNSPECIFIED", 
        topic: str = "UNSPECIFIED"
    ) -> List[Dict[str, Any]]:
        """Fast text extraction from PDF with live progress tracking."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        reader = PdfReader(str(path))
        total_pages = len(reader.pages)
        extracted_pages = []

        print(f"📄 Reading '{path.name}' ({total_pages} pages)...")
        
        for page_num, page in enumerate(reader.pages, start=1):
            # Print progress every 200 pages
            if page_num % 200 == 0 or page_num == total_pages:
                print(f"  ↳ Extracted {page_num}/{total_pages} pages...")
                
            text = page.extract_text() or ""
            if text.strip():
                extracted_pages.append(text)

        full_text = "\n".join(extracted_pages)
        
        return self.process_raw_text(
            text_content=full_text,
            subject=subject,
            topic=topic,
            source_doc=path.name
        )


# %% [unit_test]
if __name__ == "__main__":
    ingestor = TextbookIngestor(chunk_size=300, chunk_overlap=40)
    
    sample_textbook_chapter = """
    Mitochondria are membrane-bound cell organelles that generate most of the chemical energy needed to power the cell's biochemical reactions. 
    Chemical energy produced by the mitochondria is stored in a small molecule called adenosine triphosphate (ATP). 
    Mitochondria contain their own small chromosomes. Generally, mitochondria, and therefore mitochondrial DNA, are inherited only from the mother.
    Cellular respiration is a set of metabolic reactions and processes that take place in the cells of organisms to convert chemical energy from nutrients into ATP.
    """
    
    docs = ingestor.process_raw_text(
        text_content=sample_textbook_chapter,
        subject="Biology",
        topic="Cell Biology",
        source_doc="Campbell_Biology_Ch7.txt"
    )
    
    print("\n--- Fixed Chunk Output ---")
    for doc in docs:
        print(f"ID: {doc['id']}")
        print(f"Metadata: {doc['metadata']}")
        print(f"Content: {doc['text']}\n" + "-"*40)