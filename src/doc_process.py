
import re
import pickle
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATORS,
    HF_DATASET_NAME,
    HF_DATA_FILE,
)

"""
Document processing and chunking for PT-BR legal documents.

This script:
1. Loads legal documents from Hugging Face
2. Cleans and chunks them with PT-BR legal structure awareness
3. Saves chunks to artifacts/chunks.pkl for later use
"""


def setup_directories() -> Path:
    """Create artifacts directory if it doesn't exist."""
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    print(f"Artifacts directory: {artifacts_dir}")
    return artifacts_dir


def load_documents() -> List[Dict]:
    """Load documents from Hugging Face dataset."""
    print("\nLoading documents...")
    documents_dataset = load_dataset(
        HF_DATASET_NAME,
        data_files=HF_DATA_FILE,
        split="train"
    )
    documents = documents_dataset.to_list()
    print(f"Loaded {len(documents)} documents")
    return documents


def clean_legal_text(text: str) -> str:
    """Clean legal text while preserving structural line breaks."""
    text = re.sub(r"[\ue000-\uf8ff]", " ", text)
    text = re.sub(
        r"\*Este texto não substitui o publicado oficialmente\.",
        " ",
        text,
        flags=re.I
    )
    text = re.sub(
        r"A visualização deste sistema.*$",
        " ",
        text,
        flags=re.I | re.MULTILINE
    )

    # normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # remove trailing spaces around lines
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    # collapse repeated spaces but keep newlines
    text = re.sub(r"[ \t]+", " ", text)

    # collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def chunk_documents(
    texts: List[Dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: List[str] = None,
) -> List[Dict]:
    if separators is None:
        separators = SEPARATORS
    """Chunk documents respecting legal structure."""
    print("\nChunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        is_separator_regex=True,
        add_start_index=True,
    )

    chunks = []
    for i, item in enumerate(tqdm(texts, desc="Chunking", ncols=100)):
        raw_text = item.get("filedata", "")
        if not raw_text:
            continue

        text = clean_legal_text(raw_text)
        filename = item.get("filename", f"doc_{i}")

        docs = text_splitter.create_documents(
            [text],
            metadatas=[{"doc_id": i, "filename": filename}]
        )

        for k, doc in enumerate(docs):
            chunks.append({
                "doc_id": i,
                "chunk_id": k,
                "filename": filename,
                "start_index": doc.metadata.get("start_index"),
                "text": doc.page_content,
            })

    print(f"Created {len(chunks)} chunks")
    return chunks

#TODO - change to json save
def save_chunks(chunks: List[Dict], artifacts_dir: Path) -> None:
    """Save chunks to pickle file."""
    print("\nSaving chunks...")
    chunks_path = artifacts_dir / "chunks.pkl"
    
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    
    print(f"Saved chunks to: {chunks_path}")
    print(f"Total chunks: {len(chunks)}")


def main():
    """Main pipeline: load, clean, chunk, and save."""
    print("="*60)
    print("Processing Documents for PT-BR Legal Documents")
    print("="*60)

    # Setup
    artifacts_dir = setup_directories()

    # Load documents
    documents = load_documents()

    # Chunk documents
    chunks = chunk_documents(documents)

    # Save chunks
    save_chunks(chunks, artifacts_dir)

    print("\n" + "="*60)
    print("Document processing completed!")
    print("="*60)
    print(f"\nArtifacts saved to: {artifacts_dir}")


if __name__ == "__main__":
    main()
