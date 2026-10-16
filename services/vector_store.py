import hashlib
import re

import chromadb

from config import CHROMA_DIR
from services.document_processor import split_into_chunks
from services.embedding_service import embed_texts


COLLECTION_NAME = "smartstudy_documents"


def _collection():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def make_document_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def add_document(
    session_id: str,
    filename: str,
    content: bytes,
    text: str,
    pages: int,
    config,
    ocr_used: bool = False,
):
    chunks = split_into_chunks(text)
    if not chunks:
        raise ValueError("The document contains no text to index.")
    digest = make_document_hash(content)
    collection = _collection()
    existing = collection.get(
        where={"$and": [{"session_id": session_id}, {"file_hash": digest}]},
        include=[],
    )
    if existing["ids"]:
        return digest, len(existing["ids"]), True

    embeddings = embed_texts(chunks, config)
    ids = [f"{session_id}:{digest}:{index}" for index in range(len(chunks))]
    metadatas = []
    for index, chunk in enumerate(chunks):
        page_match = re.search(r"\[Page (\d+)\]", chunk)
        metadatas.append({
            "session_id": session_id,
            "file_hash": digest,
            "filename": filename,
            "page": int(page_match.group(1)) if page_match else 0,
            "chunk_index": index,
            "page_count": pages,
            "size_bytes": len(content),
            "ocr_used": ocr_used,
        })
    collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    return digest, len(chunks), False


def list_documents(session_id: str):
    result = _collection().get(
        where={"session_id": session_id},
        include=["metadatas"],
    )
    documents = {}
    for metadata in result["metadatas"]:
        file_hash = metadata["file_hash"]
        if file_hash not in documents:
            documents[file_hash] = {
                "filename": metadata["filename"],
                "hash": file_hash,
                "size": metadata.get("size_bytes", 0),
                "pages": metadata.get("page_count", 1),
                "chunks": 0,
                "ocr": metadata.get("ocr_used", False),
            }
        documents[file_hash]["chunks"] += 1
    return sorted(documents.values(), key=lambda item: item["filename"].lower())


def query_documents(session_id: str, question: str, config, limit: int = 5):
    collection = _collection()
    count = collection.count()
    if not count:
        return []
    query_embedding = embed_texts([question], config, task_type="RETRIEVAL_QUERY")[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(limit, count),
        where={"session_id": session_id},
        include=["documents", "metadatas", "distances"],
    )
    matches = []
    for document, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        matches.append({"text": document, **metadata, "distance": distance})
    return matches


def get_document_chunks(session_id: str, file_hash: str):
    collection = _collection()
    result = collection.get(
        where={"$and": [{"session_id": session_id}, {"file_hash": file_hash}]},
        include=["documents", "metadatas"],
    )
    items = sorted(
        zip(result["documents"], result["metadatas"]),
        key=lambda item: item[1].get("chunk_index", 0),
    )
    return [item[0] for item in items]


def delete_document(session_id: str, file_hash: str):
    _collection().delete(where={"$and": [{"session_id": session_id}, {"file_hash": file_hash}]})
