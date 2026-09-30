"""Command line entry point: python -m rag.build_index --db crime_warehouse.db"""
from __future__ import annotations
import argparse
from .document_builder import CrimeDocumentBuilder
from .retriever import FaissRetriever

def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS evidence index from warehouse summaries.")
    parser.add_argument("--db", default="crime_warehouse.db")
    parser.add_argument("--index-dir", default="rag_index")
    parser.add_argument("--download-model", action="store_true", help="Download MiniLM if it is not already cached.")
    args = parser.parse_args()
    documents = CrimeDocumentBuilder(args.db).build()
    retriever = FaissRetriever(args.index_dir, embedder=__import__('rag.embeddings', fromlist=['EmbeddingModel']).EmbeddingModel(allow_download=args.download_model))
    retriever.build(documents)
    print(f"Built FAISS index with {len(documents)} aggregated analytical documents in {args.index_dir}.")

if __name__ == "__main__": main()
