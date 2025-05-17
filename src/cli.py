import argparse
from .ingest import ingest_document
from .retriever import answer_query


def main():
    parser = argparse.ArgumentParser(description="Concert Tour RAG CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest a new text file")
    p_ingest.add_argument("doc_id", help="Unique ID for the document")
    p_ingest.add_argument("file", help="Path to .txt file to ingest")

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question about ingested tours")
    p_ask.add_argument("query", help="Your question string")

    args = parser.parse_args()
    if args.cmd == "ingest":
        try:
            with open(args.file, encoding="utf-8") as f:
                text = f.read()
            summary = ingest_document(args.doc_id, text)
            print("✅ Ingested! Summary:\n", summary)
        except ValueError as e:
            print(f"❌ {e}")

    elif args.cmd == "ask":
        ans = answer_query(args.query)
        print("🎤 Answer:\n", ans)


if __name__ == "__main__":
    main()