import argparse
from .ingest import ingest_document
from .retriever import answer_query


def main():
    parser = argparse.ArgumentParser(description="Concert Tour RAG CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest a new text file")
    p_ingest.add_argument(
        "-i", "--id",
        dest = "doc_id",
        required = False,
        help = "Optional document ID; if omitted, one is auto-generated")
    p_ingest.add_argument("file", help="Path to .txt file to ingest")

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question about ingested tours")
    p_ask.add_argument("query", help="Your question string")

    args = parser.parse_args()
    if args.cmd == "ingest":
        try:
            with open(args.file, encoding="utf-8") as f:
                text = f.read()
            summary = ingest_document(text, doc_id=args.doc_id)
            print("✅ Ingested! Summary:\n", summary)
        except ValueError as e:
            print(f"❌ {e}")

    elif args.cmd == "ask":
        ans = answer_query(args.query)
        print("🎤 Answer:\n", ans)


if __name__ == "__main__":
    main()