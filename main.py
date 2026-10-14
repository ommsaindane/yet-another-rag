import argparse
import subprocess
import sys
import uvicorn

from agent.builder import build_rag
from evaluation.dataset import EvaluationDataset
from evaluation.metrics import EvaluationMetrics
from evaluation.runner import EvaluationRunner


def cmd_serve(host: str, port: int, reload: bool) -> None:
    uvicorn.run("app:app", host=host, port=port, reload=reload)


def cmd_sync() -> None:
    build_rag()
    print("\nIndex synchronization complete.")


def cmd_chat() -> None:
    graph = build_rag()
    print("\n" + "=" * 70)
    print("Yet Another RAG — Interactive Streaming CLI (type 'exit' to quit)")
    print("=" * 70 + "\n")

    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        state, token_stream = graph.stream(query)

        print("\nAnswer: ", end="", flush=True)
        for token in token_stream:
            print(token, end="", flush=True)
        print("\n")

        if state["context_is_relevant"]:
            sources = {
                f"{p.metadata.get('filename', p.document_id)} "
                f"({p.metadata.get('section_title', 'Introduction')})"
                for p in state["relevant_parents"]
            }
            print(f"Sources (Local): {', '.join(sorted(sources))}")
        else:
            urls = [w.url for w in state["web_results"] if w.url]
            print(f"Web Fallback (Query: '{state['search_query']}'): {', '.join(urls)}")
        print("-" * 70 + "\n")


def cmd_eval(fast: bool, threshold: float) -> None:
    print("Building RAG pipeline...")
    graph = build_rag()

    print("Loading evaluation dataset...")
    dataset = EvaluationDataset()

    print("Running RAG pipeline on evaluation questions...")
    runner = EvaluationRunner(graph=graph, dataset=dataset)
    test_cases = runner.run()
    print(f"Generated {len(test_cases)} test cases.")

    mode_label = "Fast (Faithfulness + AnswerRelevancy)" if fast else "Full (4 Metrics)"
    print(f"Running DeepEval in sequential mode [{mode_label}]...")
    metrics = EvaluationMetrics(threshold=threshold, fast_mode=fast)
    metrics.evaluate(test_cases)


def cmd_ui(port: int) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "ui/app.py",
            "--server.port",
            str(port),
            "--server.headless",
            "true",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="yet-another-rag",
        description="Corrective RAG (CRAG) Pipeline with Hybrid Search & Cross-Encoder Reranking.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # serve
    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI server.")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload.")

    # chat
    subparsers.add_parser("chat", help="Start an interactive streaming CLI chat session.")

    # sync
    subparsers.add_parser("sync", help="Incrementally sync `data/raw` with ChromaDB and JSON stores.")

    # eval
    eval_parser = subparsers.add_parser("eval", help="Run the DeepEval benchmark suite.")
    eval_parser.add_argument(
        "--fast",
        action="store_true",
        help="Run only Faithfulness and AnswerRelevancy for faster iteration on local GPUs.",
    )
    eval_parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum passing score threshold (default: 0.7).",
    )

    # ui
    ui_parser = subparsers.add_parser("ui", help="Launch the Streamlit console UI.")
    ui_parser.add_argument("--port", type=int, default=8501, help="Streamlit port.")

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(host=args.host, port=args.port, reload=args.reload)
    elif args.command == "chat":
        cmd_chat()
    elif args.command == "sync":
        cmd_sync()
    elif args.command == "eval":
        cmd_eval(fast=args.fast, threshold=args.threshold)
    elif args.command == "ui":
        cmd_ui(port=args.port)


if __name__ == "__main__":
    main()