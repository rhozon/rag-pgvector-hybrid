"""CLI.

python -m rag.cli ingest data/handbook [--reset]
python -m rag.cli ask "How many telehealth visits are free?"
python -m rag.cli eval [--k 3] [--min-hit 0.8]
"""
import argparse
import sys
from pathlib import Path

from .config import Settings
from .evals import evaluate_all, load_golden
from .service import build


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("ingest")
    pi.add_argument("folder")
    pi.add_argument("--reset", action="store_true")
    pa = sub.add_parser("ask")
    pa.add_argument("question")
    pa.add_argument("--mode", default="hybrid")
    pe = sub.add_parser("eval")
    pe.add_argument("--golden", default="data/golden.jsonl")
    pe.add_argument("--k", type=int, default=3)
    pe.add_argument("--min-hit", type=float, default=0.0, help="fail if hybrid hit@k is below this")
    args = p.parse_args(argv)

    s = Settings()
    rag, _ = build(s, reset=getattr(args, "reset", False))

    if args.cmd == "ingest":
        total = 0
        for path in sorted(Path(args.folder).glob("*.md")):
            total += rag.retriever.ingest(path.stem, path.read_text(encoding="utf-8"), s.chunk_words, s.chunk_overlap)
        print(f"ingested {total} chunks from {args.folder}")
    elif args.cmd == "ask":
        ans = rag.ask(args.question, mode=args.mode)
        print(ans.text)
        for i, h in enumerate(ans.sources, start=1):
            print(f"  [{i}] {h.doc_id}#{h.idx}  score={h.score:.4f}")
    else:
        results = evaluate_all(rag.retriever, load_golden(args.golden), k=args.k)
        print(f"embedder={s.embedder}  k={args.k}  n={results[0]['n']}")
        print(f"{'mode':<8} {'hit@k':>6} {'MRR':>6}")
        for r in results:
            print(f"{r['mode']:<8} {r['hit_at_k']:>6.2f} {r['mrr']:>6.2f}")
        hybrid = next(r for r in results if r["mode"] == "hybrid")
        for q in hybrid["misses"]:
            print(f"  hybrid miss: {q}")
        if hybrid["hit_at_k"] < args.min_hit:
            print(f"hybrid hit@{args.k} {hybrid['hit_at_k']:.2f} < {args.min_hit}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
