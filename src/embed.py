#!/usr/bin/env python3
"""Embed chunk passages with a SentenceTransformer model (run on the GPU host).

Reads data/chunks/<corpus>.jsonl, writes data/embeddings/<corpus>.npy (float32,
L2-normalized, row-aligned to the jsonl) plus <corpus>.ids.json.
"""
import argparse, json, os, sys, time
import numpy as np

def load_chunks(path):
    ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ids.append(r["id"]); texts.append(r["text"])
    return ids, texts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", default="data/embeddings")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-seq", type=int, default=1024)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    import torch

    corpus = os.path.splitext(os.path.basename(args.chunks))[0]
    os.makedirs(args.out, exist_ok=True)
    ids, texts = load_chunks(args.chunks)
    print(f"[embed] {corpus}: {len(texts)} chunks, model={args.model}", file=sys.stderr)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mk = {"torch_dtype": torch.bfloat16} if dev == "cuda" else {}
    m = SentenceTransformer(args.model, trust_remote_code=True, device=dev,
                            model_kwargs=mk)
    m.max_seq_length = args.max_seq

    t = time.time()
    emb = m.encode(texts, batch_size=args.batch, normalize_embeddings=True,
                   convert_to_numpy=True, show_progress_bar=True)
    emb = emb.astype(np.float32)
    dt = time.time() - t
    print(f"[embed] done: {emb.shape} in {dt:.1f}s ({len(texts)/dt:.0f}/s)", file=sys.stderr)

    np.save(os.path.join(args.out, f"{corpus}.npy"), emb)
    with open(os.path.join(args.out, f"{corpus}.ids.json"), "w") as w:
        json.dump(ids, w)
    print(f"[embed] saved -> {args.out}/{corpus}.npy", file=sys.stderr)

if __name__ == "__main__":
    main()
