#!/usr/bin/env python3
"""Clean CCEL plain-text volumes and chunk them into ~fixed-size passages.

Stdlib only (runs anywhere). Produces data/chunks/<corpus>.jsonl, one JSON
object per chunk with text + metadata. Corpus-agnostic: point --src at any
directory of .txt files and give it a --corpus name.
"""
import argparse, json, os, re, sys

# ---- CCEL boilerplate cleaning -------------------------------------------------
UNDERSCORE = re.compile(r"^\s*_{4,}\s*$")
IMAGE_PAGE = re.compile(r"^\s*Image of page\b.*$", re.I)
HEADER_LABEL = re.compile(r"^\s*(Title|Creator\(s\)|Print Basis|Rights|Date|Status|"
                          r"CCEL Subjects|LC Call no|LC Subjects|Source|Editor|"
                          r"Publisher|Contributor|Language|Subjects)\s*:", re.I)
FOOTNOTE_MARK = re.compile(r"\[\d+\]")           # inline footnote refs like [12]
PAGE_REF = re.compile(r"\bp{1,2}\.\s*[ivxlcdm0-9]+\b", re.I)  # leave content, light touch
MULTISPACE = re.compile(r"[ \t]+")

# Tail boilerplate: CCEL appends an "About this document" / indexes block.
TAIL_MARKERS = re.compile(r"^\s*(About This (Book|Document)|Indexes|"
                          r"This document is from the|The Christian Classics Ethereal)", re.I)

# Volume titles for nicer hover labels (Schaff set).
SCHAFF_TITLES = {
    "anf01": "ANF01 Apostolic Fathers, Justin Martyr, Irenaeus",
    "anf02": "ANF02 Fathers of the Second Century",
    "anf03": "ANF03 Latin Christianity: Tertullian",
    "anf04": "ANF04 Fathers of the Third Century: Tertullian IV, Minucius Felix, Commodian, Origen",
    "anf05": "ANF05 Fathers of the Third Century: Hippolytus, Cyprian, Caius, Novatian",
    "anf06": "ANF06 Fathers of the Third Century: Gregory Thaumaturgus, Dionysius, et al.",
    "anf07": "ANF07 Fathers of the Third & Fourth Centuries: Lactantius, et al.",
    "anf08": "ANF08 Twelve Patriarchs, Excerpts, Epistles, Clementina, Apocrypha",
    "anf09": "ANF09 Gospel of Peter, Diatessaron, Apocalypse of Peter",
    "anf10": "ANF10 Bibliographic Synopsis; General Index",
    "npnf101": "NPNF1-01 Augustine: Confessions and Letters",
    "npnf102": "NPNF1-02 Augustine: City of God, Christian Doctrine",
    "npnf103": "NPNF1-03 Augustine: Holy Trinity, Doctrinal & Moral Treatises",
    "npnf104": "NPNF1-04 Augustine: Against Manichaeans and Donatists",
    "npnf105": "NPNF1-05 Augustine: Anti-Pelagian Writings",
    "npnf106": "NPNF1-06 Augustine: Sermon on the Mount, Harmony of the Gospels",
    "npnf107": "NPNF1-07 Augustine: Homilies on the Gospel of John",
    "npnf108": "NPNF1-08 Augustine: Expositions on the Psalms",
    "npnf109": "NPNF1-09 Chrysostom: On the Priesthood, Ascetic Treatises",
    "npnf110": "NPNF1-10 Chrysostom: Homilies on Matthew",
    "npnf111": "NPNF1-11 Chrysostom: Homilies on Acts and Romans",
    "npnf112": "NPNF1-12 Chrysostom: Homilies on Corinthians",
    "npnf113": "NPNF1-13 Chrysostom: Homilies on Galatians–Colossians",
    "npnf114": "NPNF1-14 Chrysostom: Homilies on John and Hebrews",
    "npnf201": "NPNF2-01 Eusebius: Church History, Life of Constantine",
    "npnf202": "NPNF2-02 Socrates, Sozomenus: Ecclesiastical Histories",
    "npnf203": "NPNF2-03 Theodoret, Jerome, Gennadius, Rufinus",
    "npnf204": "NPNF2-04 Athanasius: Select Works and Letters",
    "npnf205": "NPNF2-05 Gregory of Nyssa: Dogmatic Treatises",
    "npnf206": "NPNF2-06 Jerome: Principal Works",
    "npnf207": "NPNF2-07 Cyril of Jerusalem, Gregory Nazianzen",
    "npnf208": "NPNF2-08 Basil: Letters and Select Works",
    "npnf209": "NPNF2-09 Hilary of Poitiers, John of Damascus",
    "npnf210": "NPNF2-10 Ambrose: Selected Works and Letters",
    "npnf211": "NPNF2-11 Sulpitius Severus, Vincent of Lerins, Cassian",
    "npnf212": "NPNF2-12 Leo the Great, Gregory the Great",
    "npnf213": "NPNF2-13 Gregory the Great II, Ephraim Syrus, Aphrahat",
    "npnf214": "NPNF2-14 The Seven Ecumenical Councils",
}

def series_of(volume: str) -> str:
    if volume.startswith("npnf1"): return "NPNF1"
    if volume.startswith("npnf2"): return "NPNF2"
    if volume.startswith("anf"):   return "ANF"
    return "OTHER"

def clean_lines(raw: str):
    """Yield cleaned paragraphs (lists of words joined) from raw CCEL text."""
    lines = raw.splitlines()
    kept = []
    for ln in lines:
        if TAIL_MARKERS.match(ln):
            break  # drop trailing CCEL apparatus
        if UNDERSCORE.match(ln) or IMAGE_PAGE.match(ln) or HEADER_LABEL.match(ln):
            continue
        kept.append(ln)
    # Group into paragraphs on blank lines; join hard-wrapped lines.
    paras, cur = [], []
    for ln in kept:
        if ln.strip() == "":
            if cur:
                paras.append(" ".join(cur)); cur = []
        else:
            cur.append(ln.strip())
    if cur:
        paras.append(" ".join(cur))
    # Final per-paragraph scrub.
    out = []
    for p in paras:
        p = FOOTNOTE_MARK.sub(" ", p)
        p = MULTISPACE.sub(" ", p).strip()
        if len(p.split()) >= 4:        # drop stray fragments / page numbers
            out.append(p)
    return out

def pack_chunks(paras, target_words, overlap_words):
    """Greedily pack paragraphs into ~target_words chunks with paragraph overlap."""
    chunks, cur, cur_n = [], [], 0
    i = 0
    carry = []
    for p in paras:
        n = len(p.split())
        if cur_n + n > target_words and cur:
            chunks.append(" ".join(cur))
            # build overlap carry from the tail of current chunk
            carry, c = [], 0
            for q in reversed(cur):
                carry.insert(0, q); c += len(q.split())
                if c >= overlap_words: break
            cur, cur_n = list(carry), sum(len(q.split()) for q in carry)
        cur.append(p); cur_n += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir of .txt files")
    ap.add_argument("--corpus", required=True, help="corpus name (e.g. schaff)")
    ap.add_argument("--out", default="data/chunks")
    ap.add_argument("--target-words", type=int, default=400)
    ap.add_argument("--overlap-words", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{args.corpus}.jsonl")
    files = sorted(f for f in os.listdir(args.src) if f.endswith(".txt"))
    total = 0
    with open(out_path, "w") as w:
        for fn in files:
            vol = os.path.splitext(fn)[0]
            raw = open(os.path.join(args.src, fn), encoding="utf-8", errors="replace").read()
            paras = clean_lines(raw)
            chunks = pack_chunks(paras, args.target_words, args.overlap_words)
            for j, ch in enumerate(chunks):
                rec = {
                    "id": f"{args.corpus}/{vol}/{j}",
                    "corpus": args.corpus,
                    "volume": vol,
                    "series": series_of(vol),
                    "title": SCHAFF_TITLES.get(vol, vol),
                    "n_words": len(ch.split()),
                    "text": ch,
                }
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += len(chunks)
            print(f"  {vol}: {len(paras)} paras -> {len(chunks)} chunks", file=sys.stderr)
    print(f"[chunk_corpus] {args.corpus}: {total} chunks -> {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
