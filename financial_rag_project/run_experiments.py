import os, time, csv, shutil
import numpy as np
import faiss
import fitz
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import openai
import anthropic
from questions import QUESTIONS

load_dotenv()
PDF_DIR            = Path("./pdfs")
RESULTS_DIR        = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)
BASE_MODEL_NAME    = "all-MiniLM-L6-v2"
FINETUNED_MODEL    = "./models/finance-tuned-embeddings"
TEMPERATURES       = [0.0, 0.7]
K_VALUES           = [3, 5]
EMBEDDING_TYPES    = ["base", "finetuned"]
CHUNK_SIZE         = 500   # words
CHUNK_OVERLAP      = 50    # words

COLUMNS = [
    "doc_name", "question_id", "question_type", "question", "ground_truth",
    "model", "model_answer", "model_latency_sec", "retrieval_confidence",
    "sources", "chunks_retrieved", "temperature", "retrieval_k", "embedding_type",
]

openai_client    = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
claude_client    = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# chunking
def extract_chunks(pdf_path):
    doc = fitz.open(str(pdf_path))
    all_words = []
    for page_num, page in enumerate(doc, start=1):
        for word in page.get_text().split():
            all_words.append((word, page_num))
    doc.close()

    chunks, total, start = [], len(all_words), 0
    while start < total:
        end   = min(start + CHUNK_SIZE, total)
        sl    = all_words[start:end]
        text  = " ".join(w for w, _ in sl)
        pages = [p for _, p in sl]
        page  = max(set(pages), key=pages.count)
        chunks.append({"text": text, "page": page})
        if end == total:
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# FAISS index 
def build_index(chunks, encoder):
    embs  = encoder.encode([c["text"] for c in chunks],
                           normalize_embeddings=True, show_progress_bar=False)
    embs  = np.array(embs, dtype=np.float32)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index

def retrieve(question, encoder, index, chunks, k):
    q = np.array(encoder.encode([question], normalize_embeddings=True), dtype=np.float32)
    scores, idxs = index.search(q, k)
    return [{"chunk": chunks[i], "score": float(s)}
            for s, i in zip(scores[0], idxs[0]) if i < len(chunks)]

# prompts
def rag_prompt(retrieved, question):
    ctx = "\n\n".join(
        f"[Excerpt {i}, Page {r['chunk']['page']}]\n{r['chunk']['text']}"
        for i, r in enumerate(retrieved, 1)
    )
    return (
        "You are a financial analyst assistant. Use the following excerpts from "
        "financial documents to answer the question accurately and concisely.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {question}\n\n"
        "Provide a specific, accurate answer based on the context. If the context "
        "does not contain enough information, state that clearly and provide your "
        "best answer from general knowledge."
    )

def norag_prompt(question):
    return (
        "You are a financial analyst assistant.\n\n"
        f"Question: {question}\n\n"
        "Provide a specific, accurate answer. If you don't have enough information, "
        "state that clearly and provide your best answer from general knowledge."
    )

# api calls
MAX_RETRIES = 5

def call_gpt(prompt, temperature):
    for attempt in range(MAX_RETRIES):
        try:
            t0   = time.time()
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=512,
            )
            return resp.choices[0].message.content.strip(), round(time.time() - t0, 3)
        except openai.RateLimitError as e:
            if "insufficient_quota" in str(e):
                raise RuntimeError(
                    "OpenAI quota exhausted — add credits at "
                    "platform.openai.com/settings/billing"
                ) from e
            wait = 2 ** attempt
            print(f"\n  [GPT rate limit] waiting {wait}s…")
            time.sleep(wait)
    raise RuntimeError("GPT failed after max retries")

def call_claude(prompt, temperature):
    for attempt in range(MAX_RETRIES):
        try:
            t0   = time.time()
            resp = claude_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip(), round(time.time() - t0, 3)
        except anthropic.BadRequestError as e:
            if "credit balance is too low" in str(e):
                raise RuntimeError(
                    "Anthropic credits exhausted — add credits at "
                    "console.anthropic.com/settings/billing"
                ) from e
            raise
        except anthropic.RateLimitError:
            wait = 2 ** attempt
            print(f"\n  [Claude rate limit] waiting {wait}s…")
            time.sleep(wait)
    raise RuntimeError("Claude failed after max retries")

#checkpoint
def load_done(path):
    done = set()
    if not path.exists():
        return done
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row["doc_name"], row["question_id"],
                      row["temperature"], row["retrieval_k"], row["embedding_type"]))
    return done

def flush(path, rows, need_header):
    if not rows:
        return
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if need_header:
            w.writeheader()
        w.writerows(rows)

# main pipeline
def _finetuned_model_ready():
    p = Path(FINETUNED_MODEL)
    cfg = p / "config.json"
    return p.is_dir() and cfg.exists() and cfg.stat().st_size > 0

def run():
    print("Loading encoders…")
    base_enc = SentenceTransformer(BASE_MODEL_NAME)
    ft_ready = _finetuned_model_ready()
    if ft_ready:
        print("Loading fine-tuned encoder…")
        ft_enc = SentenceTransformer(FINETUNED_MODEL)
    else:
        print(
            f"[WARN] Fine-tuned model not found at {FINETUNED_MODEL} — "
            "skipping 'finetuned' embedding type. "
            "Add the model and re-run to fill those rows via checkpoint."
        )
        ft_enc = None

    for i, q in enumerate(QUESTIONS):
        q["question_id"] = str(i)

    by_doc = defaultdict(list)
    for q in QUESTIONS:
        by_doc[q["doc"]].append(q)

    gpt_ckpt = RESULTS_DIR / "gpt_checkpoint.csv"
    cld_ckpt = RESULTS_DIR / "claude_checkpoint.csv"

    gpt_done = load_done(gpt_ckpt)
    cld_done = load_done(cld_ckpt)
    gpt_hdr  = not gpt_ckpt.exists()
    cld_hdr  = not cld_ckpt.exists()

    total_gpt = total_cld = 0
    lat_gpt   = lat_cld   = 0.0

    for doc_name, qs in by_doc.items():
        pdf_path = PDF_DIR / doc_name
        if not pdf_path.exists():
            print(f"\n[SKIP] {doc_name} — PDF not found in ./pdfs/")
            continue

        print(f"\n{'='*60}\nDocument : {doc_name}  ({len(qs)} questions)")
        chunks  = extract_chunks(pdf_path)
        print(f"Chunks   : {len(chunks)}")
        print("Building base index…")
        base_idx = build_index(chunks, base_enc)

        encoders = {"base": (base_enc, base_idx)}
        if ft_ready:
            print("Building fine-tuned index…")
            encoders["finetuned"] = (ft_enc, build_index(chunks, ft_enc))

        # RAG conditions
        for emb_type, (enc, idx) in encoders.items():
            for temp in TEMPERATURES:
                for k in K_VALUES:
                    gpt_batch, cld_batch = [], []
                    for qi, q in enumerate(qs, 1):
                        qid = q["question_id"]
                        key = (doc_name, qid, str(temp), str(k), emb_type)
                        print(f"  {emb_type:10s} | temp={temp} | k={k} | "
                              f"{qi:3d}/{len(qs)} {q['question'][:50]}…", end="\r")
                        hits    = retrieve(q["question"], enc, idx, chunks, k)
                        prompt  = rag_prompt(hits, q["question"])
                        conf    = float(np.mean([r["score"] for r in hits])) if hits else 0.0
                        sources = "|".join(f"p{r['chunk']['page']}" for r in hits)
                        base = {
                            "doc_name": doc_name, "question_id": qid,
                            "question_type": q["question_type"],
                            "question": q["question"], "ground_truth": q["ground_truth"],
                            "retrieval_confidence": round(conf, 4), "sources": sources,
                            "chunks_retrieved": len(hits),
                            "temperature": temp, "retrieval_k": k, "embedding_type": emb_type,
                        }
                        if key not in gpt_done:
                            ans, lat = call_gpt(prompt, temp)
                            gpt_batch.append({**base, "model": "gpt-4o-mini",
                                              "model_answer": ans, "model_latency_sec": lat})
                            gpt_done.add(key); total_gpt += 1; lat_gpt += lat
                        if key not in cld_done:
                            ans, lat = call_claude(prompt, temp)
                            cld_batch.append({**base, "model": "claude-haiku",
                                              "model_answer": ans, "model_latency_sec": lat})
                            cld_done.add(key); total_cld += 1; lat_cld += lat
                    print()  
                    flush(gpt_ckpt, gpt_batch, gpt_hdr); gpt_hdr = False
                    flush(cld_ckpt, cld_batch, cld_hdr); cld_hdr = False

        # no rag baseline
        print("  Running no-RAG baseline…")
        gpt_batch, cld_batch = [], []
        for q in qs:
            qid = q["question_id"]
            key = (doc_name, qid, "0.0", "5", "norag")
            base = {
                "doc_name": doc_name, "question_id": qid,
                "question_type": q["question_type"],
                "question": q["question"], "ground_truth": q["ground_truth"],
                "retrieval_confidence": 0.0, "sources": "", "chunks_retrieved": 0,
                "temperature": 0.0, "retrieval_k": 5, "embedding_type": "norag",
            }
            if key not in gpt_done:
                ans, lat = call_gpt(norag_prompt(q["question"]), 0.0)
                gpt_batch.append({**base, "model": "gpt-4o-mini",
                                  "model_answer": ans, "model_latency_sec": lat})
                gpt_done.add(key); total_gpt += 1; lat_gpt += lat
            if key not in cld_done:
                ans, lat = call_claude(norag_prompt(q["question"]), 0.0)
                cld_batch.append({**base, "model": "claude-haiku",
                                  "model_answer": ans, "model_latency_sec": lat})
                cld_done.add(key); total_cld += 1; lat_cld += lat

        flush(gpt_ckpt, gpt_batch, gpt_hdr); gpt_hdr = False
        flush(cld_ckpt, cld_batch, cld_hdr); cld_hdr = False

    shutil.copy(gpt_ckpt, RESULTS_DIR / "gpt_results.csv")
    shutil.copy(cld_ckpt, RESULTS_DIR / "claude_results.csv")

    print(f"\n{'='*60}")
    print(f"GPT    rows: {total_gpt:4d}  avg latency: {lat_gpt/max(total_gpt,1):.2f}s")
    print(f"Claude rows: {total_cld:4d}  avg latency: {lat_cld/max(total_cld,1):.2f}s")
    print("Saved → results/gpt_results.csv  &  results/claude_results.csv")

if __name__ == "__main__":
    run()