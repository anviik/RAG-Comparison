"""
merfe GPT, Claude, and Mistral CSVs and print a summary comparison table.
"""
import argparse, re
from pathlib import Path
from collections import defaultdict
import pandas as pd

def _tokens(text):
    return re.sub(r"[^\w\s]", "", str(text).lower()).split()

def token_f1(pred, gold):
    p_toks, g_toks = _tokens(pred), _tokens(gold)
    if not p_toks or not g_toks:
        return 0.0
    common = set(p_toks) & set(g_toks)
    if not common:
        return 0.0
    prec = len(common) / len(p_toks)
    rec  = len(common) / len(g_toks)
    return 2 * prec * rec / (prec + rec)

def answer_contains(pred, gold):
    """1 if any gold token appears in the prediction (loose recall)."""
    g_toks = _tokens(gold)
    p_text = str(pred).lower()
    return float(any(t in p_text for t in g_toks))

# load & score 
def load_and_score(path, model_label):
    df = pd.read_csv(path, dtype=str)
    df["model"] = model_label
    df["token_f1"]      = df.apply(lambda r: token_f1(r["model_answer"], r["ground_truth"]), axis=1)
    df["answer_recall"] = df.apply(lambda r: answer_contains(r["model_answer"], r["ground_truth"]), axis=1)
    df["model_latency_sec"] = pd.to_numeric(df["model_latency_sec"], errors="coerce")
    df["retrieval_confidence"] = pd.to_numeric(df["retrieval_confidence"], errors="coerce")
    return df

def section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")

def main(mistral_path):
    results_dir = Path("./results")
    gpt_path    = results_dir / "gpt_results.csv"
    claude_path = results_dir / "claude_results.csv"

    missing = [p for p in [gpt_path, claude_path, mistral_path] if not Path(p).exists()]
    if missing:
        print("Missing files:")
        for m in missing:
            print(f"  {m}")
        return

    gpt    = load_and_score(gpt_path,    "gpt-4o-mini")
    claude = load_and_score(claude_path, "claude-haiku")
    mistral = load_and_score(mistral_path, "mistral-7b")
    all_df = pd.concat([gpt, claude, mistral], ignore_index=True)

    section("1. Overall Performance (all 750 rows per model)")
    overall = (
        all_df.groupby("model")
        .agg(
            avg_token_f1       =("token_f1",          "mean"),
            avg_answer_recall  =("answer_recall",      "mean"),
            avg_latency_sec    =("model_latency_sec",  "mean"),
            total_rows         =("question_id",        "count"),
        )
        .round(4)
    )
    print(overall.to_string())

    # by question type
    section("2. Token-F1 by Question Type")
    by_type = (
        all_df.groupby(["model", "question_type"])["token_f1"]
        .mean().round(4).unstack("question_type")
    )
    print(by_type.to_string())

    #rag vs no rag
    section("3. RAG vs No-RAG (token_f1)")
    all_df["rag_flag"] = all_df["embedding_type"].apply(
        lambda x: "no-rag" if x == "norag" else "rag"
    )
    rag_comp = (
        all_df.groupby(["model", "rag_flag"])["token_f1"]
        .mean().round(4).unstack("rag_flag")
    )
    print(rag_comp.to_string())

    # base vs fine-tuned embeddings (RAG rows only)
    section("4. Base vs Fine-tuned Embeddings (RAG rows only)")
    rag_only = all_df[all_df["embedding_type"].isin(["base", "finetuned"])]
    emb_comp = (
        rag_only.groupby(["model", "embedding_type"])["token_f1"]
        .mean().round(4).unstack("embedding_type")
    )
    print(emb_comp.to_string())

    # temperature effects
    section("5. Temperature Effect on Token-F1 (RAG rows)")
    temp_df = rag_only.copy()
    temp_df["temperature"] = pd.to_numeric(temp_df["temperature"], errors="coerce")
    temp_comp = (
        temp_df.groupby(["model", "temperature"])["token_f1"]
        .mean().round(4).unstack("temperature")
    )
    print(temp_comp.to_string())

    # retrieval k effect 
    section("6. Retrieval k Effect on Token-F1 (RAG rows)")
    k_df = rag_only.copy()
    k_df["retrieval_k"] = pd.to_numeric(k_df["retrieval_k"], errors="coerce")
    k_comp = (
        k_df.groupby(["model", "retrieval_k"])["token_f1"]
        .mean().round(4).unstack("retrieval_k")
    )
    print(k_comp.to_string())

    # retrieval confidence by model 
    section("7. Avg Retrieval Confidence by Model & Embedding Type (RAG only)")
    conf = (
        rag_only.groupby(["model", "embedding_type"])["retrieval_confidence"]
        .mean().round(4).unstack("embedding_type")
    )
    print(conf.to_string())

    # per-document token-F1 
    section("8. Per-Document Token-F1 (all conditions)")
    doc_f1 = (
        all_df.groupby(["model", "doc_name"])["token_f1"]
        .mean().round(4).unstack("doc_name")
    )
    print(doc_f1.to_string())

    # latency by model 
    section("9. Latency Summary (seconds)")
    lat = (
        all_df.groupby("model")["model_latency_sec"]
        .agg(["mean", "median", "min", "max"])
        .round(3)
    )
    print(lat.to_string())

    # save merged CSV 
    out = results_dir / "all_results_merged.csv"
    all_df.to_csv(out, index=False)
    print(f"\nMerged CSV saved → {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mistral",
        default="./results/mistral_results.csv",
        help="Path to Mistral results CSV from Colab (default: ./results/mistral_results.csv)",
    )
    args = parser.parse_args()
    main(args.mistral)