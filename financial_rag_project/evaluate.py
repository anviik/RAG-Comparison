import re, pandas as pd
from pathlib import Path
from rouge_score import rouge_scorer as rs

def rouge(pred, gold, sc):
    s = sc.score(str(gold), str(pred))
    return s['rouge1'].fmeasure, s['rougeL'].fmeasure

def exact(pred, gold):
    return int(str(pred).strip().lower() == str(gold).strip().lower())

def num_val(text):
    t = str(text).lower().replace(',','').replace('$','').replace('%','')
    for suf, mult in [('trillion',1e12),('billion',1e9),('million',1e6),('thousand',1e3)]:
        m = re.search(rf'([\d.]+)\s*{suf}', t)
        if m: return float(m.group(1)) * mult
    m = re.search(r'(\d+\.?\d*|\.\d+)', t)
    try: return float(m.group(1)) if m else None
    except ValueError: return None

def num_acc(pred, gold):
    gv = num_val(gold)
    if not gv: return -1
    pv = num_val(pred)
    if pv is None: return 0
    return int(abs(pv - gv) / abs(gv) <= 0.10)

def main():
    df = pd.read_csv('results/all_results_merged.csv', dtype=str)
    scorer = rs.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    r1, rL, em, na = [], [], [], []
    for i, row in df.iterrows():
        p, g = str(row['model_answer']), str(row['ground_truth'])
        a, b = rouge(p, g, scorer)
        r1.append(round(a,6)); rL.append(round(b,6)); em.append(exact(p,g))
        na.append(num_acc(p,g) if row['question_type']=='numerical' else -1)
        if (i+1) % 500 == 0: print(f'  {i+1}/{len(df)}')
    df['rouge1'] = r1; df['rougeL'] = rL; df['exact_match'] = em
    df['llm_judge_score'] = -1; df['numeric_accuracy'] = na
    df.to_csv('results/all_results_evaluated.csv', index=False)
    df['rouge1'] = pd.to_numeric(df['rouge1'])
    print(df.groupby('model')['rouge1'].mean().round(4))
    print('Saved → results/all_results_evaluated.csv')

if __name__ == '__main__': main()
