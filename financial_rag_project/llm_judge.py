import time, pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

PROMPT = """You are evaluating an answer to a financial document question.
Question: {question}
Ground Truth: {ground_truth}
Model Answer: {model_answer}
Score the model answer on a 0-3 scale:
3 - Correct and complete
2 - Mostly correct, minor error or missing detail
1 - Partially correct, missing key information
0 - Wrong, irrelevant, or refuses to answer
Reply with a single integer (0, 1, 2, or 3) and nothing else."""

def judge(question, ground_truth, model_answer):
    try:
        resp = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role':'user','content':PROMPT.format(
                question=question, ground_truth=ground_truth, model_answer=model_answer)}],
            max_tokens=5, temperature=0,
        )
        return int(resp.choices[0].message.content.strip()[0])
    except Exception as e:
        print(f'    API error: {e}')
        return -1

def main():
    path = Path('results/all_results_evaluated.csv')
    df = pd.read_csv(path)
    for c in ['temperature','retrieval_k','llm_judge_score']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    target_idx = df[(df.temperature==0.0)&(df.retrieval_k==5)&(df.llm_judge_score==-1)].index
    total = len(target_idx)
    print(f'Scoring {total} rows (~{total*0.5/60:.1f} min)...\n')

    for i, idx in enumerate(target_idx):
        row = df.loc[idx]
        score = judge(row['question'], row['ground_truth'], row['model_answer'])
        df.at[idx, 'llm_judge_score'] = score

        if (i+1) % 50 == 0 or (i+1) == total:
            df.to_csv(path, index=False)
            pct = (i+1)/total*100
            done = df.loc[target_idx[:i+1], 'llm_judge_score']
            avg  = done[done!=-1].mean()
            print(f'  [{i+1}/{total}] {pct:.0f}%  avg score so far: {avg:.2f}  saved.')

        time.sleep(0.5)

    print(f'\nDone. Saved → {path}')
    scored = df.loc[target_idx, 'llm_judge_score']
    print(f'Avg judge score: {scored[scored!=-1].mean():.2f}')
    print(df.loc[target_idx].groupby(['model','embedding_type'])['llm_judge_score'].mean().round(3).to_string())

if __name__ == '__main__': main()