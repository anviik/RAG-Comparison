import textwrap, pandas as pd, numpy as np
from pathlib import Path

MODELS  = ['gpt-4o-mini','claude-haiku','mistral-7b']
LABELS  = {'gpt-4o-mini':'GPT-4o-mini','claude-haiku':'Claude-3.5-Haiku','mistral-7b':'Mistral-7B'}
EMB     = {'base':'Base','finetuned':'Fine-tuned','norag':'No-RAG'}

def load(path):
    df = pd.read_csv(path)
    for c in ['rouge1','rougeL','exact_match','model_latency_sec','retrieval_confidence',
              'temperature','retrieval_k','llm_judge_score','numeric_accuracy']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def to_md(df):
    w = [max(len(str(c)), df[c].astype(str).str.len().max()) for c in df.columns]
    sep  = '| ' + ' | '.join('-'*x for x in w) + ' |'
    head = '| ' + ' | '.join(str(c).ljust(x) for c,x in zip(df.columns,w)) + ' |'
    rows = ['| ' + ' | '.join(str(v).ljust(x) for v,x in zip(r,w)) + ' |'
            for r in df.itertuples(index=False)]
    return '\n'.join([head, sep]+rows)

def save(df, stem, out, title, report_lines):
    df.to_csv(out/f'{stem}.csv', index=False)
    (out/f'{stem}.md').write_text(f'# {title}\n\n{to_md(df)}\n')
    block = f'\n{"="*60}\n{title}\n{"="*60}\n{df.to_string(index=False)}\n'
    report_lines.append(block)
    print(f'  {stem}.csv / .md')

def t1(df, out, rep):
    s = df[(df.temperature==0)&(df.retrieval_k==5)&df.embedding_type.isin(['base','finetuned'])]
    vj = df[df.llm_judge_score!=-1]; vn = df[df.numeric_accuracy!=-1]
    rows = []
    for m in MODELS:
        for e in ['base','finetuned']:
            c  = s[(s.model==m)&(s.embedding_type==e)]
            jc = vj[(vj.temperature==0)&(vj.retrieval_k==5)&(vj.model==m)&(vj.embedding_type==e)]
            nc = vn[(vn.temperature==0)&(vn.retrieval_k==5)&(vn.model==m)&(vn.embedding_type==e)&(vn.question_type=='numerical')]
            rows.append({'Model':LABELS[m],'Embedding':EMB[e],
                'ROUGE-1':round(c.rouge1.mean(),4),'ROUGE-L':round(c.rougeL.mean(),4),
                'Exact Match':round(c.exact_match.mean(),4),
                'LLM Judge (/3)':round(jc.llm_judge_score.mean(),4) if len(jc) else 'N/A',
                'Num Acc (%)':round(nc.numeric_accuracy.mean()*100,2) if len(nc) else 'N/A',
                'Latency (s)':round(c.model_latency_sec.mean(),3)})
    res = pd.DataFrame(rows)
    deltas = []
    for m in MODELS:
        b = res[(res.Model==LABELS[m])&(res.Embedding=='Base')].iloc[0]
        f = res[(res.Model==LABELS[m])&(res.Embedding=='Fine-tuned')].iloc[0]
        def d(a,b):
            try: return round(float(b)-float(a),4)
            except: return 'N/A'
        deltas.append({'Model':f'Δ {LABELS[m]}','Embedding':'ft−base',
            'ROUGE-1':d(b['ROUGE-1'],f['ROUGE-1']),'ROUGE-L':d(b['ROUGE-L'],f['ROUGE-L']),
            'Exact Match':d(b['Exact Match'],f['Exact Match']),
            'LLM Judge (/3)':d(b['LLM Judge (/3)'],f['LLM Judge (/3)']),
            'Num Acc (%)':d(b['Num Acc (%)'],f['Num Acc (%)']),
            'Latency (s)':d(b['Latency (s)'],f['Latency (s)'])})
    full = pd.concat([res, pd.DataFrame(deltas)], ignore_index=True)
    save(full,'table1_main_results',out,'Table 1: Main Results (temp=0.0, k=5)',rep)

def t2(df, out, rep):
    s = df[(df.temperature==0)&(df.retrieval_k==5)&df.embedding_type.isin(['base','finetuned'])]
    vj = s[s.llm_judge_score!=-1]; qt = ['factual','numerical','reasoning']
    rows = []
    for m in MODELS:
        for e in ['base','finetuned']:
            row = {'Model':LABELS[m],'Embedding':EMB[e]}
            for q in qt:
                c = s[(s.model==m)&(s.embedding_type==e)&(s.question_type==q)]
                j = vj[(vj.model==m)&(vj.embedding_type==e)&(vj.question_type==q)]
                row[f'R1 {q.capitalize()}']=round(c.rouge1.mean(),4)
                row[f'Judge {q.capitalize()}']=round(j.llm_judge_score.mean(),4) if len(j) else 'N/A'
            rows.append(row)
    save(pd.DataFrame(rows),'table2_by_question_type',out,'Table 2: Results by Question Type (temp=0.0, k=5)',rep)

def t3(df, out, rep):
    s = df[df.embedding_type.isin(['base','finetuned'])]
    vj = s[s.llm_judge_score!=-1]
    rows = []
    for m in MODELS:
        for e in ['base','finetuned']:
            for t in [0.0,0.7]:
                for k in [3,5]:
                    c = s[(s.model==m)&(s.embedding_type==e)&(s.temperature==t)&(s.retrieval_k==k)]
                    j = vj[(vj.model==m)&(vj.embedding_type==e)&(vj.temperature==t)&(vj.retrieval_k==k)]
                    rows.append({'Model':LABELS[m],'Embedding':EMB[e],'Temp':t,'k':k,
                        'ROUGE-1':round(c.rouge1.mean(),4),'ROUGE-L':round(c.rougeL.mean(),4),
                        'Exact Match':round(c.exact_match.mean(),4),
                        'LLM Judge':round(j.llm_judge_score.mean(),4) if len(j) else 'N/A','N':len(c)})
    save(pd.DataFrame(rows),'table3_ablation',out,'Table 3: Full Ablation',rep)

def t4(df, out, rep):
    rg = df[(df.temperature==0)&(df.retrieval_k==5)]
    nr = df[df.embedding_type=='norag']
    vj_rg = rg[rg.llm_judge_score!=-1]; vj_nr = nr[nr.llm_judge_score!=-1]
    rows = []
    for m in MODELS:
        def r1(sub,e): v=sub[(sub.model==m)&(sub.embedding_type==e)].rouge1.dropna(); return round(v.mean(),4) if len(v) else 'N/A'
        def jd(sub,e): v=sub[(sub.model==m)&(sub.embedding_type==e)].llm_judge_score.dropna(); return round(v.mean(),4) if len(v) else 'N/A'
        rn,rb,rf = r1(nr,'norag'),r1(rg,'base'),r1(rg,'finetuned')
        try: delta=round(float(rf)-float(rn),4)
        except: delta='N/A'
        rows.append({'Model':LABELS[m],'R1 No-RAG':rn,'R1 Base':rb,'R1 Fine-tuned':rf,
            'Δ (norag→ft)':delta,'Judge No-RAG':jd(vj_nr,'norag'),'Judge Fine-tuned':jd(vj_rg,'finetuned')})
    save(pd.DataFrame(rows),'table4_norag_baseline',out,'Table 4: No-RAG Baseline Comparison (temp=0.0, k=5)',rep)

def t5(df, out, rep):
    fails = df[(df.llm_judge_score==0)&(df.temperature==0)&(df.retrieval_k==5)&(df.embedding_type=='finetuned')]
    if fails.empty:
        msg = 'No llm_judge_score==0 rows (all placeholders). Run LLM evaluation first.'
        save(pd.DataFrame({'Note':[msg]}),'table5_error_analysis',out,'Table 5: Error Analysis',rep)
        return
    bkdn = fails.groupby(['model','question_type']).size().reset_index(name='Failures')
    bkdn['Model'] = bkdn.model.map(LABELS)
    bkdn['Type']  = bkdn.question_type.str.capitalize()
    ex = []
    for m in MODELS:
        for _, row in fails[fails.model==m].head(3).iterrows():
            ex.append({'Model':LABELS[m],
                'Question':textwrap.shorten(str(row.question),width=80,placeholder='…'),
                'Ground Truth':textwrap.shorten(str(row.ground_truth),width=40,placeholder='…'),
                'Answer':textwrap.shorten(str(row.model_answer),width=80,placeholder='…'),
                'Score':int(row.llm_judge_score)})
    save(bkdn[['Model','Type','Failures']],'table5a_failure_counts',out,'Table 5a: Failure Counts',rep)
    save(pd.DataFrame(ex),'table5b_failure_examples',out,'Table 5b: Failure Examples',rep)

def main():
    data = Path('results/all_results_evaluated.csv')
    if not data.exists(): print('Run evaluate.py first'); return
    out = Path('tables'); out.mkdir(exist_ok=True)
    df = load(data); print(f'Loaded {len(df)} rows\n')
    rep = []
    t1(df,out,rep); t2(df,out,rep); t3(df,out,rep); t4(df,out,rep); t5(df,out,rep)

    # write full text report
    txt_path = out/'report.txt'
    txt_path.write_text('\n'.join(rep))
    print(f'\n  report.txt saved → {txt_path}')
    print('\n'+'='*50+'\nSAVED:')
    for p in sorted(out.iterdir()): print(f'  {p}')
    print('='*50)

if __name__ == '__main__': main()