import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.patches as mp, seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from scipy import stats

plt.rcParams.update({
    'font.size':15,'axes.titlesize':16,'axes.labelsize':15,
    'xtick.labelsize':13,'ytick.labelsize':13,'legend.fontsize':12,
    'axes.titleweight':'bold','axes.labelweight':'bold',
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'axes.grid.axis':'y','grid.alpha':0.3,
    'grid.linestyle':'--','figure.facecolor':'white','axes.facecolor':'white',
})

MODELS  = ['gpt-4o-mini','claude-haiku','mistral-7b']
COLORS  = {'gpt-4o-mini':'#2196F3','claude-haiku':'#FF9800','mistral-7b':'#4CAF50'}
LABELS  = {'gpt-4o-mini':'GPT-4o-mini','claude-haiku':'Claude-3.5-Haiku','mistral-7b':'Mistral-7B'}
EMB     = {'base':'Base','finetuned':'Fine-tuned','norag':'No-RAG'}

def load(path):
    df = pd.read_csv(path)
    for c in ['rouge1','rougeL','exact_match','model_latency_sec',
              'retrieval_confidence','temperature','retrieval_k',
              'llm_judge_score','numeric_accuracy']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f'  {path}'); return fig

def grouped_bar(means_base, means_ft, stds_base, stds_ft, ylabel, title):
    fig, ax = plt.subplots(figsize=(9,5))
    x, w = np.arange(len(MODELS)), 0.35
    colors = [COLORS[m] for m in MODELS]
    bars_b = ax.bar(x-.5*w, means_base, w, color=colors, alpha=0.75, edgecolor='white',
                    yerr=stds_base, capsize=4, error_kw={'lw':1.2})
    [b.set_hatch('//') for b in bars_b]
    ax.bar(x+.5*w, means_ft, w, color=colors, alpha=1.0, edgecolor='white',
           yerr=stds_ft, capsize=4, error_kw={'lw':1.2})
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in MODELS])
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(handles=[mp.Patch(fc='#888',hatch='//',alpha=.75,label='Base'),
                        mp.Patch(fc='#888',alpha=1.,label='Fine-tuned')], loc='upper right')
    return fig, ax

def fig1(df, out):
    s = df[(df.temperature==0)&(df.retrieval_k==5)&df.embedding_type.isin(['base','finetuned'])]
    nr = df[df.embedding_type=='norag']
    mb = [s[(s.model==m)&(s.embedding_type=='base')].rouge1.dropna() for m in MODELS]
    mf = [s[(s.model==m)&(s.embedding_type=='finetuned')].rouge1.dropna() for m in MODELS]
    fig, ax = grouped_bar(
        [v.mean() for v in mb],[v.mean() for v in mf],
        [v.std()  for v in mb],[v.std()  for v in mf],
        'Mean ROUGE-1',
        'ROUGE-1 by Model and Embedding Type\n(temp=0.0, k=5 — dashed = no-RAG baseline)')
    x = np.arange(len(MODELS))
    for j, m in enumerate(MODELS):
        v = nr[nr.model==m].rouge1.dropna()
        if len(v): ax.hlines(v.mean(), x[j]-.45, x[j]+.45, colors=COLORS[m],
                              linestyles='--', lw=1.6, alpha=.65, zorder=5)
    ax.legend(handles=[mp.Patch(fc='#888',hatch='//',alpha=.75,label='Base'),
                        mp.Patch(fc='#888',alpha=1.,label='Fine-tuned'),
                        mp.Patch(fc='none',ec='#555',ls='--',label='No-RAG')], loc='upper right')
    return save(fig, out/'fig1_main_results.png')

def fig2(df, out):
    valid = df[df.llm_judge_score!=-1]
    if valid.empty: print('  [Fig 2] No judge data — skipping'); return None
    s = valid[(valid.temperature==0)&(valid.retrieval_k==5)&valid.embedding_type.isin(['base','finetuned'])]
    mb = [s[(s.model==m)&(s.embedding_type=='base')].llm_judge_score.dropna() for m in MODELS]
    mf = [s[(s.model==m)&(s.embedding_type=='finetuned')].llm_judge_score.dropna() for m in MODELS]
    fig, ax = grouped_bar(
        [v.mean() for v in mb],[v.mean() for v in mf],
        [v.std()  for v in mb],[v.std()  for v in mf],
        'Mean LLM Judge Score (out of 3)',
        'LLM Judge Score by Model and Embedding Type\n(temp=0.0, k=5)')
    ax.set_ylim(0,3.2)
    return save(fig, out/'fig2_llm_judge.png')

def fig3(df, out):
    nr = df[df.embedding_type=='norag']
    rg = df[(df.temperature==0)&(df.retrieval_k==5)&df.embedding_type.isin(['base','finetuned'])]
    combo = pd.concat([nr,rg])
    fig, ax = plt.subplots(figsize=(11,5))
    x, w = np.arange(len(MODELS)), 0.25
    for i, (emb, alpha) in enumerate([('norag',.45),('base',.72),('finetuned',1.)]):
        means = [combo[(combo.model==m)&(combo.embedding_type==emb)].rouge1.dropna().mean() for m in MODELS]
        bars = ax.bar(x+(i-1)*w, means, w, color=[COLORS[m] for m in MODELS],
                      alpha=alpha, edgecolor='white')
        for b, v in zip(bars, means):
            if not np.isnan(v): ax.text(b.get_x()+b.get_width()/2, b.get_height()+.001,
                                         f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in MODELS])
    ax.set_ylabel('Mean ROUGE-1')
    ax.set_title('RAG vs No-RAG Baseline\n(temp=0.0, k=5 — Mistral has no no-RAG condition)')
    ax.legend(handles=[mp.Patch(fc='#888',alpha=a,label=EMB[e]) for e,a in [('norag',.45),('base',.72),('finetuned',1.)]], loc='upper right')
    return save(fig, out/'fig3_rag_vs_norag.png')

def fig4(df, out):
    s = df[(df.temperature==0)&(df.retrieval_k==5)&df.embedding_type.isin(['base','finetuned'])]
    qt = ['factual','numerical','reasoning']
    fig, ax = plt.subplots(figsize=(9,5))
    x, w = np.arange(len(qt)), 0.25
    for i, m in enumerate(MODELS):
        deltas = [(s[(s.model==m)&(s.embedding_type=='finetuned')&(s.question_type==q)].rouge1.dropna().mean() -
                   s[(s.model==m)&(s.embedding_type=='base')&(s.question_type==q)].rouge1.dropna().mean()) for q in qt]
        ax.bar(x+(i-1)*w, deltas, w, color=COLORS[m], label=LABELS[m], edgecolor='white')
    ax.axhline(0, color='black', lw=1.5, ls='--', alpha=.6, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([q.capitalize() for q in qt])
    ax.set_xlabel('Question Type'); ax.set_ylabel('ROUGE-1 Delta (Fine-tuned − Base)')
    ax.set_title('Fine-tuning Delta by Question Type\n(temp=0.0, k=5)'); ax.legend()
    return save(fig, out/'fig4_finetune_delta_qtype.png')

def _two_subplots(df, xvals, xcol, suptitle, xlabel, path, filter_col=None, filter_val=None):
    s = df[df.embedding_type.isin(['base','finetuned'])]
    if filter_col: s = s[s[filter_col]==filter_val]
    fig, axes = plt.subplots(1,2,figsize=(13,5),sharey=True)
    fig.suptitle(suptitle, fontsize=16, fontweight='bold')
    for ax, emb in zip(axes, ['base','finetuned']):
        for m in MODELS:
            means = [s[(s.model==m)&(s.embedding_type==emb)&(s[xcol]==v)].rouge1.dropna().mean() for v in xvals]
            ax.plot(xvals, means, 'o-', color=COLORS[m], label=LABELS[m], lw=2.5, ms=9)
        ax.set_xticks(xvals); ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel('Mean ROUGE-1' if emb=='base' else '', fontweight='bold')
        ax.set_title(f'{"Base" if emb=="base" else "Fine-tuned"} Embeddings', fontweight='bold')
        if emb=='finetuned': ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f'  {path}'); plt.close(fig); return fig

def fig5(df, out):
    return _two_subplots(df,[0.0,0.7],'temperature',
        'Effect of Temperature on ROUGE-1\n(Lower temp → more focused answers)',
        'Temperature', out/'fig5_temperature.png', filter_col='retrieval_k', filter_val=5)

def fig6(df, out):
    return _two_subplots(df,[3,5],'retrieval_k',
        'Effect of Retrieval Depth k on ROUGE-1\n(temp=0.0)',
        'Retrieval k', out/'fig6_retrieval_k.png', filter_col='temperature', filter_val=0.0)

def fig7(df, out):
    s = df[df.embedding_type.isin(['base','finetuned'])]
    rows = [{'model':m,'emb':e,'lat':s[(s.model==m)&(s.embedding_type==e)].model_latency_sec.dropna().mean(),
             'label':f'{LABELS[m]} ({EMB[e]})'} for m in MODELS for e in ['base','finetuned']]
    ld = pd.DataFrame(rows); ovr = s.model_latency_sec.dropna().mean()
    fig, ax = plt.subplots(figsize=(10,5))
    ax.grid(axis='x',ls='--',alpha=.3); ax.grid(axis='y',visible=False)
    bars = ax.barh(ld.label, ld.lat, color=[COLORS[r.model] for _,r in ld.iterrows()], edgecolor='white')
    for bar, r in zip(bars, ld.itertuples()): bar.set_alpha(1. if r.emb=='finetuned' else .6)
    ax.axvline(ovr, color='#E53935', ls='--', lw=1.8, label=f'Overall mean ({ovr:.2f}s)')
    for b, v in zip(bars, ld.lat): ax.text(v+.05, b.get_y()+b.get_height()/2, f'{v:.2f}s', va='center', fontsize=12, fontweight='bold')
    ax.set_xlabel('Mean Latency (seconds)')
    ax.set_title('Mean Latency by Model and Embedding Type\n* Mistral = local GPU;  GPT/Claude = API calls')
    ax.legend(loc='lower right')
    return save(fig, out/'fig7_latency.png')

def fig8(df, out):
    s = df[(df.temperature==0)&(df.retrieval_k==5)&(df.embedding_type=='finetuned')].copy()
    s['company'] = s.doc_name.str.replace('_2022_10K.pdf','',regex=False).str.replace('_',' ').str.title()
    pivot = s.groupby(['company','model']).rouge1.mean().unstack('model').rename(columns=LABELS)
    pivot = pivot[[LABELS[m] for m in MODELS if LABELS[m] in pivot.columns]]
    fig, ax = plt.subplots(figsize=(11,7))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=pivot.values.mean(),
                ax=ax, linewidths=.5, linecolor='white',
                cbar_kws={'label':'Mean ROUGE-1','shrink':.8},
                annot_kws={'size':13,'weight':'bold'})
    ax.set_title('Per-Company ROUGE-1 Heatmap\n(Fine-tuned embeddings, temp=0.0, k=5)', pad=14)
    ax.set_xlabel(''); ax.set_ylabel('')
    plt.xticks(rotation=20, ha='right'); plt.yticks(rotation=0)
    return save(fig, out/'fig8_company_heatmap.png')

def fig9(df, out):
    v = df[(df.numeric_accuracy!=-1)&(df.question_type=='numerical')&
           (df.temperature==0)&(df.retrieval_k==5)&df.embedding_type.isin(['base','finetuned'])]
    if v.empty: print('  [Fig 9] No numeric_accuracy data — skipping'); return None
    mb = [v[(v.model==m)&(v.embedding_type=='base')].numeric_accuracy.dropna() for m in MODELS]
    mf = [v[(v.model==m)&(v.embedding_type=='finetuned')].numeric_accuracy.dropna() for m in MODELS]
    fig, _ = grouped_bar(
        [x.mean()*100 for x in mb],[x.mean()*100 for x in mf],
        [x.std()*100  for x in mb],[x.std()*100  for x in mf],
        'Numeric Accuracy (%, ±10% tolerance)',
        'Numeric Accuracy on Numerical Questions\n(temp=0.0, k=5)')
    return save(fig, out/'fig9_numeric_accuracy.png')

def fig10(df, out):
    v = df[(df.llm_judge_score!=-1)&(df.embedding_type!='norag')].dropna(subset=['retrieval_confidence','llm_judge_score'])
    if v.empty: print('  [Fig 10] No judge data — skipping'); return None
    fig, ax = plt.subplots(figsize=(9,6))
    for m in MODELS:
        mdf = v[v.model==m]
        if len(mdf)<2: continue
        xv, yv = mdf.retrieval_confidence.values, mdf.llm_judge_score.values
        ax.scatter(xv, yv, alpha=.25, s=18, color=COLORS[m])
        sl, ic, r, *_ = stats.linregress(xv, yv)
        xl = np.linspace(xv.min(), xv.max(), 200)
        ax.plot(xl, sl*xl+ic, color=COLORS[m], lw=2.5, label=f'{LABELS[m]} (r={r:.2f})')
    ax.set_xlabel('Retrieval Confidence'); ax.set_ylabel('LLM Judge Score (0–3)')
    ax.set_title('Retrieval Confidence vs Answer Quality\n(RAG rows only)'); ax.legend()
    return save(fig, out/'fig10_confidence_scatter.png')

def main():
    data = Path('results/all_results_evaluated.csv')
    if not data.exists(): print('Run evaluate.py first'); return
    out = Path('plots'); out.mkdir(exist_ok=True)
    df = load(data); print(f'Loaded {len(df)} rows\n')

    for i, fn in enumerate([fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,fig9,fig10], 1):
        print(f'Figure {i}...'); fn(df, out)

    pdf_path = out / 'all_figures.pdf'
    with PdfPages(pdf_path) as pdf:
        for png in sorted(out.glob('fig*.png')):
            img = plt.imread(png)
            fig, ax = plt.subplots(figsize=(11,8.5))
            ax.imshow(img); ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    print(f'\n  PDF → {pdf_path}')
    print('\n'+'='*50+'\nSAVED:')
    for p in sorted(out.glob('fig*.png')): print(f'  {p}')
    print(f'  {pdf_path}')
    print('='*50)

if __name__ == '__main__': main()