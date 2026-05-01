"""
Fine-tunes all-MiniLM-L6-v2 on financial 10-K question-passage pairs using MultipleNegativesRankingLoss then saves the model to ./models/finance-tuned-embeddings/ so run_experiments.py can load it.
"""
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from pathlib import Path

FINANCE_10K_PAIRS = [
    ("What was total revenue for fiscal year 2022?", "Total net revenues for fiscal year 2022 were $43.0 billion representing an increase of 11% versus the prior year period."),
    ("What was net income attributable to shareholders?", "Net income attributable to shareowners of The Coca-Cola Company was $9.5 billion for the fiscal year ended December 31 2022."),
    ("What was earnings per diluted share?", "Earnings per diluted share was $2.19 for fiscal year 2022 compared to $2.25 in the prior year."),
    ("What was operating income for the year?", "Operating income was $10.9 billion for fiscal year 2022 an increase from the prior year driven by strong organic revenue growth."),
    ("What was AWS net sales in 2022?", "Amazon Web Services segment net sales were $80.1 billion for the year ended December 31 2022 representing growth of 29% year over year."),
    ("What was the company's net loss for the year?", "The company reported a net loss of $2.7 billion for fiscal year 2022 primarily due to the pre-tax valuation loss of $12.7 billion from the Rivian investment."),
    ("What was free cash flow for fiscal year 2022?", "Free cash flow was negative $19.7 billion for the trailing twelve months ended December 31 2022 compared to negative $9.1 billion for the prior period."),
    ("What was total net sales for fiscal year 2022?", "Total net sales were $514.0 billion for the fiscal year ended December 31 2022 an increase of 9% compared to $469.8 billion in 2021."),
    ("What was iPhone net sales in fiscal 2022?", "iPhone net sales were $205.5 billion for fiscal year 2022 compared to $191.9 billion in fiscal 2021 representing an increase of 7%."),
    ("What was Apple gross margin percentage?", "Gross margin percentage was 43.3% for fiscal year 2022 compared to 41.8% in fiscal year 2021 reflecting favorable product mix and pricing."),
    ("What was net income for fiscal year 2022?", "Net income was $99.8 billion for fiscal year 2022 compared to $94.7 billion in fiscal year 2021 an increase of 5%."),
    ("What were Services net sales in fiscal 2022?", "Services net sales were $78.1 billion for fiscal year 2022 compared to $68.4 billion in fiscal year 2021 an increase of 14%."),
    ("What was total revenue for Best Buy fiscal 2022?", "Total revenue was $51.8 billion for fiscal year 2022 a decrease of 1.1% compared to fiscal year 2021 revenue of $51.8 billion."),
    ("What was comparable sales growth?", "Comparable sales growth was 10.4% for fiscal year 2022 driven by strong consumer electronics demand and improved in-store and online execution."),
    ("What was JPMorgan net revenue for 2022?", "Total net revenue was $128.7 billion for fiscal year 2022 an increase of 6% compared to $121.7 billion in the prior year."),
    ("What was net interest income for the year?", "Net interest income was $66.3 billion for fiscal year 2022 reflecting the benefit of rising interest rates on rate-sensitive assets across the firm."),
    ("What was the CET1 capital ratio?", "The CET1 capital ratio was 15.0% at December 31 2022 above the firm's regulatory minimum and stress capital buffer requirements."),
    ("What was Netflix total revenues for 2022?", "Total revenues were $31.6 billion for fiscal year 2022 an increase of 6.5% compared to $29.7 billion in fiscal year 2021."),
    ("How many paid memberships did Netflix have?", "Global streaming paid memberships were 231.7 million at December 31 2022 compared to 221.8 million at December 31 2021."),
    ("What was Netflix operating margin in 2022?", "Operating margin was 17.8% for fiscal year 2022 compared to 20.9% in fiscal year 2021 reflecting higher content amortization costs."),
    ("What was Oracle total revenues for fiscal 2022?", "Total revenues were $42.4 billion for fiscal year 2022 an increase of 18% compared to $40.5 billion in fiscal year 2021."),
    ("What was cloud services revenue growth?", "Cloud services and license support revenues were $33.1 billion for fiscal year 2022 an increase of 6% from the prior year period."),
    ("What was Walmart total revenues for fiscal 2022?", "Total revenues were $572.8 billion for fiscal year 2022 an increase of 2.4% compared to $559.2 billion in fiscal year 2021."),
    ("What was Walmart US comparable sales increase?", "Walmart US comparable sales increased 6.4% for fiscal year 2022 driven by strength in grocery and health and wellness categories."),
    ("What was eBay gross merchandise volume?", "Gross merchandise volume was $73.9 billion for fiscal year 2022 a decrease of 15% compared to $87.4 billion in fiscal year 2021."),
    ("How many active buyers did eBay have?", "eBay had approximately 134 million active buyers at December 31 2022 a decrease of 9% compared to 147 million at December 31 2021."),
    ("What were American Express total revenues?", "Total revenues net of interest expense were $52.9 billion for fiscal year 2022 an increase of 25% compared to $42.4 billion in fiscal year 2021."),
    ("What was billed business volume in 2022?", "Billed business volume was $1.5 trillion for fiscal year 2022 an increase of 21% versus the prior year driven by strong card member spending."),
    ("What risks does the company identify related to supply chain?", "The company identifies several supply chain risks including single source suppliers concentration of manufacturing in Asia Pacific geopolitical tensions and component shortages that could disrupt production and delivery."),
    ("How does management describe the competitive landscape?", "Management describes an increasingly competitive landscape with competition from both traditional industry participants and new technology-driven entrants across all business segments and geographies."),
    ("What was the company's strategy for growing advertising revenue?", "The company's advertising strategy leverages its first-party customer data and high-intent shopping signals to deliver targeted advertising solutions for brands and third-party sellers across its platforms."),
    ("How did rising interest rates affect net interest income?", "Rising interest rates had a significant positive impact on net interest income as the company's rate-sensitive assets repriced upward faster than its funding costs resulting in meaningful margin expansion."),
    ("What was the impact of currency headwinds on reported results?", "Currency headwinds reduced reported revenue growth by approximately 5 percentage points as the strong US dollar negatively impacted the translation of international revenues into US dollar terms."),
    ("What was the company's approach to capital return in 2022?", "The company returned capital to shareholders through share repurchases and dividends totaling several billion dollars while maintaining a strong balance sheet and investment grade credit rating."),
    ("How does management describe its cloud transition strategy?", "Management describes an accelerating transition from on-premises licensed software to cloud-based subscription services which provides more predictable recurring revenue streams and deeper customer relationships over time."),
]

OUTPUT_PATH = "./models/finance-tuned-embeddings"
BASE_MODEL  = "all-MiniLM-L6-v2"
BATCH_SIZE  = 8
N_EPOCHS    = 10

def main():
    print(f"Training pairs : {len(FINANCE_10K_PAIRS)}")
    print(f"Base model     : {BASE_MODEL}")
    print(f"Epochs         : {N_EPOCHS}  |  Batch size: {BATCH_SIZE}")
    print(f"Output path    : {OUTPUT_PATH}\n")

    model   = SentenceTransformer(BASE_MODEL)
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    examples = [InputExample(texts=[q, p]) for q, p in FINANCE_10K_PAIRS]
    loader   = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)
    warmup_steps = int(len(loader) * N_EPOCHS * 0.1)

    model.fit(
        train_objectives=[(loader, loss_fn)],
        epochs=N_EPOCHS,
        warmup_steps=warmup_steps,
        output_path=OUTPUT_PATH,
        show_progress_bar=True,
    )
if __name__ == "__main__":
    main()