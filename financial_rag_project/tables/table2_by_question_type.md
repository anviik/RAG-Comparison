# Table 2: Results by Question Type (temp=0.0, k=5)

| Model            | Embedding  | R1 Factual | Judge Factual | R1 Numerical | Judge Numerical | R1 Reasoning | Judge Reasoning |
| ---------------- | ---------- | ---------- | ------------- | ------------ | --------------- | ------------ | --------------- |
| GPT-4o-mini      | Base       | 0.1239     | 1.9024        | 0.1118       | 1.7297          | 0.097        | 2.2581          |
| GPT-4o-mini      | Fine-tuned | 0.1316     | 1.9634        | 0.1014       | 1.5946          | 0.0992       | 2.2903          |
| Claude-3.5-Haiku | Base       | 0.0548     | 2.0854        | 0.0385       | 1.8378          | 0.0704       | 2.2903          |
| Claude-3.5-Haiku | Fine-tuned | 0.0618     | 2.0488        | 0.0372       | 1.7568          | 0.0723       | 2.4839          |
| Mistral-7B       | Base       | 0.0813     | 1.1951        | 0.0545       | 0.8649          | 0.0894       | 1.8387          |
| Mistral-7B       | Fine-tuned | 0.0864     | 1.3659        | 0.0475       | 0.7297          | 0.0957       | 1.8387          |
