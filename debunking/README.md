# 7. Debunking

Generates a debunking in the fact, myth, fallacy, fact structure for a statement that
has already been given CARDS and FLICC labels.

| File | Role |
|---|---|
| `sample_claims.py` | Draws one statement per CARDS code from the corpus |
| `extend_climate_fever.py` | Adds two UK specific claims and their evidence to the evidence base |
| `run_debunker.py` | Runs the debunker over the sample |
| `core.py` | The debunker: one method per layer of the response |
| `templates.py`, `definitions.py`, `examples.py` | Prompts, fallacy definitions and worked examples |
| `climate_fever_cards.csv` | Evidence base: Climate FEVER claims with CARDS labels, plus the two additions |

How a response is built:

1. **Fact.** A search agent turns the statement into a question and finds a short factual answer.
2. **Myth.** The statement is paraphrased in 30 words or fewer.
3. **Fallacy.** The FLICC label selects the most similar worked example, which guides an
   explanation of the reasoning error.
4. **Fact.** The CARDS label selects verified claims from the evidence base. The most
   similar one, by cosine similarity, supplies evidence for the closing fact.

`core.py`, `templates.py`, `definitions.py`, `examples.py` and the evidence base are
adapted from the system released by Zanartu, Otmakhova, Cook and Frermann (2024),
"Generative Debunking of Climate Misinformation"
([code](https://huggingface.co/spaces/fzanartu/flicc-agent),
[examples](https://huggingface.co/datasets/fzanartu/CARDSexamples)). The changes made
for this project are listed at the top of `core.py`.

Example outputs are in `data/public/debunkings.csv`. Run instructions are in
[`docs/gpu_runs.md`](../docs/gpu_runs.md).
