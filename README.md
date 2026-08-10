# Formal Language Bias in Large Language Models

This project detects, quantifies, and mitigates formality bias in LLM-generated conversational text. We compare assistant outputs from the LMSYS-Chat-1M dataset against three human conversational baselines using a lexicon-based Formality Rate (FR) metric and a fine-tuned RoBERTa classifier, then apply LoRA fine-tuning on Qwen2.5-3B-Instruct to test a proof-of-concept on reducing the bias we find.

## Overview

LLM assistants tend to default to a more formal, polished register than humans use in comparable conversational contexts, utilizing elevated vocabulary (delve, utilize, furthermore), and structured phrasing even in casual exchanges. This project measures that gap empirically and tests whether targeted fine-tuning can close it without degrading response quality.

**Pipeline:**
1. **Data collection**: Sample assistant turns from LMSYS-Chat-1M (25 models, real user conversations) and three human conversational baselines: BlendedSkillTalk, Reddit (TIFU), and ELI5.
2. **Formality Rate (FR)**: A formal-register lexicon based metric that tracks the frequency of formal language measured in hits per 1,000 tokens, using a two-tier vocabulary built from LLM-characteristic terms.
4. **RoBERTa classifier**: Fine-tuned `s-nlp/roberta-base-formality-ranker`, a RoBERTa model pre-trained on the GYAFC corpus, on the Pavlick formality dataset to produce a continuous, ML-based formality score (`bert_formal_p`) as a second, corroborating measurement for formality.
5. **Bias source analysis**: Break down formality by conversation type (task vs. casual vs. general) and by model family, testing whether instruction-tuned models are more formal than base models.
6. **LoRA mitigation**: Fine-tune Qwen2.5-3B-Instruct with PEFT/LoRA on the lowest-formality subset of the human corpora, aiming to shift model generations toward a more conversational register, measured using our two constructed formality metrics.
7. **Evaluation**: Compare base vs. fine-tuned model outputs on held-out prompts using FR delta, BERT-score delta, and ROUGE-L, plus statistical significance testing (one-sided Mann-Whitney U) on the corpus-level formality gap in order to measure a decrease in formality without a deacrease in other performance metrics.

## Data

** LLM Model Corpus used to measure and detect model formality **
LMSYS-Chat-1M (`lmsys/lmsys-chat-1m`) (https://huggingface.co/datasets/lmsys/lmsys-chat-1m): Requires Hugging Face Token

** Human baselines used to compare to model formality metrics to find distinct differences **
BlendedSkillTalk (`blended_skill_talk`): Scripted dialogue
Reddit (TIFU) (`HuggingFaceGECLM/REDDIT_comments`): Online Reddit corpora
ELI5 (`sentence-transformers/eli5`): Explanatory Q&A writing

** Formality scores used to train RoBERTa classifier (Formality Metric)
Pavlick formality scores (`osyvokon/pavlick-formality-scores`)

### Hugging Face authentication

Set your HF token as an environment variable rather than pasting it into the notebook:

```python
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])
```

In Colab, use the **Secrets** tab (key icon in the sidebar) to store `HF_TOKEN` and load it with `from google.colab import userdata; token = userdata.get('HF_TOKEN')`.

## Models used

- `s-nlp/roberta-base-formality-ranker`: formality classifier (fine-tuned)
- `Qwen/Qwen2.5-3B-Instruct`: base model for LoRA mitigation
