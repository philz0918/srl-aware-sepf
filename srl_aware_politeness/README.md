# SRL-Aware Politeness Detection and Explanation

This package provides:
1. preprocessing for SRL-based utterance analysis
2. SRL-aware politeness detection with a directional BERT model
3. post-hoc explanation generation with GPT-OSS
4. a runnable example script

## Files
- `preprocessing.py`: SRL preprocessing and utterance conversion
- `detector.py`: politeness model and prediction pipeline
- `explainer.py`: prompt construction and GPT-OSS explanation
- `example.py`: end-to-end usage example

## Environment variables
Set these before running `example.py`:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="your-url"
export POLITENESS_CKPT_FILE="/path/to/best_model_4_directional.ckpt"
export GPT_OSS_MODEL="gpt-oss-120b"
```

## Run
```bash
python example.py
```
