# AIPI 590: Project 2: Meeting Alert Agent
### Author: Matana Pornluanprasert

A real-time meeting monitoring agent that listens to live audio, transcribes speech, and fires topic alerts when specific financial terms are discussed. Built entirely with local, offline models — no cloud APIs required.

---

## Motivation and Task Context

In earnings calls and financial meetings, decision-relevant topics (dividend policy, capex, gross margin, etc.) can surface at any point during a long call. Participants who are not fully focused can miss these moments. The agent monitors the audio stream continuously and alerts the user the instant a target topic is detected.

The core challenge is latency vs. accuracy trade-off: rule-based keyword matching is fast but fragile under noisy transcription; LLM-based detection is more robust but too slow for real-time use on every 15-second chunk. This project evaluates four detector configurations to identify the best practical combination for local hardware.

---

## Architecture

The app runs three concurrent threads, each feeding the next via a shared queue.

```
Microphone -> [Listener Thread] -> audio_queue
                                        |
                               [Scribe Thread]    <- Whisper / PocketSphinx
                                        |
                               transcription_queue
                                        |
                               [Analyst Thread]   <- Keywords / Embeddings / LLM
                                        |
                               Streamlit UI (topic badges + transcript log)
```

### Perception — Speech-to-Text

| Model | Type | Notes |
|---|---|---|
| Whisper (small) | Deep Learning | `faster-whisper` with CUDA; rolling 15-s context window reduces boundary errors |
| PocketSphinx | Non-DL | Pure CPU; fast but high WER on natural speech |

**Design rationale:** Whisper `small` was chosen over `base` or `tiny` for its substantially lower WER on financial vocabulary without exceeding real-time speed on an RTX 4060. PocketSphinx is kept as the non-DL baseline to demonstrate how much transcription quality affects downstream detection.

Whisper is loaded inside a daemon thread with `torch.cuda.init()` pre-warming the CUDA DLLs in the main thread. This avoids a `ctranslate2` deadlock that occurs on Windows when the DLL loader lock is held during model init.

### Planning — Topic Detection

| Mode | Type | Notes |
|---|---|---|
| Non-DL (Keywords) | Non-DL | Exact word/bigram match against topic text; single-hit threshold chosen because PocketSphinx frequently drops one word in a multi-word term |
| Transformer (Embeddings) | Deep Learning | `all-MiniLM-L6-v2` cosine similarity > 0.45; topic embeddings cached across chunks |
| LLM (Ollama - Qwen 2.5) | Deep Learning | Best accuracy; binary TRUE/FALSE prompt; model-calibrated prompt template |
| LLM (Ollama - Gemma 2) | Deep Learning | Slightly lower precision; requires explicit strictness instructions in prompt |

**Design rationale:** Four planning modes are offered because different use cases have different latency budgets. Keywords fire in under 1 ms and run on every chunk. Embeddings take ~5 ms per chunk and offer better generalisation than exact matching. LLMs take ~1-2 s per topic per chunk and are used only when accuracy matters more than latency. All models run fully locally via Ollama, preserving meeting privacy.

---

## Evaluation Approach

Evaluation uses three sections, all operating on the **same stratified sample** of 15-second audio chunks drawn from five annotated earnings-call recordings.

### Test Data

- **Audio:** 5 earnings call MP3 files (~30-60 min each) in `testset_audio/`
- **Labels:** Each file chunked into 15-second windows; each chunk labeled `topic_present = 1` if it contains any of 10 target topics (dividend, gross margin, inventory, capital allocation, net income, operating income, pricing, capex, investment portfolio, sales growth)
- **Ground truth:** `testset_transcript/` — `.aligned.nlp` files converted to `_labeled.json` via `transcript_processing.py`

### Sampling Strategy

A stratified sample of up to 40 chunks per file is drawn, keeping **all positive chunks** when they fit within the budget, then filling the remaining slots with evenly-spaced negatives. This produces ~50% positive rate across the ~200-chunk shared evaluation set, ensuring Precision/Recall/F1 metrics are meaningful and directly comparable across all sections.

The LLM budget is capped at 50 chunks (also stratified, ~50% positive). **The same 50 chunks are used for both Section 2 and Section 3 LLM rows**, so the Section 2 vs Section 3 gap isolates the pure cost of transcription errors.

Sampling decisions are saved to `eval_sample_cache.json`. Delete this file to re-sample.

### Section 1 — Perception: Word Error Rate

Whisper and PocketSphinx transcribe each audio chunk in the shared sample. WER is measured against the ground-truth text.

### Section 2 — Planning: Precision/Recall/F1 on Ground-Truth Text (Upper Bound)

Each planner receives the ground-truth text (no transcription errors). Precision/Recall/F1 here is the best the planner can achieve — an upper bound.

### Section 3 — End-to-End: Precision/Recall/F1 on Transcribed Audio (Real World)

Each planner receives the transcription from Section 1. The gap between Section 2 and Section 3 on identical chunks is the **pure perception penalty** caused by transcription errors.

### Results

*(Numbers below are placeholder estimates. Run `test_evaluation.ipynb` or `python eval.py` to populate with real results, then update this table.)*

**Section 1 — Perception (WER, lower is better)**

| Model | Avg WER | Chunks |
|---|---|---|
| Whisper (small) | — | 200 |
| PocketSphinx | — | 200 |

**Section 2 — Planning on Ground-Truth Text (upper bound)**

| Planning Mode | Prec | Rec | F1 | N |
|---|---|---|---|---|
| LLM (Qwen 2.5) | — | — | — | 50* |
| LLM (Gemma 2) | — | — | — | 50* |
| Transformer (Embeddings) | — | — | — | 200 |
| Non-DL (Keywords) | — | — | — | 200 |

**Section 3 — End-to-End (real-world, same chunks as Section 2)**

| Perception | Planning | Prec | Rec | F1 | N |
|---|---|---|---|---|---|
| Whisper | LLM (Qwen 2.5) | — | — | — | 50* |
| Whisper | LLM (Gemma 2) | — | — | — | 50* |
| Whisper | Transformer (Embeddings) | — | — | — | 200 |
| PocketSphinx | Non-DL (Keywords) | — | — | — | 200 |

*\* LLM rows use 50 stratified chunks with ~50% positive rate — identical between Section 2 and Section 3.*

---

## Lessons Learned

**1. Per-model prompt engineering is not optional for LLMs.**
Gemma 2 2B over-fires without explicit strictness instructions in the prompt. A shared template across models degrades precision noticeably. Each model branch in `_build_ollama_prompt()` is individually tuned and should not be unified without re-testing.

**2. Rolling context improves Whisper in the live agent but not in eval.**
In `app.py`, the Scribe worker passes the previous chunk's transcript as `initial_prompt` to Whisper, which reduces boundary errors and improves coherence. The eval harness runs chunks independently (no rolling context) to keep measurements reproducible and avoid context contamination across files. Actual performance with rolling context is therefore slightly better than the eval numbers suggest.

**3. Keyword matching needs a low hit threshold for PocketSphinx output.**
The first implementation required 2 keyword hits per chunk. This caused many false negatives with PocketSphinx because one word of a two-word financial term was commonly dropped or mis-transcribed. Switching to single-hit (any content word OR bigram match) recovered significant recall without harming precision on the Whisper pipeline.

**4. Evaluation dataset design matters more than metric selection.**
An early version compared Section 2 and Section 3 on datasets with different class distributions (12% vs 70% positive), making E2E appear better than the upper-bound planner — an impossible result. The fix was a single shared stratified sample (~50% positive) used identically across all three sections.

---

## Data Sources and Credits

### Test Audio and Transcripts

Audio files and original transcripts in `testset_audio/` and `testset_transcript/` are sourced from the **earnings22** dataset:

> Rev.com, Inc. *speech-datasets — earnings22 subset (English, North American dialect).*
> https://github.com/revdotcom/speech-datasets

Used here for non-commercial, educational research only. Full credit to Rev.com, Inc.

### Finance Keywords

The keyword list in `finance_keywords_accountingcoach.csv` is sourced from the **AccountingCoach** online glossary:

> AccountingCoach, LLC. *Accounting Terms & Definitions.*
> https://www.accountingcoach.com/terms

Used here for non-commercial, educational research only. Full credit to AccountingCoach, LLC.

---

## Files

```
app.py                               Streamlit UI + three worker threads
agent_core.py                        Pure business logic: transcription + topic detection
eval.py                              Offline evaluation harness (Section 1/2/3)
transcript_processing.py             .aligned.nlp -> _parsed.json -> _labeled.json pipeline
test_evaluation.ipynb                Interactive evaluation notebook
requirements.txt                     Python dependencies
finance_keywords_accountingcoach.csv Finance keyword list (used by Non-DL Keywords mode)
testset_audio/                       5 earnings-call MP3 files (earnings22 subset)
testset_transcript/                  Labeled 15-s chunk JSON files
eval_results/                        Timestamped evaluation result text files
eval_sample_cache.json               Saved sampling decisions (delete to re-sample)
```

---

## Ethics statement
This project is intended for research and educational purposes in large language model and intelligent agent development. All data collection and deployment are conducted with respect for privacy and copyright. Care has been taken to avoid misuse of the model and to ensure responsible use of the technology, particularly in relation to surveilance, personal data, and public safety.

---

## How to Run

**Requirements**:
```
pocketsphinx==5.0.4
faster-whisper==1.2.1
sounddevice==0.5.5
webrtcvad==2.0.10
SpeechRecognition==3.16.0
torch==2.6.0+cu124
sentence-transformers==5.0.0
numpy==1.26.4
ollama==0.6.1
streamlit==1.45.1
pydub==0.25.1
requests==2.32.4
scikit-learn==1.3.2
```

**Install dependencies:**
```
pip install -r requirements.txt
```

**Start Ollama (required for LLM planning modes):**
```
ollama serve
ollama pull qwen2.5
ollama pull gemma2:2b
```

**Run the live agent:**
```
streamlit run app.py
```

**Run offline evaluation (~2 hours, GPU recommended):**
```
python eval.py 
# on Windows:
py eval.py
```

Results are saved to `eval_results/eval_YYYYMMDD_HHMMSS.txt`. On subsequent runs the same sample is reused from `eval_sample_cache.json`. Delete that file to draw a new sample.

Alternatively, open `test_evaluation.ipynb` and run all cells in order for an interactive version with per-section output.

**Process raw transcripts (one-time setup):**
```
python transcript_processing.py
# on Windows:
py transcript_processing.py
```
Or run the corresponding cells in `test_evaluation.ipynb`.
