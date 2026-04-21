"""
eval.py — Offline evaluation harness for the Meeting Alert Agent.

Three evaluation sections — all use the SAME set of chunks:
  1. Perception   — WER for Whisper (small) and PocketSphinx on test audio
  2. Planning     — P/R/F1 for each planning model on ground-truth text
                    (upper bound — no transcription errors)
  3. End-to-end   — P/R/F1 for each combo on transcribed audio
                    (real world — perception errors included)

Sec 2 vs Sec 3 on identical chunks isolates the cost of imperfect transcription:
  gap = Sec3_metric - Sec2_metric  ← pure perception penalty

Test data layout
----------------
  testset_audio/       *.mp3
  testset_transcript/  *_labeled.json  (start_time, end_time, text, topic_present)

Test data source
----------------
  Audio and original transcripts are a subset of the earnings22 dataset from:
    https://github.com/revdotcom/speech-datasets
  (Subset: English language, North American dialect.)
  Full credit to Rev.com, Inc. Used here for non-commercial, educational research.

Usage
-----
  cd meeting-alert-agent
  python eval.py          # (use the same Python / venv as app.py)

Runtime estimate (default settings)
-------------------------------------
  Whisper transcription  ~5-10 min  (GPU, ~200 audio chunks)
  PocketSphinx           ~2-4 min   (CPU)
  Embeddings planning    ~1-2 min
  Ollama LLM planning    ~10-20 min per model
"""

import os
import glob
import json
import sys
import time
import shutil

import numpy as np
# pydub and sklearn are imported lazily inside the functions that use them.
# Importing them at module level loads scipy/OpenBLAS before faster-whisper,
# which corrupts ctranslate2's native memory init on Windows (0xC0000005).

from agent_core import (
    LETTERS, OLLAMA_MODELS, FINANCE_KEYWORDS,
    transcribe_chunk, detect_topics,
)

# Configuration — edit these to control scope and runtime

AUDIO_DIR      = "testset_audio"
TRANSCRIPT_DIR = "testset_transcript"

# The 10 topics used when labeling the test set
TARGET_TOPICS = [
    "dividend",
    "gross margin",
    "inventory",
    "capital allocation",
    "net income",
    "operating income",
    "pricing",
    "capex",
    "investment portfolio",
    "sales growth",
]

# Convert to the {"id", "text"} format expected by detect_topics()
_EVAL_TOPICS = [{"id": LETTERS[i], "text": t} for i, t in enumerate(TARGET_TOPICS)]

# Shared evaluation sample (all three sections).
# Audio transcription is slow, so we use a stratified sample per file.
# The SAME sample is used for Sec 1 (audio WER), Sec 2 (GT text planning),
# and Sec 3 (E2E) so all results are directly comparable.
# Per-file: keep all positives when under budget, then fill with negatives.
# Set to None to process every chunk (much slower).
MAX_AUDIO_PER_FILE = 40

# LLM query budget (Sections 2 & 3).
# Each LLM call takes ~1-2 s per topic. Limit total chunks sent to LLMs.
# Applied via stratified sub-sampling of the shared evaluation set (same
# positive rate) so LLM rows in Sec 2 and Sec 3 use identical chunks.
MAX_CHUNKS_LLM = 50

# Whisper device.
# CUDA is required on this Windows+RTX4060 machine.  The CPU build of
# ctranslate2 exits at the C level (not a Python exception) due to a missing
# CPU feature or internal init failure — no Python try/except can catch that.
# The CUDA path works when:
#   (a) torch.cuda.init() pre-warms the CUDA DLLs in the main thread, and
#   (b) WhisperModel() is called inside a daemon thread (not the main thread).
# Both conditions are satisfied below in _load_whisper_with_progress().
WHISPER_DEVICE       = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

SAMPLE_CACHE_PATH = "eval_sample_cache.json"   # set to None to always re-sample
RESULTS_DIR       = "eval_results"              # directory for saved result files

# Planning modes to evaluate in Section 2
PLANNING_MODES = [
    "LLM (Ollama - Qwen 2.5)",
    "LLM (Ollama - Gemma 2)",
    "Transformer (Embeddings)",
    "Non-DL (Keywords)",
]

# End-to-end combinations for Section 3
E2E_COMBOS = [
    ("Deep Learning (Whisper)", "LLM (Ollama - Qwen 2.5)"),
    ("Deep Learning (Whisper)", "LLM (Ollama - Gemma 2)"),
    ("Deep Learning (Whisper)", "Transformer (Embeddings)"),
    ("Non-DL (PocketSphinx)",   "Non-DL (Keywords)"),
]


# Dataset loading

def _load_labeled_files():
    """Return list of (base_name, all_chunks) for every *_labeled.json."""
    files = sorted(glob.glob(os.path.join(TRANSCRIPT_DIR, "*_labeled.json")))
    if not files:
        sys.exit(f"[eval] No *_labeled.json files found in '{TRANSCRIPT_DIR}'.")
    result = []
    for fpath in files:
        base = os.path.basename(fpath).replace("_labeled.json", "")
        with open(fpath, encoding="utf-8") as f:
            result.append((base, json.load(f)))
    return result


def _stratified_sample(chunks, max_per_file):
    """Return an evenly-spaced stratified sample of chunks up to max_per_file.

    Two cases:

    * positives < budget  — keep ALL positives (guarantees 100 % recall coverage
      in the eval sample), then fill the remaining slots with evenly-spaced
      negatives.  This is the normal case for per-file sampling where the
      financial discussion is clustered and we want every positive chunk.

    * positives >= budget — subsample both classes proportionally so the
      positive rate of the input is preserved and the total stays at max_per_file.
      This is the case when building the LLM budget subset from the shared set
      (e.g. 100 positives + 100 negatives → 50-chunk subset at ~50 % positive).
    """
    if max_per_file is None:
        return chunks

    positives = [c for c in chunks if c["topic_present"] == 1]
    negatives = [c for c in chunks if c["topic_present"] == 0]

    def _pick(lst, n):
        """Evenly-spaced selection of n items from lst."""
        if n <= 0 or not lst:
            return []
        if n >= len(lst):
            return lst
        return [lst[int(i * len(lst) / n)] for i in range(n)]

    if len(positives) < max_per_file:
        # Keep all positives; fill with evenly-spaced negatives.
        sampled_pos = positives
        sampled_neg = _pick(negatives, max_per_file - len(positives))
    else:
        # Positives alone exceed budget — maintain the observed positive rate.
        pos_rate    = len(positives) / max(len(chunks), 1)
        n_pos       = max(1, round(max_per_file * pos_rate))
        n_neg       = max_per_file - n_pos
        sampled_pos = _pick(positives, n_pos)
        sampled_neg = _pick(negatives, n_neg)

    return sorted(sampled_pos + sampled_neg, key=lambda c: c["start_time"])


def _save_sample_cache(dataset_sample, llm_subset, path):
    """Save the sampling decisions to JSON so subsequent runs use identical chunks."""
    cache = {
        "dataset_sample": {
            base: [c["start_time"] for c in chunks]
            for base, chunks in dataset_sample
        },
        "llm_subset": [
            {"_base": c["_base"], "start_time": c["start_time"]}
            for c in llm_subset
        ],
        "config": {
            "MAX_AUDIO_PER_FILE": MAX_AUDIO_PER_FILE,
            "MAX_CHUNKS_LLM":     MAX_CHUNKS_LLM,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _load_sample_cache(files_all, path):
    """Reconstruct dataset_sample and llm_subset from a previously saved cache file.

    Returns (dataset_sample, llm_subset) with _base tags already applied, or
    raises FileNotFoundError if path does not exist.
    """
    with open(path, encoding="utf-8") as f:
        cache = json.load(f)

    files_dict = {base: chunks for base, chunks in files_all}
    dataset_sample = []
    for base, start_times in cache["dataset_sample"].items():
        all_chunks = files_dict.get(base, [])
        by_time    = {c["start_time"]: c for c in all_chunks}
        sampled    = [by_time[t] for t in start_times if t in by_time]
        for c in sampled:
            c["_base"] = base
        dataset_sample.append((base, sampled))

    flat = {(c["_base"], c["start_time"]): c
            for _, cs in dataset_sample for c in cs}
    llm_subset = [
        flat[(e["_base"], e["start_time"])]
        for e in cache["llm_subset"]
        if (e["_base"], e["start_time"]) in flat
    ]
    return dataset_sample, llm_subset


def _sample_cache_is_valid(dataset_sample, llm_subset):
    """Return True if the cached sample is usable.

    Checks two conditions:
      1. LLM subset positive rate is between 15 % and 85 %.  A rate outside
         this range means the cache was built before the proportional-sampling
         fix (when positives > budget, all positives were returned and no
         negatives were included).
      2. LLM subset is non-empty.
    """
    if not llm_subset:
        return False
    llm_pos_rate = sum(c["topic_present"] for c in llm_subset) / len(llm_subset)
    return 0.15 <= llm_pos_rate <= 0.85


def _build_sample(files_all):
    """Build a fresh stratified dataset_sample and llm_subset from labeled files."""
    dataset_sample = []
    for base, chunks in files_all:
        sampled = _stratified_sample(chunks, MAX_AUDIO_PER_FILE)
        for c in sampled:
            c["_base"] = base
        dataset_sample.append((base, sampled))
    flat_sample = [c for _, cs in dataset_sample for c in cs]
    llm_subset  = _stratified_sample(flat_sample, MAX_CHUNKS_LLM)
    return dataset_sample, llm_subset


# Utilities

_output_log: list[str] = []


def _p(msg, **kw):
    """Print with flush=True for real-time output; also appends to _output_log for saving."""
    print(msg, flush=True, **kw)
    if kw.get("end", "\n") != "\r":
        _output_log.append(str(msg))


def _save_results(path_dir=RESULTS_DIR):
    """Write the captured _output_log to a timestamped .txt file in path_dir."""
    os.makedirs(path_dir, exist_ok=True)
    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(path_dir, f"eval_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(_output_log))
    return path


def _divider(title, width=68):
    """Print a centered section divider, clearing any dangling \\r progress line."""
    print(f"\r{' ' * 90}", flush=True)
    _p("=" * width)
    pad = max(0, (width - len(title) - 2) // 2)
    _p(" " * pad + " " + title + " " + " " * pad)
    _p("=" * width)


def _pydub_to_int16(segment):
    """Convert pydub AudioSegment to a 16 kHz mono int16 numpy array."""
    from pydub import AudioSegment as _AS  # lazy — must not load before faster-whisper
    seg = segment.set_frame_rate(16000).set_channels(1)
    return np.array(seg.get_array_of_samples(), dtype=np.int16)


def _wer(reference, hypothesis):
    """Compute Word Error Rate via O(n)-space dynamic-programming edit distance.

    WER = edit_distance(ref_words, hyp_words) / len(ref_words).
    Both strings are lowercased and split on whitespace before comparison.
    Returns 0.0 when reference is empty to avoid division by zero.
    """
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref:
        return 0.0
    prev = list(range(len(hyp) + 1))
    for r_word in ref:
        curr = [prev[0] + 1] + [0] * len(hyp)
        for j, h_word in enumerate(hyp, 1):
            curr[j] = (prev[j - 1] if r_word == h_word
                       else 1 + min(prev[j], curr[j - 1], prev[j - 1]))
        prev = curr
    return prev[len(hyp)] / len(ref)


def _prf(y_true, y_pred):
    """Return (precision, recall, f1) for binary classification."""
    from sklearn.metrics import precision_recall_fscore_support  # lazy — must not load before faster-whisper
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return float(p), float(r), float(f)


def _counts(y_true, y_pred):
    """Return (TP, FP, FN) counts for binary predictions."""
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    return tp, fp, fn


def _progress(label, done, total, t0):
    """Print an in-place \\r progress ticker showing elapsed/ETA."""
    elapsed = time.time() - t0
    eta  = elapsed / done * (total - done) if done > 0 else 0
    line = f"    {label}  {done}/{total}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)"
    # Pad to 88 chars so \r fully overwrites any previous longer line
    print(f"\r{line:<88}", end="", flush=True)


# Whisper model loader — with cache check and progress ticker

def _check_whisper_cached(model_size="small"):
    """Return True if the faster-whisper model snapshot exists in HF hub cache.

    faster-whisper downloads from HuggingFace Hub into:
      ~/.cache/huggingface/hub/models--Systran--faster-whisper-<size>/snapshots/
    This path is user-level (not venv-level), so it is shared across all Python
    environments for the same OS user.
    """
    snapshots_dir = os.path.join(
        os.path.expanduser("~"),
        ".cache", "huggingface", "hub",
        f"models--Systran--faster-whisper-{model_size}",
        "snapshots",
    )
    if not os.path.isdir(snapshots_dir):
        return False
    try:
        return any(True for _ in os.scandir(snapshots_dir))
    except Exception:
        return False


def _load_whisper_with_progress(device, compute_type, model_size="small"):
    """Load WhisperModel in a background thread, printing progress every 5 s.

    Why a thread?
    -------------
    On Windows, ctranslate2 / faster-whisper model init (even on CPU) can block
    the main thread for 30-90 s while it decompresses the model.  If the model
    is NOT cached it will also download ~460 MB silently — making it look hung.
    Running in a thread lets us print heartbeat messages so the user can tell
    something is happening.

    Returns the loaded WhisperModel, or calls sys.exit() on timeout/error.
    """
    import threading

    # Pre-warm CUDA DLLs in the main thread BEFORE ctranslate2 starts.
    # On Windows, ctranslate2's CUDA init deadlocks if the DLL loader lock is
    # held by the main thread.  Calling torch.cuda.init() first loads the CUDA
    # runtime DLLs so ctranslate2 finds them already mapped when it runs in the
    # daemon thread.  This mirrors what app.py achieves via the torch.classes
    # patch at its top level.
    if device == "cuda":
        try:
            import torch as _torch
            if not _torch.cuda.is_initialized():
                _torch.cuda.init()
                _p(f"    (torch CUDA pre-init: {_torch.cuda.get_device_name(0)})")
        except Exception as _cuda_e:
            _p(f"    ⚠  torch CUDA pre-init failed: {_cuda_e}")
            _p(f"       Falling back to CPU / int8")
            device       = "cpu"
            compute_type = "int8"

    is_cached = _check_whisper_cached(model_size)
    if not is_cached:
        _p(f"\n  ⚠  Whisper '{model_size}' model NOT found in HuggingFace cache.")
        _p(f"     It will be downloaded now (~460 MB).  This can take several minutes.")
        _p(f"     If it appears stuck, make sure you are running from the correct venv:")
        _p(f"       python eval.py    ← correct (uses activated venv)")
        _p(f"       py eval.py        ← may use a different Python / re-download\n")
    else:
        _p(f"    (model cached — loading from disk…)")

    result = [None]
    error  = [None]

    def _worker():
        try:
            from faster_whisper import WhisperModel as _WM
            result[0] = _WM(model_size, device=device, compute_type=compute_type)
        except Exception as exc:
            error[0] = exc

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    heartbeat = 5    # seconds between progress dots
    timeout   = 600  # 10 minutes hard limit
    elapsed   = 0
    while t.is_alive() and elapsed < timeout:
        t.join(timeout=heartbeat)
        elapsed += heartbeat
        if t.is_alive():
            mins = elapsed // 60
            secs = elapsed % 60
            _p(f"    … still loading  ({mins:02d}:{secs:02d} elapsed)")

    if t.is_alive():
        sys.exit(
            f"\n[eval] WhisperModel load timed out after {timeout}s.\n"
            f"       Try running:  python eval.py   (not  py eval.py)\n"
            f"       Ensure faster_whisper is installed in the active venv."
        )
    if error[0] is not None:
        sys.exit(
            f"\n[eval] Cannot load Whisper ({device}/{compute_type}): {error[0]}\n"
            f"       Try running:  python eval.py   (not  py eval.py)"
        )
    return result[0]


# Section 1 — Perception: WER

def run_perception_eval(dataset_sample, whisper_model, recognizer):
    """Transcribe the shared evaluation sample with Whisper and PocketSphinx.

    Parameters
    ----------
    dataset_sample : list of (base_name, [chunk, ...])
        Pre-built stratified sample — the same list used in Sections 2 & 3.
    whisper_model  : faster_whisper.WhisperModel instance
    recognizer     : speech_recognition.Recognizer instance

    Returns
    -------
    dict  base_name -> list of (chunk_dict, whisper_text, sphinx_text)
          Reused in Section 3 to avoid re-transcribing.
    """
    _divider("SECTION 1  ·  PERCEPTION — Word Error Rate (WER)")

    total = sum(len(cs) for _, cs in dataset_sample)
    pos   = sum(c["topic_present"] for _, cs in dataset_sample for c in cs)
    _p(f"  Shared evaluation set: {total} chunks across {len(dataset_sample)} files "
       f"({pos} positive, {100*pos/max(total,1):.1f}%)")
    _p(f"  (Stratified per file: all positives kept when under budget, "
       f"proportional otherwise; max {MAX_AUDIO_PER_FILE}/file)\n")

    wer_w_all, wer_s_all = [], []
    transcripts = {}

    for base, chunks in dataset_sample:
        audio_path = os.path.join(AUDIO_DIR, f"{base}.mp3")
        if not os.path.exists(audio_path):
            _p(f"  ⚠  Missing audio: {audio_path}  (skipped)")
            continue

        from pydub import AudioSegment
        full_audio = AudioSegment.from_mp3(audio_path)
        file_rows  = []
        t0 = time.time()

        for i, chunk in enumerate(chunks):
            s_ms = int(chunk["start_time"] * 1000)
            e_ms = int(chunk["end_time"]   * 1000)
            a16  = _pydub_to_int16(full_audio[s_ms:e_ms])

            w_text = transcribe_chunk(a16, "Deep Learning (Whisper)",
                                      whisper_model=whisper_model)
            s_text = transcribe_chunk(a16, "Non-DL (PocketSphinx)",
                                      recognizer=recognizer)

            wer_w_all.append(_wer(chunk["text"], w_text))
            wer_s_all.append(_wer(chunk["text"], s_text))
            file_rows.append((chunk, w_text, s_text))

            _progress(f"{base}", i + 1, len(chunks), t0)

        _p(f"  ✓ {base}  ({len(chunks)} chunks, "
           f"{sum(c['topic_present'] for c in chunks)} positive)          ")
        transcripts[base] = file_rows

    _p("")
    _p(f"  {'Model':<22}  {'Avg WER':>8}  {'Chunks':>7}")
    _p(f"  {'─'*22}  {'─'*8}  {'─'*7}")
    _p(f"  {'Whisper (small)':<22}  {np.mean(wer_w_all):>8.4f}  {len(wer_w_all):>7}")
    _p(f"  {'PocketSphinx':<22}  {np.mean(wer_s_all):>8.4f}  {len(wer_s_all):>7}")
    _p("")
    _p("  Lower WER = better. WER 1.0 means as many errors as reference words.")

    return transcripts


# Section 2 — Planning: P/R/F1 on ground-truth text

def run_planning_eval(dataset_sample, llm_subset, semantic_model, eval_topics=None):
    """Feed ground-truth text to each planning model and report P/R/F1.

    Uses the SAME stratified sample as Sections 1 & 3.
    Sec 2 vs Sec 3 on identical chunks isolates the cost of imperfect
    transcription: gap = Sec3 - Sec2 = pure perception penalty.

    Parameters
    ----------
    dataset_sample : list of (base_name, [chunk, ...])
        Shared evaluation sample (same as Sec 1 / Sec 3).
    llm_subset     : list of chunk dicts (stratified, tagged with _base)
        The same LLM-budget subset used in Section 3.
    semantic_model : SentenceTransformer instance for Embeddings mode
    eval_topics    : list of {'id': str, 'text': str} dicts (default: _EVAL_TOPICS)
        Override to use a different topic list than the module-level config.
    """
    if eval_topics is None:
        eval_topics = _EVAL_TOPICS
    _divider("SECTION 2  ·  PLANNING — on Ground Truth Text")
    _p("  (Same chunks as Sec 1/3 — GT text gives the upper bound for planning)\n")

    flat_sample = [c for _, cs in dataset_sample for c in cs]
    y_true_all  = [c["topic_present"] for c in flat_sample]
    texts_all   = [c["text"]          for c in flat_sample]

    llm_texts = [c["text"]          for c in llm_subset]
    llm_ytrue = [c["topic_present"] for c in llm_subset]

    total = len(flat_sample);  pos   = sum(y_true_all)
    lltot = len(llm_subset);   llpos = sum(llm_ytrue)
    _p(f"  Shared set : {total} chunks | {pos} positive ({100*pos/max(total,1):.1f}%)")
    _p(f"  LLM subset : {lltot} chunks | {llpos} positive ({100*llpos/max(lltot,1):.1f}%)"
       f"  [stratified — same subset used in Sec 3]\n")

    _p(f"  {'Planning Mode':<30}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}"
       f"  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'N':>6}")
    _p(f"  {'─'*30}  {'─'*6}  {'─'*6}  {'─'*6}"
       f"  {'─'*4}  {'─'*4}  {'─'*4}  {'─'*6}")

    embedding_cache = {}
    results = {}

    for mode in PLANNING_MODES:
        is_llm = mode in OLLAMA_MODELS
        subset_texts = llm_texts  if is_llm else texts_all
        subset_ytrue = llm_ytrue  if is_llm else y_true_all

        y_pred = []
        t0 = time.time()

        for i, text in enumerate(subset_texts):
            matched = detect_topics(
                text, mode, eval_topics,
                finance_keywords=FINANCE_KEYWORDS,
                semantic_model=semantic_model,
                embedding_cache=embedding_cache,
            )
            y_pred.append(1 if matched else 0)

            if (i + 1) % 10 == 0 or (i + 1) == len(subset_texts):
                _progress(mode[:30], i + 1, len(subset_texts), t0)

        p, r, f = _prf(subset_ytrue, y_pred)
        tp, fp, fn = _counts(subset_ytrue, y_pred)
        n_str = str(len(subset_texts)) + ("*" if is_llm else " ")

        short = mode.replace("LLM (Ollama - ", "LLM (")
        _p(f"\r  {short:<30}  {p:>6.3f}  {r:>6.3f}  {f:>6.3f}"
           f"  {tp:>4}  {fp:>4}  {fn:>4}  {n_str:>6}    ")
        results[mode] = (y_pred, subset_ytrue)

    _p("")
    _p(f"  * LLM subset: {lltot} stratified chunks ({llpos} positive, "
       f"{100*llpos/max(lltot,1):.1f}%) — identical to Sec 3 LLM rows.")

    return results


# Section 3 — End-to-end: P/R/F1 using transcribed audio

def run_e2e_eval(transcripts, llm_subset, semantic_model, eval_topics=None):
    """Evaluate the four production (perception + planning) combinations.

    Uses transcriptions cached from Section 1 — no re-transcription.
    LLM rows use the same llm_subset as Section 2 for a direct comparison.

    Note: Whisper here runs WITHOUT rolling context (independent chunks).
          The live agent's rolling context typically improves Whisper accuracy,
          so real-world E2E performance may be slightly higher.

    Parameters
    ----------
    transcripts  : dict returned by run_perception_eval
    llm_subset   : list of chunk dicts (same as used in Section 2)
    semantic_model : SentenceTransformer instance
    eval_topics  : list of {'id': str, 'text': str} dicts (default: _EVAL_TOPICS)
    """
    if eval_topics is None:
        eval_topics = _EVAL_TOPICS
    _divider("SECTION 3  ·  END-TO-END EVALUATION")
    _p("  (Same chunks as Sec 1/2 — transcribed text reveals perception penalty)\n")

    w_texts, s_texts, y_true_all = [], [], []
    for base in sorted(transcripts.keys()):
        for chunk, w_text, s_text in transcripts[base]:
            w_texts.append(w_text)
            s_texts.append(s_text)
            y_true_all.append(chunk["topic_present"])

    total = len(y_true_all);  pos = sum(y_true_all)
    _p(f"  Shared set : {total} transcribed chunks ({pos} positive, "
       f"{100*pos/max(total,1):.1f}%)")

    # Look up transcribed texts for each LLM-subset chunk by (base, start_time)
    _lookup: dict = {}
    for base, rows in transcripts.items():
        for chunk, w_text, s_text in rows:
            _lookup[(base, chunk["start_time"])] = (w_text, s_text)

    llm_w, llm_s, llm_ytrue = [], [], []
    for c in llm_subset:
        key = (c["_base"], c["start_time"])
        if key in _lookup:
            w, s = _lookup[key]
            llm_w.append(w);  llm_s.append(s);  llm_ytrue.append(c["topic_present"])

    lltot = len(llm_ytrue);  llpos = sum(llm_ytrue)
    _p(f"  LLM subset : {lltot} chunks ({llpos} positive, "
       f"{100*llpos/max(lltot,1):.1f}%)  [same subset as Sec 2 LLM rows]\n")

    _p(f"  {'Perception':<16}  {'Planning':<30}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}"
       f"  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'N':>6}")
    _p(f"  {'─'*16}  {'─'*30}  {'─'*6}  {'─'*6}  {'─'*6}"
       f"  {'─'*4}  {'─'*4}  {'─'*4}  {'─'*6}")

    embedding_cache = {}

    for perc_mode, plan_mode in E2E_COMBOS:
        is_llm = plan_mode in OLLAMA_MODELS

        if is_llm:
            subset_texts = llm_w if "Whisper" in perc_mode else llm_s
            subset_ytrue = llm_ytrue
        else:
            subset_texts = w_texts if "Whisper" in perc_mode else s_texts
            subset_ytrue = y_true_all

        y_pred = []
        t0 = time.time()

        for i, text in enumerate(subset_texts):
            matched = detect_topics(
                text, plan_mode, eval_topics,
                finance_keywords=FINANCE_KEYWORDS,
                semantic_model=semantic_model,
                embedding_cache=embedding_cache,
            )
            y_pred.append(1 if matched else 0)

            if (i + 1) % 10 == 0 or (i + 1) == len(subset_texts):
                label = f"{perc_mode[:8]}+{plan_mode[:20]}"
                _progress(label, i + 1, len(subset_texts), t0)

        p, r, f = _prf(subset_ytrue, y_pred)
        tp, fp, fn = _counts(subset_ytrue, y_pred)
        n_str = str(len(subset_texts)) + ("*" if is_llm else " ")

        p_short = "Whisper"  if "Whisper" in perc_mode else "PocketSphinx"
        b_short = plan_mode.replace("LLM (Ollama - ", "LLM (")
        _p(f"\r  {p_short:<16}  {b_short:<30}  {p:>6.3f}  {r:>6.3f}  {f:>6.3f}"
           f"  {tp:>4}  {fp:>4}  {fn:>4}  {n_str:>6}    ")

    _p("")
    _p(f"  * LLM rows: {lltot} chunks (stratified, same as Sec 2 LLM rows).")
    _p("  Note: Whisper here has no rolling context (independent chunks).")
    _p("        Live-agent performance with rolling context is typically higher.")


# Main

def main():
    """Run the full three-section evaluation pipeline."""
    _divider("Meeting Alert Agent  ·  Offline Evaluation Harness")

    _p("\n  Loading dataset…")
    files_all    = _load_labeled_files()
    total_chunks = sum(len(c) for _, c in files_all)
    total_pos    = sum(c["topic_present"] for _, cs in files_all for c in cs)
    _p(f"  Full corpus: {len(files_all)} files | {total_chunks} chunks "
       f"| {total_pos} positive ({100*total_pos/max(total_chunks,1):.1f}%)")

    dataset_sample, llm_subset = None, None

    if SAMPLE_CACHE_PATH and os.path.exists(SAMPLE_CACHE_PATH):
        _p(f"  Loading sample from cache: {SAMPLE_CACHE_PATH}")
        dataset_sample, llm_subset = _load_sample_cache(files_all, SAMPLE_CACHE_PATH)
        if not _sample_cache_is_valid(dataset_sample, llm_subset):
            llm_pos_rate = sum(c["topic_present"] for c in llm_subset) / max(len(llm_subset), 1)
            _p(f"  Cache invalid (LLM subset positive rate = {llm_pos_rate:.1%}) — regenerating.")
            os.remove(SAMPLE_CACHE_PATH)
            dataset_sample, llm_subset = None, None

    if dataset_sample is None:
        dataset_sample, llm_subset = _build_sample(files_all)
        if SAMPLE_CACHE_PATH:
            _save_sample_cache(dataset_sample, llm_subset, SAMPLE_CACHE_PATH)
            _p(f"  Sample saved: {SAMPLE_CACHE_PATH}  (delete to re-sample)")

    flat_sample  = [c for _, cs in dataset_sample for c in cs]
    sample_total = len(flat_sample)
    sample_pos   = sum(c["topic_present"] for c in flat_sample)
    llm_total    = len(llm_subset)
    llm_pos      = sum(c["topic_present"] for c in llm_subset)

    _p(f"\n  Shared evaluation set (all sections):")
    _p(f"    {sample_total} chunks | {sample_pos} positive "
       f"({100*sample_pos/max(sample_total,1):.1f}%) "
       f"| ≤{MAX_AUDIO_PER_FILE} chunks/file, all positives preserved")
    _p(f"    LLM subset: {llm_total} chunks | {llm_pos} positive "
       f"({100*llm_pos/max(llm_total,1):.1f}%) | stratified from shared set")
    _p(f"\n  Sec 2 vs Sec 3 use identical chunks — gap = perception penalty.")

    _p("\n  Loading models…")
    _p(f"  (Whisper device: {WHISPER_DEVICE} / {WHISPER_COMPUTE_TYPE}"
       "  — change WHISPER_DEVICE/WHISPER_COMPUTE_TYPE at the top of this file)")

    _p(f"    → Whisper (small, {WHISPER_DEVICE} / {WHISPER_COMPUTE_TYPE})")
    whisper_model = _load_whisper_with_progress(WHISPER_DEVICE, WHISPER_COMPUTE_TYPE)
    _p("    → Whisper: ✓")

    _p("    → SentenceTransformer (all-MiniLM-L6-v2)…", end=" ")
    from sentence_transformers import SentenceTransformer
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    _p("✓")

    _p("    → PocketSphinx recognizer…", end=" ")
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    _p("✓")

    _p("  All models loaded.\n")

    transcripts = run_perception_eval(dataset_sample, whisper_model, recognizer)
    run_planning_eval(dataset_sample, llm_subset, semantic_model)
    run_e2e_eval(transcripts, llm_subset, semantic_model)

    result_path = _save_results()
    _p(f"  Results saved → {result_path}")

    _divider("Evaluation complete")
    _p("")


if __name__ == "__main__":
    main()
