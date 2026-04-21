"""
agent_core.py — Pure, Streamlit-free business logic.

Shared between:
  app.py   — live Streamlit agent (threading workers call these functions)
  eval.py  — offline evaluation harness (calls them directly on test data)

No Streamlit imports. No queue/threading. All functions are stateless
(rolling context is managed by the caller when needed).
"""

import csv
import os
import re
import numpy as np
import requests

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BADGE_COLORS = [
    "#e74c3c", "#2980b9", "#27ae60", "#8e44ad",
    "#e67e22", "#16a085", "#d35400", "#1abc9c",
]

# Ollama model registry — display name → Ollama model tag
OLLAMA_MODELS = {
    "LLM (Ollama - Gemma 2)":  "gemma2:2b",
    "LLM (Ollama - Qwen 2.5)": "qwen2.5",
}

_STOP_WORDS = frozenset({
    "the","a","an","is","was","are","were","did","why","how","what","when",
    "where","who","i","my","me","we","our","you","it","its","this","that",
    "for","to","of","in","on","at","by","with","about","and","or","but",
    "not","no","do","does","be","been","have","has","had","will","would",
    "could","should","may","might","can","than","more","much","very","so",
    "if","then","now","there","up","all","any","some","other","most","many",
    "few","each","both","while","after","before","since","per","as","yet",
    "also","just","still","even","only","too","such","same","last","year",
    "years","quarter","quarters","said","says","well","right","like","made",
    "get","got","let","into","from","these","those","here","down","over",
    "between","during","significantly","slightly","overall","total","current",
})

# Compiled once at import time for performance
_RE_INLINE_ARTIFACT = re.compile(r'[_.]{3,}')
_RE_MULTI_SPACE     = re.compile(r' {2,}')
_RE_CJK             = re.compile(r'[\u2e80-\u9fff\uf900-\ufaff]')


def load_finance_keywords(filepath=None):
    """Load finance keywords from CSV and return a list of lowercase strings.

    The default keyword list is sourced from AccountingCoach's online glossary:
      https://www.accountingcoach.com/terms
    Full credit to AccountingCoach, LLC. The list is used here solely for
    non-commercial, educational topic-detection research.

    The default path is resolved relative to this file's directory so the
    function works regardless of the caller's working directory.

    Parameters
    ----------
    filepath : path to a CSV file with a 'word' column (optional)

    Returns
    -------
    list of str
    """
    if filepath is None:
        filepath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "finance_keywords_accountingcoach.csv",
        )
    keywords = set()
    try:
        with open(filepath, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers  = [h.lower() for h in reader.fieldnames]
            word_col = "word" if "word" in headers else reader.fieldnames[0]
            for row in reader:
                word = row[word_col].strip().lower()
                if word:
                    keywords.add(word)
    except FileNotFoundError:
        print("[Warning] finance_keywords CSV not found — using fallback list.")
        return [
            "profit", "drop", "margin", "revenue", "loss", "ebitda",
            "decrease", "dividend", "inventory", "allocation", "income",
            "pricing", "capex", "investment", "portfolio", "sales", "growth",
        ]
    return list(keywords)


# Computed once at import, shared across app.py & eval.py
FINANCE_KEYWORDS = load_finance_keywords()


def badge_color(letter):
    """Return the hex badge colour for a given topic letter.

    Parameters
    ----------
    letter : single uppercase letter (e.g. 'A')

    Returns
    -------
    str  — CSS hex colour string, or '#888888' for unknown letters
    """
    idx = LETTERS.find(letter)
    return BADGE_COLORS[idx % len(BADGE_COLORS)] if idx >= 0 else "#888888"


def _clean_whisper_text(raw):
    """Sanitise a Whisper transcription chunk and return cleaned text or ''.

    Handles four distinct failure modes in order:
      1. CJK characters   — discard entire chunk (Whisper hallucinates Chinese/Japanese
                            on ambient noise even with language='en')
      2. Inline artifacts — strip '______' / '......' inside text so surrounding
                            words are preserved
      3. Alpha-less text  — discard if fewer than 3 alphabetic characters remain
                            after cleanup (catches '. . .', '...', etc.)
      4. Repetition loop  — deduplicate consecutive identical sentences (Whisper
                            decoder stuck in a loop)

    Parameters
    ----------
    raw : raw string from Whisper

    Returns
    -------
    str — cleaned text, or '' to signal the caller to discard the chunk
    """
    if not raw:
        return ""
    if _RE_CJK.search(raw):
        return ""
    text = _RE_INLINE_ARTIFACT.sub("", raw)
    text = _RE_MULTI_SPACE.sub(" ", text).strip()
    if sum(1 for c in text if c.isalpha()) < 3:
        return ""
    parts = re.split(r'(?<=[.!?])\s+', text)
    seen, deduped = set(), []
    for p in parts:
        key = p.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(p.strip())
    return " ".join(deduped).strip()


def _build_ollama_prompt(model_tag, topic_text, transcript_text, context=""):
    """Return a model-calibrated prompt string for binary topic detection.

    Each model branch has been individually tuned based on observed behaviour.
    Do NOT merge cases into a shared branch without re-testing.

    Parameters
    ----------
    model_tag       : Ollama model tag (e.g. 'gemma2:2b', 'qwen2.5')
    topic_text      : the topic description to detect
    transcript_text : the transcript excerpt to evaluate
    context         : optional meeting context injected at the top of the prompt

    Returns
    -------
    str — formatted prompt
    """
    ctx = f"Meeting context: {context.strip()}\n\n" if context.strip() else ""

    if model_tag == "gemma2:2b":
        # Gemma 2: historically over-sensitive — requires explicit strictness rules
        return (
            f"{ctx}You are a strict topic detector for a live meeting transcript.\n"
            f"Only output TRUE if the topic is DIRECTLY and EXPLICITLY the main focus of the excerpt.\n"
            f"Output FALSE for indirect mentions, tangential references, or general background context.\n\n"
            f"Topic: \"{topic_text}\"\n"
            f"Transcript: \"{transcript_text}\"\n"
            f"Answer (TRUE or FALSE only):"
        )
    else:
        # Qwen 2.5 — calibrated and working well, do NOT change this branch
        return (
            f"{ctx}You are monitoring a live meeting transcript for a specific topic.\n"
            f"Topic to alert on: \"{topic_text}\"\n"
            f"Transcript excerpt: \"{transcript_text}\"\n"
            f"Does this excerpt discuss the topic? Reply with TRUE or FALSE only."
        )


def transcribe_chunk(
    audio_int16,
    mode,
    whisper_model=None,
    recognizer=None,
    initial_prompt="",
    no_speech_max=0.70,
):
    """Transcribe one int16 audio chunk and return cleaned text ('' on failure).

    Parameters
    ----------
    audio_int16    : np.ndarray, dtype=int16, 16 kHz mono
    mode           : 'Deep Learning (Whisper)' or 'Non-DL (PocketSphinx)'
    whisper_model  : faster_whisper.WhisperModel instance  (Whisper mode)
    recognizer     : speech_recognition.Recognizer instance  (Sphinx mode)
    initial_prompt : rolling context string passed to Whisper (optional)
    no_speech_max  : discard Whisper segments with no_speech_prob above this

    Returns
    -------
    str — transcribed and cleaned text, or '' on failure or silence
    """
    if mode == "Deep Learning (Whisper)":
        audio_float32 = audio_int16.astype(np.float32) / 32767.0
        try:
            segments, _ = whisper_model.transcribe(
                audio_float32,
                beam_size=5,
                language="en",
                initial_prompt=initial_prompt or None,
                compression_ratio_threshold=1.8,
            )
            raw_text = " ".join(
                s.text for s in segments if s.no_speech_prob < no_speech_max
            ).strip()
            return _clean_whisper_text(raw_text)
        except Exception as exc:
            print(f"[Whisper] transcribe error: {exc}")
            return ""

    else:  # Non-DL (PocketSphinx)
        import speech_recognition as _sr
        audio_float = audio_int16.astype(np.float32)
        peak = np.max(np.abs(audio_float))
        if peak > 0:
            audio_norm = (audio_float * (27852.0 / peak)).clip(-32768, 32767).astype(np.int16)
        else:
            audio_norm = audio_int16
        audio_data = _sr.AudioData(audio_norm.tobytes(), 16000, 2)
        try:
            return recognizer.recognize_sphinx(audio_data)
        except _sr.UnknownValueError:
            return ""
        except Exception as exc:
            print(f"[PocketSphinx] {type(exc).__name__}: {exc}")
            return ""


def detect_topics(
    text,
    plan_mode,
    topics,
    finance_keywords=None,
    semantic_model=None,
    embedding_cache=None,
    meeting_context="",
):
    """Detect which topics are present in text and return their IDs.

    Parameters
    ----------
    text             : transcribed or ground-truth text to analyse
    plan_mode        : 'Non-DL (Keywords)', 'Transformer (Embeddings)', or a key
                       from OLLAMA_MODELS
    topics           : list of {'id': str, 'text': str}
    finance_keywords : keyword list used as fallback when a topic yields no
                       extractable content words
    semantic_model   : SentenceTransformer instance  (Embeddings mode)
    embedding_cache  : mutable dict for caching topic embeddings across calls;
                       pass the same dict object on every call to avoid
                       re-encoding the same topics
    meeting_context  : injected into LLM prompts to reduce false positives

    Returns
    -------
    list[str] — matched topic IDs (e.g. ['A', 'C']).  Empty list = no match.
    """
    if not text.strip():
        return []
    if embedding_cache is None:
        embedding_cache = {}
    if finance_keywords is None:
        finance_keywords = []

    matched_ids = []

    if plan_mode == "Non-DL (Keywords)":
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        for tp in topics:
            tp_all     = re.findall(r'\b\w+\b', tp["text"].lower())
            tp_content = [w for w in tp_all if w not in _STOP_WORDS and len(w) >= 3]
            if not tp_content:
                # Topic has no extractable keywords → fall back to finance CSV
                if set(finance_keywords).intersection(words):
                    matched_ids.append(tp["id"])
                continue
            tp_set    = set(tp_content)
            tp_bigrams = {
                f"{tp_content[i]} {tp_content[i+1]}"
                for i in range(len(tp_content) - 1)
            }
            hits       = tp_set.intersection(words)
            bigram_hit = any(bg in text_lower for bg in tp_bigrams)
            # Any single content-word OR bigram match fires the alert.
            # PocketSphinx transcription is imperfect — requiring 2 hits caused
            # misses whenever even one keyword was transcribed incorrectly.
            if hits or bigram_hit:
                matched_ids.append(tp["id"])

    elif plan_mode == "Transformer (Embeddings)":
        from sentence_transformers import util as _st_util
        text_emb = semantic_model.encode(text.lower())
        for tp in topics:
            key = tp["text"]
            if key not in embedding_cache:
                embedding_cache[key] = semantic_model.encode(key)
            sim = _st_util.cos_sim(text_emb, embedding_cache[key]).item()
            if sim > 0.45:
                matched_ids.append(tp["id"])

    elif plan_mode in OLLAMA_MODELS:
        ollama_model = OLLAMA_MODELS[plan_mode]
        for tp in topics:
            prompt = _build_ollama_prompt(
                ollama_model, tp["text"], text, meeting_context
            )
            payload = {
                "model":   ollama_model,
                "prompt":  prompt,
                "stream":  False,
                "options": {"temperature": 0.0},
            }
            try:
                res = requests.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=30,
                ).json()
                if "TRUE" in res.get("response", "").upper():
                    matched_ids.append(tp["id"])
            except Exception:
                pass

    return matched_ids
