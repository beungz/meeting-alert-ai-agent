import streamlit as st
import streamlit.components.v1 as components
import threading
import queue
import time
import numpy as np
import sounddevice as sd

# ---- Streamlit + PyTorch compatibility patch ----
# Streamlit's LocalSourcesWatcher calls list(module.__path__._path) on every
# sys.modules entry. torch.classes routes all attribute access through PyTorch's
# C++ class registry, so asking for __path__._path raises:
#   RuntimeError: Tried to instantiate class '__path__._path', but it does not exist!
# Fix: replace torch.classes.__path__ with a stand-in whose ._path is an empty
# list, so extract_paths() returns [] without raising.
try:
    import torch as _torch
    class _TorchClassesPath:
        _path: list = []
        def __iter__(self): return iter([])
        def __len__(self): return 0
    _torch.classes.__path__ = _TorchClassesPath()  # type: ignore[assignment]
    del _torch, _TorchClassesPath
except Exception:
    pass

# ---- Shared constants and pure functions (importable by eval.py too) ----
from agent_core import (
    LETTERS, BADGE_COLORS, OLLAMA_MODELS, FINANCE_KEYWORDS,
    badge_color, transcribe_chunk, detect_topics,
)


def listener_worker(device_id, q_audio, q_vol, run_flag, short_interval, long_interval):
    """Capture audio from a device and emit short/long chunk tuples to q_audio.

    Runs in a daemon thread.  Accumulates samples into two rolling buffers
    (one per interval length) and flushes each when its sample budget is reached.
    Volume readings are pushed to q_vol for the live audio-level indicator.

    Parameters
    ----------
    device_id      : sounddevice device index
    q_audio        : Queue receiving ('short'|'long', audio_int16, timestamp) tuples
    q_vol          : Queue receiving instantaneous volume ints (kept shallow)
    run_flag       : threading.Event — worker exits when cleared
    short_interval : seconds between short-chunk flushes
    long_interval  : seconds between long-chunk flushes (used for topic detection)
    """
    sample_rate  = 16000
    device_info  = sd.query_devices(device_id)
    num_channels = int(device_info.get('max_input_channels', 1)) or 1

    buffer_short, accumulated_short = [], 0
    buffer_long,  accumulated_long  = [], 0
    MAX_SAMPLES_SHORT = sample_rate * short_interval
    MAX_SAMPLES_LONG  = sample_rate * long_interval

    def callback(indata, frames, time_info, status):
        nonlocal buffer_short, accumulated_short, buffer_long, accumulated_long
        audio_data = (indata[:, 0] * 32767).astype(np.int16)
        vol = int(np.abs(audio_data).mean())
        if q_vol.qsize() < 2:
            q_vol.put(vol)
        if vol > 20 or len(buffer_short) > 0:
            buffer_short.append(audio_data)
            accumulated_short += len(audio_data)
            buffer_long.append(audio_data)
            accumulated_long  += len(audio_data)
        timestamp = time.time()
        if accumulated_short >= MAX_SAMPLES_SHORT:
            q_audio.put(("short", np.concatenate(buffer_short), timestamp))
            buffer_short, accumulated_short = [], 0
        if accumulated_long >= MAX_SAMPLES_LONG:
            q_audio.put(("long", np.concatenate(buffer_long), timestamp))
            buffer_long, accumulated_long = [], 0

    try:
        with sd.InputStream(samplerate=sample_rate, channels=num_channels,
                            device=device_id, callback=callback):
            while run_flag.is_set():
                time.sleep(0.1)
    except Exception as e:
        print(f"Audio error: {e}")


def scribe_worker(mode, q_audio, q_text, q_log, run_flag, initial_prompt=""):
    """Transcribe audio chunks from q_audio and forward text to q_text.

    Loads either WhisperModel or a PocketSphinx Recognizer depending on mode,
    then loops until run_flag is cleared.

    For Whisper, maintains a rolling context window (last ~50 words of real
    speech) so each chunk benefits from meeting-domain priming.  _after_silence
    flags a gap so the next real chunk re-injects the meeting context to
    re-anchor Whisper after a speaker change or pause.

    Parameters
    ----------
    mode           : 'Deep Learning (Whisper)' or 'Non-DL (PocketSphinx)'
    q_audio        : Queue of ('short'|'long', audio_int16, timestamp) tuples
    q_text         : Queue receiving ('short'|'long', text, timestamp) tuples
    q_log          : Queue for error messages
    run_flag       : threading.Event
    initial_prompt : optional meeting context seeded into Whisper's prompt
    """
    try:
        if mode == "Deep Learning (Whisper)":
            from faster_whisper import WhisperModel
            whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
            recognizer    = None
        else:
            import speech_recognition as sr
            whisper_model = None
            recognizer    = sr.Recognizer()
    except Exception as e:
        q_log.put(("error", f"MODEL LOAD ERROR: {e}", 0, None))
        return

    _CONTEXT_WORDS = 50
    _NO_SPEECH_MAX = 0.70
    _whisper_ctx   = initial_prompt.strip() if initial_prompt else ""
    _after_silence = False

    while run_flag.is_set():
        try:
            chunk_type, audio_chunk, t = q_audio.get(timeout=1)
            text = ""
            try:
                if mode == "Deep Learning (Whisper)":
                    # Re-inject meeting context on first chunk after a silent gap
                    if _after_silence and initial_prompt.strip():
                        reinject     = initial_prompt.strip() + " " + _whisper_ctx
                        _whisper_ctx = " ".join(reinject.split()[-_CONTEXT_WORDS:])
                    _after_silence = False

                    text = transcribe_chunk(
                        audio_chunk, mode,
                        whisper_model=whisper_model,
                        initial_prompt=_whisper_ctx,
                        no_speech_max=_NO_SPEECH_MAX,
                    )

                    if text:
                        combined     = (_whisper_ctx + " " + text) if _whisper_ctx else text
                        _whisper_ctx = " ".join(combined.split()[-_CONTEXT_WORDS:])
                    else:
                        _after_silence = True  # silence/noise — re-inject context next chunk
                else:
                    text = transcribe_chunk(audio_chunk, mode, recognizer=recognizer)

            except Exception as transcribe_e:
                print(f"[Scribe] Unexpected error: {type(transcribe_e).__name__}: {transcribe_e}")

            if text:
                q_text.put((chunk_type, text, t))
        except queue.Empty:
            continue


def analyst_worker(plan_mode, topics_ref, topics_lock, q_text, q_log, run_flag, meeting_context=""):
    """Detect topics in long transcript chunks and emit alerts to q_log.

    Each matched topic fires a separate 'alert_meta' message so the UI can
    display per-topic badges independently.

    Parameters
    ----------
    plan_mode       : planning mode string (key into OLLAMA_MODELS or special value)
    topics_ref      : shared list of {'id', 'text'} dicts (mutated by UI thread)
    topics_lock     : threading.Lock protecting topics_ref
    q_text          : Queue of ('short'|'long', text, timestamp) tuples
    q_log           : Queue receiving UI update tuples
    run_flag        : threading.Event
    meeting_context : injected into LLM prompts to reduce false-positive alerts
    """
    embedding_cache = {}
    semantic_model  = None

    try:
        if plan_mode == "Transformer (Embeddings)":
            from sentence_transformers import SentenceTransformer
            semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        q_log.put(("error", f"MODEL LOAD ERROR: {e}", 0, None))
        return

    while run_flag.is_set():
        try:
            chunk_type, text, t = q_text.get(timeout=1)

            if chunk_type == "short":
                q_log.put(("short", text, t, None))
                continue

            with topics_lock:
                current_topics = [tp.copy() for tp in topics_ref if tp.get("text", "").strip()]

            if not current_topics:
                q_log.put(("long", text, t, None))
                continue

            matched_ids = detect_topics(
                text, plan_mode, current_topics,
                finance_keywords=FINANCE_KEYWORDS,
                semantic_model=semantic_model,
                embedding_cache=embedding_cache,
                meeting_context=meeting_context,
            )

            if matched_ids:
                q_log.put(("long_alert", text, t, matched_ids[0]))
                for tid in matched_ids:
                    q_log.put(("alert_meta", text[:60] + "...", t, tid))
            else:
                q_log.put(("long", text, t, None))

        except queue.Empty:
            continue


st.set_page_config(page_title="Earnings Call Agent", layout="wide")

if 'sys_state' not in st.session_state:
    st.session_state.sys_state = {
        'q_audio': queue.Queue(),
        'q_text': queue.Queue(),
        'q_log': queue.Queue(),
        'q_vol': queue.Queue(),
        'run_flag': threading.Event(),
        'topics_lock': threading.Lock(),
        'topics_list': []
    }

if 'topics' not in st.session_state:
    st.session_state.topics = [{"uid": 0, "text": "Sales Growth"}]
    st.session_state.topic_uid_counter = 1

if 'initial_context' not in st.session_state:
    st.session_state.initial_context = ""

sys = st.session_state.sys_state

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<div style='font-size:1.1rem;font-weight:700;margin-bottom:0.4rem;'>Agent Configuration</div>", unsafe_allow_html=True)

    # ---- Multi-Topic Management ----
    st.markdown("<div style='font-size:0.85rem;font-weight:600;color:#aaa;margin-bottom:4px;'>MONITORING TOPICS</div>", unsafe_allow_html=True)

    to_delete_uid = None
    for idx, topic in enumerate(st.session_state.topics):
        letter = LETTERS[idx] if idx < 26 else f"T{idx}"
        uid = topic['uid']
        bc = badge_color(letter)

        c_lbl, c_inp, c_del = st.columns([0.32, 3.5, 0.38])
        c_lbl.markdown(
            f"<div style='margin-top:5px;font-size:0.95rem;font-weight:bold;"
            f"background:{bc};color:#fff;border-radius:4px;"
            f"text-align:center;padding:3px 2px;'>{letter}</div>",
            unsafe_allow_html=True
        )
        new_text = c_inp.text_input(
            f"Topic {letter}",
            value=topic['text'],
            key=f"ta_{uid}",
            label_visibility="collapsed",
        )
        topic['text'] = new_text
        if c_del.button("✕", key=f"del_{uid}", disabled=len(st.session_state.topics) == 1, help="Remove topic"):
            to_delete_uid = uid

    if to_delete_uid is not None:
        st.session_state.topics = [t for t in st.session_state.topics if t['uid'] != to_delete_uid]
        st.rerun()

    if st.button("➕ Add Topic", use_container_width=True, disabled=len(st.session_state.topics) >= 26):
        st.session_state.topics.append({"uid": st.session_state.topic_uid_counter, "text": ""})
        st.session_state.topic_uid_counter += 1
        st.rerun()

    # Sync to shared list for background analyst thread
    with sys['topics_lock']:
        sys['topics_list'][:] = [
            {"id": LETTERS[i] if i < 26 else f"T{i}", "text": t['text']}
            for i, t in enumerate(st.session_state.topics)
        ]

    st.markdown("<hr class='sb-sep'>", unsafe_allow_html=True)
    perc_mode = st.radio("Perception Model (Ears)", ["Deep Learning (Whisper)", "Non-DL (PocketSphinx)"])
    plan_options = (
        list(OLLAMA_MODELS.keys()) + ["Transformer (Embeddings)"]
        if perc_mode == "Deep Learning (Whisper)"
        else ["Non-DL (Keywords)"]
    )
    plan_mode = st.selectbox("Planning Model (Brain)", plan_options)

    with st.expander("Meeting Context", expanded=False):
        st.markdown(
            "<div style='font-size:0.76rem;color:#888;margin-bottom:5px;line-height:1.4;'>"
            "Used by <b>Whisper</b> (improves transcription accuracy from the first chunk) "
            "and <b>LLM Ollama</b> planning models (reduces false-positive alerts). "
            "Not used by PocketSphinx or Transformer/Keywords modes."
            "</div>",
            unsafe_allow_html=True,
        )
        st.text_area(
            "Meeting context",
            key="initial_context",
            label_visibility="collapsed",
            placeholder="e.g. Q4 2024 earnings call for Apple Inc. Focus on revenue growth, operating margins, and forward guidance.",
            height=68,
        )
    st.markdown("<hr class='sb-sep'>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.82rem;font-weight:600;color:#aaa;margin-bottom:0px;'>TRANSCRIPTION PACING</div>", unsafe_allow_html=True)
    short_interval = st.slider("Short Interval (secs)", 2, 10, 4)
    long_interval = st.slider("Long Interval (secs)", 10, 45, 15)
    st.markdown("<hr class='sb-sep'>", unsafe_allow_html=True)

    devices = sd.query_devices()
    device_names, seen_devices = [], set()
    for i, d in enumerate(devices):
        if d['max_input_channels'] == 0: continue
        name_lower = d['name'].lower()
        if any(ign in name_lower for ign in ["mapper", "primary sound capture"]): continue
        fingerprint = name_lower[:35]
        if fingerprint not in seen_devices:
            seen_devices.add(fingerprint)
            tag = "[ZOOM/TEAMS]" if any(x in name_lower for x in ["cable", "loopback", "stereo mix"]) else "[PHYSICAL MIC]"
            device_names.append(f"{i}: {tag} {d['name']}")

    selected_device_str = st.selectbox("Audio Input Source", device_names)
    selected_device_id = int(selected_device_str.split(":")[0]) if selected_device_str else None

    st.markdown("<hr class='sb-sep'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    if col1.button("▶️ Start", type="primary", disabled=sys['run_flag'].is_set()):
        if selected_device_id is not None:
            ctx = st.session_state.get('initial_context', '')
            sys['run_flag'].set()
            threading.Thread(target=listener_worker, args=(selected_device_id, sys['q_audio'], sys['q_vol'], sys['run_flag'], short_interval, long_interval), daemon=True).start()
            threading.Thread(target=scribe_worker, args=(perc_mode, sys['q_audio'], sys['q_text'], sys['q_log'], sys['run_flag'], ctx), daemon=True).start()
            threading.Thread(target=analyst_worker, args=(plan_mode, sys['topics_list'], sys['topics_lock'], sys['q_text'], sys['q_log'], sys['run_flag'], ctx), daemon=True).start()
            # No explicit st.rerun() — Streamlit's natural button-click rerun handles this.
            # Adding st.rerun() here causes a double-rerun race with the live loop and
            # can disconnect the WebSocket, showing "Is Streamlit still running?" popup.

    if col2.button("⏹️ Stop", disabled=not sys['run_flag'].is_set()):
        sys['run_flag'].clear()
        # No explicit st.rerun() — same reason as above. The natural button rerun fires,
        # Section 4 captures is_running=False, JS sets vol-bar to Offline, live loop exits.

    # ---- Jitter-free status & volume (static shell, updated via JS) ----
    st.markdown("<hr class='sb-sep'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.85rem;font-weight:600;color:#aaa;margin-bottom:4px;'>AGENT STATUS</div>", unsafe_allow_html=True)
    st.markdown("""
<div id="vol-widget" style="min-height:90px;display:flex;flex-direction:column;gap:6px;">
    <div id="vol-status" style="background:rgba(255,255,255,0.05);padding:8px 10px;border-radius:6px;font-size:13px;line-height:1.4;">
        🔴 Offline. Click Start.
    </div>
    <div>
        <div style="font-size:11px;margin-bottom:3px;color:#888;">Live Audio Level</div>
        <div style="width:100%;height:8px;background-color:rgba(128,128,128,0.2);border-radius:4px;overflow:hidden;">
            <div id="vol-bar-fill" style="width:0%;height:100%;background-color:#32cd32;transition:width 0.1s ease-in-out;"></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


st.markdown("<h1 style='font-size:1.35rem;margin:0 0 0.2rem 0;'>🎧 Live Meeting AI Copilot</h1>", unsafe_allow_html=True)

if 'history_alerts' not in st.session_state:
    st.session_state.history_short = []
    st.session_state.history_long = []
    st.session_state.history_alerts = []

st.markdown("---")

# Navigation bar
alert_options = {
    a["time"]: f"[{a.get('topic_id','?')}] {a['snippet']}"
    for a in st.session_state.history_alerts
}
alert_choices = ["-- Auto-Scroll to Live Audio --"] + list(alert_options.values())

nav_col1, nav_col2, nav_col3 = st.columns([2.4, 0.9, 0.9])
with nav_col1:
    selected_label = st.selectbox("Jump to detected topic:", alert_choices, label_visibility="collapsed")
with nav_col2:
    live_update = st.checkbox("Live Refresh", value=True)
with nav_col3:
    if st.button("Clear All", type="secondary", use_container_width=True):
        st.session_state.history_short.clear()
        st.session_state.history_long.clear()
        st.session_state.history_alerts.clear()
        with sys['q_log'].mutex: sys['q_log'].queue.clear()
        st.rerun()

selected_time = None
if selected_label != "-- Auto-Scroll to Live Audio --":
    selected_time = [k for k, v in alert_options.items() if v == selected_label][0]

st.markdown("---")
col_short, col_long, col_alert = st.columns([1, 1, 1])

with col_short:
    st.markdown("<div class='col-header'>Short Transcript</div>", unsafe_allow_html=True)
    short_container = st.empty()
with col_long:
    st.markdown("<div class='col-header'>Continuous Context</div>", unsafe_allow_html=True)
    long_container = st.empty()
with col_alert:
    st.markdown("<div class='col-header'>Attention Alerts</div>", unsafe_allow_html=True)
    alert_container = st.empty()

# --- CSS ---
badge_css = "\n    ".join(
    f".badge-{LETTERS[i]} {{ background:{BADGE_COLORS[i % len(BADGE_COLORS)]}; color:#fff; }}"
    for i in range(26)
)

st.markdown(f"""
<style>
    /* Layout compaction */
    .block-container {{ padding-top: 0.6rem !important; padding-bottom: 0.3rem !important; }}
    div[data-testid="stSidebar"] > div {{ padding-top: 0.5rem !important; }}
    .stMarkdown p {{ margin-bottom: 0.15rem; }}

    /* Sidebar — compact element spacing */
    .sb-sep {{ margin: 3px 0 !important; border: none !important; border-top: 1px solid rgba(128,128,128,0.18) !important; }}
    section[data-testid="stSidebar"] .stTextInput  {{ margin-bottom: -8px !important; }}
    section[data-testid="stSidebar"] .stTextArea   {{ margin-bottom: -4px !important; }}
    section[data-testid="stSidebar"] .stRadio      {{ margin-bottom: -4px !important; }}
    section[data-testid="stSidebar"] .stSelectbox  {{ margin-bottom: -4px !important; }}
    section[data-testid="stSidebar"] .stSlider     {{ margin-bottom: -6px !important; }}
    section[data-testid="stSidebar"] .stButton     {{ margin-bottom: -2px !important; }}
    section[data-testid="stSidebar"] .stExpander   {{ margin-bottom: -2px !important; }}
    section[data-testid="stSidebar"] label         {{ font-size: 0.82rem !important; }}
    section[data-testid="stSidebar"] p             {{ margin-bottom: 0.1rem !important; }}
    /* Shrink text_input height in sidebar topic rows */
    section[data-testid="stSidebar"] .stTextInput input {{ padding-top: 4px !important; padding-bottom: 4px !important; font-size: 0.83rem !important; }}

    /* Column headers */
    .col-header {{ font-size: 0.95rem; font-weight: 600; margin-bottom: 5px; }}

    /* Scroll boxes — auto-height for 1080p+ */
    .scroll-box {{
        height: calc(100vh - 380px);
        min-height: 300px;
        max-height: 640px;
        overflow-y: auto;
        padding: 10px;
        background: rgba(128,128,128,0.05);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 8px;
        font-size: 0.88rem;
    }}
    .short-item {{ margin-bottom: 4px; padding: 3px 5px; border-radius: 4px; line-height: 1.35; }}
    .long-item {{ display: inline; line-height: 1.65; padding: 1px 3px; border-radius: 3px; }}
    .alert-item {{ margin-bottom: 6px; padding: 7px 9px; background: rgba(255,0,0,0.09); border-left: 3px solid #e74c3c; border-radius: 4px; font-size: 0.86rem; }}

    /* Highlights */
    .hl-yellow {{ background-color: #ffd700; color: #000; font-weight: 500; }}
    .hl-green  {{ background-color: #32cd32; color: #000; font-weight: bold; box-shadow: 0 0 5px green; }}

    /* Topic badges */
    .topic-badge {{
        display: inline-block;
        padding: 0 5px;
        border-radius: 3px;
        font-size: 0.7rem;
        font-weight: bold;
        margin-right: 3px;
        vertical-align: middle;
        line-height: 1.6;
    }}
    {badge_css}
</style>
""", unsafe_allow_html=True)


# Always read volume & running state — ensures JS/vol-bar updates even after Stop
latest_vol = 0
while not sys['q_vol'].empty():
    latest_vol = sys['q_vol'].get()
normalized_vol = min(latest_vol / 2000.0, 1.0) * 100 if latest_vol >= 0 else 0
is_running = sys['run_flag'].is_set()
safe_device = (selected_device_str or "").replace("'", "\\'")
target_short_id = None

# Only drain queues and render transcript panels when there is content
if is_running or st.session_state.history_short:

    while not sys['q_log'].empty():
        raw = sys['q_log'].get()
        msg_type, text, t = raw[0], raw[1], raw[2]
        topic_id = raw[3] if len(raw) > 3 else None

        if msg_type == "short":
            st.session_state.history_short.append({"time": t, "text": text, "alert_topic": None})
        elif msg_type == "long":
            st.session_state.history_long.append({"time": t, "text": text, "alert_topic": None})
        elif msg_type == "long_alert":
            st.session_state.history_long.append({"time": t, "text": text, "alert_topic": topic_id})
        elif msg_type == "alert_meta":
            st.session_state.history_alerts.append({"time": t, "snippet": text, "topic_id": topic_id or "?"})
            for s in st.session_state.history_short:
                if (t - long_interval - 1) <= s["time"] <= (t + 1):
                    if s.get("alert_topic") is None:
                        s["alert_topic"] = topic_id

    short_html = "<div id='short_box' class='scroll-box'>"
    for item in st.session_state.history_short:
        # Handle both (alert_topic) and legacy (alert) field names
        at = item.get("alert_topic") or ("A" if item.get("alert") else None)
        css_class = ""
        if selected_time and (selected_time - long_interval - 1) <= item["time"] <= (selected_time + 1):
            css_class = "hl-green"
            if not target_short_id: target_short_id = f"short_{item['time']}"
        elif at:
            css_class = "hl-yellow"
        badge = f"<span class='topic-badge badge-{at}'>{at}</span>" if at else ""
        short_html += f"<div id='short_{item['time']}' class='short-item {css_class}'>{badge}{item['text']}</div>"
    short_html += "<div id='short_bottom'></div></div>"
    short_container.markdown(short_html, unsafe_allow_html=True)

    long_html = "<div id='long_box' class='scroll-box'>"
    for item in st.session_state.history_long:
        at = item.get("alert_topic") or ("A" if item.get("alert") else None)
        if selected_time == item["time"]:
            css_class = "hl-green"
        elif at:
            css_class = "hl-yellow"
        else:
            css_class = ""
        badge = f"<span class='topic-badge badge-{at}'>{at}</span>" if at else ""
        long_html += f"<span id='long_{item['time']}' class='long-item {css_class}'>{badge}{item['text']} </span>"
    long_html += "<div id='long_bottom'></div></div>"
    long_container.markdown(long_html, unsafe_allow_html=True)

    alert_html = "<div id='alert_box' class='scroll-box'>"
    for item in reversed(st.session_state.history_alerts):
        tid = item.get("topic_id", "?")
        bc = badge_color(tid)
        item_t = item["time"]
        bg_style = "background:rgba(50,205,50,0.2);border-left-color:#27ae60;" if item_t == selected_time else ""
        alert_html += (
            f"<div id='alert_{item_t}' class='alert-item' style='{bg_style}'>"
            f"<span class='topic-badge' style='background:{bc};color:#fff;'>{tid}</span>"
            f" <b>Topic {tid} Detected:</b><br>{item['snippet']}"
            f"</div>"
        )
    alert_html += "<div id='alert_bottom'></div></div>"
    alert_container.markdown(alert_html, unsafe_allow_html=True)

# JS volume + scroll — ALWAYS runs so vol-bar reflects Stop state
is_jumping     = str(selected_time is not None).lower()
is_live        = str(live_update).lower()
is_running_js  = "true" if is_running else "false"
safe_target_id = target_short_id if target_short_id else "None"

scroll_js = f"""
<script>
    const doc = window.parent.document;

    // Volume bar — direct DOM update, zero Streamlit re-render
    const volFill = doc.getElementById('vol-bar-fill');
    if (volFill) volFill.style.width = '{normalized_vol:.1f}%';
    const volStatus = doc.getElementById('vol-status');
    if (volStatus) {{
        if ("{is_running_js}" === "true") {{
            volStatus.innerHTML = '🟢 Listening on:<br><span style="font-size:0.8em;color:#aaa;">{safe_device}</span>';
        }} else {{
            volStatus.innerHTML = '🔴 Offline. Click Start.';
        }}
    }}

    // Scroll logic
    function forceScroll() {{
        if ("{is_jumping}" === "true") {{
            const longEl = doc.getElementById('long_{selected_time}');
            if (longEl) longEl.scrollIntoView({{behavior: 'smooth', block: 'center'}});
            if ("{safe_target_id}" !== "None") {{
                const shortEl = doc.getElementById('{safe_target_id}');
                if (shortEl) shortEl.scrollIntoView({{behavior: 'smooth', block: 'center'}});
            }}
            const alertEl = doc.getElementById('alert_{selected_time}');
            if (alertEl) alertEl.scrollIntoView({{behavior: 'smooth', block: 'center'}});
        }} else if ("{is_live}" === "true") {{
            const shortBottom = doc.getElementById('short_bottom');
            const longBottom  = doc.getElementById('long_bottom');
            const alertBottom = doc.getElementById('alert_bottom');
            if (shortBottom) shortBottom.scrollIntoView({{behavior: 'instant', block: 'end'}});
            if (longBottom)  longBottom.scrollIntoView({{behavior: 'instant', block: 'end'}});
            if (alertBottom) alertBottom.scrollIntoView({{behavior: 'instant', block: 'end'}});
        }}
    }}

    forceScroll();
    let t = setInterval(forceScroll, 50);
    setTimeout(() => clearInterval(t), 500);
</script>
"""
components.html(scroll_js, height=0)

# ALWAYS checked so the loop stops cleanly when run_flag clears
if is_running and live_update:
    time.sleep(0.2)
    st.rerun()
