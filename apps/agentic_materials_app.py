import base64
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from agent.orchestrator import run_agent
from agent.tools import MOTOR_SUMMARY
import pandas as pd

# ── Voice mic (injected beside the chat send arrow) ───────────────────────────
_VOICE = st.components.v2.component(
    "voice_input_clear",
    html="<div id='voice-host'></div>",
    css="#voice-host { display: none; }",
    js="""
export default function (component) {
  const { data, setTriggerValue } = component
  const MIC_ID = "inline-voice-mic"
  const STYLE_ID = "inline-voice-mic-style"
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition

  if (!document.getElementById(STYLE_ID)) {
    const style = document.createElement("style")
    style.id = STYLE_ID
    style.textContent = `
      #inline-voice-mic {
        width: 2.25rem;
        height: 2.25rem;
        border: none;
        background: transparent;
        color: #d97757;
        border-radius: 999px;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        padding: 0;
        margin: 0 2px 0 0;
      }
      #inline-voice-mic:hover { background: rgba(217,119,87,0.12); }
      #inline-voice-mic.active {
        background: #d97757;
        color: #fff;
        animation: inline-voice-pulse 1.2s infinite;
      }
      @keyframes inline-voice-pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(217,119,87,0.45); }
        50% { box-shadow: 0 0 0 7px rgba(217,119,87,0); }
      }
    `
    document.head.appendChild(style)
  }

  if (!window.__voiceMicState) {
    window.__voiceMicState = {
      listening: false,
      finalText: "",
      recognition: null,
      lastReset: -1,
      lastSent: "",
    }
  }
  const state = window.__voiceMicState
  state.setTriggerValue = setTriggerValue

  function fillChatBox(text) {
    const textarea = document.querySelector('[data-testid="stChatInputTextArea"]')
    if (!textarea) return
    const proto = window.HTMLTextAreaElement.prototype
    const desc = Object.getOwnPropertyDescriptor(proto, "value")
    if (desc && desc.set) desc.set.call(textarea, text)
    else textarea.value = text
    textarea.dispatchEvent(new Event("input", { bubbles: true }))
    textarea.dispatchEvent(new Event("change", { bubbles: true }))
  }

  function setMicActive(on) {
    document.querySelectorAll("#" + MIC_ID).forEach((b) => {
      b.classList.toggle("active", on)
    })
  }

  function resetVoice() {
    state.listening = false
    state.finalText = ""
    if (state.recognition) {
      try { state.recognition.stop() } catch (_) {}
    }
    setMicActive(false)
    fillChatBox("")
  }

  function ensureRecognition() {
    if (state.recognition || !SpeechRecognition) return
    const recognition = new SpeechRecognition()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = "en-US"
    recognition.onresult = (e) => {
      if (!state.listening) return
      let interim = ""
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const t = e.results[i][0].transcript
        if (e.results[i].isFinal) state.finalText += t + " "
        else interim += t
      }
      fillChatBox((state.finalText + interim).trim())
    }
    recognition.onerror = (e) => {
      if (e.error === "not-allowed") {
        fillChatBox("Allow microphone access in the address bar, then try again.")
      }
      state.listening = false
      setMicActive(false)
    }
    recognition.onend = () => {
      if (state.listening) {
        try { recognition.start() } catch (_) {}
      }
    }
    state.recognition = recognition
  }

  function startListening() {
    if (!SpeechRecognition) return
    ensureRecognition()
    state.finalText = ""
    state.lastSent = ""
    state.listening = true
    setMicActive(true)
    fillChatBox("")
    try { state.recognition.start() } catch (_) {}
  }

  function stopListening(send) {
    const text = state.finalText.trim()
    state.listening = false
    setMicActive(false)
    if (state.recognition) {
      try { state.recognition.stop() } catch (_) {}
    }
    if (send && text && text !== state.lastSent) {
      state.lastSent = text
      state.setTriggerValue("submitted", text)
    }
    fillChatBox("")
    state.finalText = ""
  }

  function watchSendButton() {
    const sendBtn = document.querySelector('[data-testid="stChatInputSubmitButton"]')
    if (!sendBtn || sendBtn.dataset.voiceWatch === "1") return
    sendBtn.dataset.voiceWatch = "1"
    sendBtn.addEventListener("click", () => {
      state.listening = false
      setMicActive(false)
      if (state.recognition) {
        try { state.recognition.stop() } catch (_) {}
      }
      state.finalText = ""
    })
  }

  function placeMic() {
    const sendBtn = document.querySelector('[data-testid="stChatInputSubmitButton"]')
    if (!sendBtn || !sendBtn.parentElement) return false
    watchSendButton()
    if (sendBtn.parentElement.querySelector("#" + MIC_ID)) {
      setMicActive(state.listening)
      return true
    }

    const btn = document.createElement("button")
    btn.id = MIC_ID
    btn.type = "button"
    btn.title = "Voice input"
    btn.setAttribute("aria-label", "Voice input")
    btn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="9" y="1" width="6" height="12" rx="3"></rect>
        <path d="M5 10a7 7 0 0 0 14 0"></path>
        <line x1="12" y1="17" x2="12" y2="22"></line>
        <line x1="8" y1="22" x2="16" y2="22"></line>
      </svg>`
    if (state.listening) btn.classList.add("active")
    if (!SpeechRecognition) {
      btn.disabled = true
      btn.style.opacity = "0.35"
      btn.title = "Voice not supported in this browser"
    } else {
      btn.addEventListener("click", (e) => {
        e.preventDefault()
        e.stopPropagation()
        if (state.listening) stopListening(true)
        else startListening()
      })
    }
    sendBtn.parentElement.insertBefore(btn, sendBtn)
    return true
  }

  const token = (data && data.reset_token) || 0
  if (state.lastReset !== token) {
    state.lastReset = token
    resetVoice()
  }

  if (!window.__voiceMicPlaced) {
    window.__voiceMicPlaced = true
    placeMic()
    const observer = new MutationObserver(() => placeMic())
    observer.observe(document.body, { childList: true, subtree: true })
  } else {
    placeMic()
  }
}
""",
)


def _mount_voice_mic():
    if "voice_reset" not in st.session_state:
        st.session_state.voice_reset = 0
    result = _VOICE(
        key="voice_mic",
        data={"reset_token": int(st.session_state.voice_reset)},
        on_submitted_change=lambda: None,
    )
    submitted = getattr(result, "submitted", None)
    if isinstance(submitted, str) and submitted.strip():
        text = submitted.strip()
        if text == st.session_state.get("_last_spoken"):
            return None
        st.session_state._last_spoken = text
        return text
    return None


def _after_prompt_sent():
    st.session_state.voice_reset = int(st.session_state.get("voice_reset", 0)) + 1
    st.session_state["_clear_chat_input"] = True
    st.session_state.pop("_last_spoken", None)


# ── Greeting (clock) ─────────────────────────────────────────────────────────
hour = datetime.now().hour
if 5 <= hour < 12:
    GREETING = "Good morning"
elif 12 <= hour < 17:
    GREETING = "Good afternoon"
else:
    GREETING = "Good evening"

st.set_page_config(
    page_title="Materials Intelligence",
    page_icon=":material/biotech:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Graphs need a Python-side flag; the page itself follows Streamlit's ⋮ theme.
try:
    is_dark = getattr(st.context.theme, "type", "light") == "dark"
except Exception:
    is_dark = hour >= 19 or hour < 7

# Colors come from Streamlit's live --st-* tokens so Light/Dark/System
# in the ⋮ menu apply immediately. Do not paint page/sidebar backgrounds.
st.html("""
<style>
:root {
    --mi-surface: var(--st-secondary-background-color, color-mix(in srgb, currentColor 7%, transparent));
    --mi-text: var(--st-text-color, inherit);
    --mi-muted: var(--st-gray-color, color-mix(in srgb, currentColor 62%, transparent));
    --mi-border: var(--st-border-color, color-mix(in srgb, currentColor 16%, transparent));
    --mi-accent: var(--st-primary-color, #d97757);
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 7rem !important;
    max-width: 1100px;
    margin: 0 auto;
}

[data-testid="stChatInput"] {
    border-radius: 16px !important;
}

.hero-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 42vh;
    padding-top: 1.5rem;
}
.hero-title {
    font-size: 2.05rem;
    font-weight: 700;
    color: var(--st-heading-color, var(--mi-text));
    margin-bottom: 0.3rem;
    text-align: center;
}
.hero-sub {
    font-size: 0.97rem;
    color: var(--mi-muted);
    text-align: center;
    max-width: 520px;
    line-height: 1.6;
    margin-bottom: 2rem;
}

.hist-wrap {
    max-height: 340px;
    overflow-y: auto;
    padding-right: 2px;
}
.hist-item {
    background: var(--mi-surface);
    border: 1px solid var(--mi-border);
    border-radius: 8px;
    padding: 0.45rem 0.65rem;
    margin-bottom: 0.3rem;
    font-size: 0.76rem;
    line-height: 1.45;
    color: var(--mi-text);
}
.hist-role-u { color: var(--mi-accent); font-weight: 600; }
.hist-role-a { color: var(--mi-muted); font-weight: 600; }
.tool-badge {
    display: inline-block;
    background: transparent;
    border: 1px solid var(--mi-border);
    color: var(--mi-muted);
    border-radius: 4px;
    font-size: 0.68rem;
    padding: 0.05rem 0.38rem;
    margin: 0.1rem 0.1rem 0 0;
    font-family: monospace;
}

.answer-card {
    background: var(--mi-surface);
    border: 1px solid var(--mi-border);
    border-radius: 12px;
    padding: 1.15rem 1.4rem;
    margin-top: 0.4rem;
    font-size: 0.96rem;
    line-height: 1.7;
    color: var(--mi-text);
}

.conflict-box {
    background: var(--st-orange-background-color, color-mix(in srgb, #d97757 16%, transparent));
    border: 1.5px solid var(--mi-accent);
    border-radius: 10px;
    padding: 0.75rem 1.1rem;
    color: var(--st-orange-color, var(--mi-accent));
    font-weight: 600;
    font-size: 0.9rem;
    margin: 0.5rem 0 0.8rem;
}

[data-testid="stMetric"] {
    background: var(--mi-surface);
    border: 1px solid var(--mi-border);
    border-radius: 8px;
    padding: 0.5rem 0.9rem;
}

[data-testid="stPillsInput"] button {
    border-radius: 20px !important;
    font-size: 0.83rem !important;
}

[data-testid="stBidiComponent"] {
    height: 0 !important;
    min-height: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.84rem !important;
}

.hist-wrap::-webkit-scrollbar { width: 4px; }
.hist-wrap::-webkit-scrollbar-thumb {
    background: var(--mi-border);
    border-radius: 4px;
}

/* Mermaid node names follow the page text color (white in dark mode). */
[data-testid="stMermaidChart"] svg,
[class*="stMermaid"] svg {
    color: inherit;
}
[data-testid="stMermaidChart"] .node text,
[data-testid="stMermaidChart"] .nodeLabel,
[data-testid="stMermaidChart"] .label,
[data-testid="stMermaidChart"] .node span,
[class*="stMermaid"] .node text,
[class*="stMermaid"] .nodeLabel,
[class*="stMermaid"] .label,
[class*="stMermaid"] .node span {
    fill: currentColor !important;
    color: currentColor !important;
}
</style>
""")

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sem_path" not in st.session_state:
    st.session_state.sem_path = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Materials Intelligence")
    st.caption("Axle fracture · SEM · ISO standards · motor load")
    st.space("small")

    env_key = os.getenv("GROQ_API_KEY", "")
    with st.expander("API key", expanded=False):
        groq_key = st.text_input(
            "Groq API key",
            value=env_key,
            type="password",
            placeholder="gsk_…",
            help="Click API key above to show or hide this field.",
        )
        if groq_key:
            st.caption("Key is loaded")
        else:
            st.caption("Paste a Groq key, then collapse this row to hide it.")
    if not groq_key:
        groq_key = env_key

    with st.container(border=True):
        st.caption("SEM image")
        uploaded = st.file_uploader(
            "Upload an SEM fracture image",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            label_visibility="collapsed",
        )
        if uploaded:
            suffix = Path(uploaded.name).suffix or ".png"
            tmp = Path(tempfile.gettempdir()) / f"agent_sem{suffix}"
            tmp.write_bytes(uploaded.getvalue())
            st.session_state.sem_path = str(tmp)
            st.image(uploaded, caption=uploaded.name)
            st.caption("Ready to classify")

    profile_id = None
    if MOTOR_SUMMARY.exists():
        profiles = pd.read_csv(MOTOR_SUMMARY)["profile_id"].astype(int).tolist()
        with st.container(border=True):
            st.caption("Motor session")
            choice = st.selectbox(
                "Plant motor session",
                options=["(none)"] + [str(p) for p in profiles],
                label_visibility="collapsed",
                help="Copper winding temperature and shaft torque from this plant motor session.",
            )
            if choice != "(none)":
                profile_id = int(choice)

    n_ex = len(st.session_state.messages) // 2
    with st.expander(
        f"Recent chats · {n_ex}",
        expanded=bool(st.session_state.messages),
    ):
        if not st.session_state.messages:
            st.caption("No messages yet.")
        else:
            items_html = '<div class="hist-wrap">'
            for msg in st.session_state.messages:
                role_html = (
                    '<span class="hist-role-u">You</span>'
                    if msg["role"] == "user"
                    else '<span class="hist-role-a">Agent</span>'
                )
                preview = str(msg["content"])[:110].replace("\n", " ").replace("<", "&lt;").replace(">", "&gt;")
                ellipsis = "…" if len(str(msg["content"])) > 110 else ""
                tools = msg.get("trace", [])
                badges = "".join(
                    f'<span class="tool-badge">{t["name"].replace("_", " ")}</span>'
                    for t in tools
                )
                items_html += (
                    f'<div class="hist-item">{role_html} {preview}{ellipsis}'
                    f'{"<br>" + badges if badges else ""}</div>'
                )
            items_html += "</div>"
            st.markdown(items_html, unsafe_allow_html=True)
            if st.button("Clear chat", width="stretch", type="tertiary"):
                st.session_state.messages = []
                st.session_state.sem_path = None
                st.rerun()

    st.markdown("---")

# ── Main area ─────────────────────────────────────────────────────────────────
TOOL_ICONS = {
    "predict_axle_fracture": ":material/bar_chart:",
    "classify_sem_image":    ":material/microscope:",
    "search_iso_standards":  ":material/menu_book:",
    "assess_motor_session":  ":material/electric_bolt:",
    "query_knowledge_graph": ":material/hub:",
    "detect_conflict":       ":material/gavel:",
}

SUGGESTIONS = {
    "HV10 = 520 — brittle?":    "This hardened steel axle has Mean HV10 = 520. Is it a brittle fracture risk, and which ISO hardness standard applies?",
    "Hot motor + axle risk":    "Use motor profile 29. Copper windings at 141 °C, torque peaks at 261 Nm. How does that shaft load relate to brittle risk?",
    "Sample prep in the lab":   "How do I prepare a metallographic sample in the lab according to the textbooks?",
    "Classify SEM image":       "Classify my uploaded SEM image and describe the fracture mode.",
}

def _kg_edges(trace: list | None) -> list | None:
    for step in trace or []:
        if step.get("name") == "query_knowledge_graph":
            edges = (step.get("result") or {}).get("edges") or []
            if edges:
                return edges
    return None


def _render_kg(edges: list) -> None:
    import importlib
    import agent.knowledge_graph as kg
    importlib.reload(kg)
    # ── 1. Connected flowchart (Mermaid) ──────────────────────────────
    mermaid = kg.edges_to_mermaid(edges, dark=is_dark)
    if mermaid:
        st.caption(":material/hub: Knowledge graph")
        st.mermaid_chart(mermaid, width="stretch")
    # ── 2. Styled digital-twin card (SVG) ─────────────────────────────
    svg = kg.edges_to_svg(edges, dark=is_dark)
    if svg:
        st.caption(":material/device_hub: Digital twin map")
        payload = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        st.markdown(
            f'<img alt="Digital twin graph" src="data:image/svg+xml;base64,{payload}" '
            f'style="display:block;width:100%;height:auto;border-radius:18px;margin:0 0 0.9rem;"/>',
            unsafe_allow_html=True,
        )


def _clean_answer(text: str) -> str:
    import re
    t = re.sub(r"<svg\b[\s\S]*?</svg>", "", text or "", flags=re.I)
    t = re.sub(r"```[\s\S]*?```", "", t)
    t = re.sub(r"</?div[^>]*>", "", t, flags=re.I)
    return t.strip()


def render_history():
    for msg in st.session_state.messages:
        avatar = ":material/person:" if msg["role"] == "user" else "⚗️"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["role"] == "user":
                st.markdown(msg["content"])
                continue
            if msg.get("trace"):
                icons = " · ".join(
                    f'{TOOL_ICONS.get(t["name"], ":material/build:")} `{t.get("agent", t["name"])}`'
                    for t in msg["trace"]
                )
                with st.status(f"Used {len(msg['trace'])} tool(s)  ·  {icons}", state="complete", type="compact"):
                    for step in msg["trace"]:
                        icon = TOOL_ICONS.get(step["name"], ":material/build:")
                        who = step.get("agent", "tool")
                        with st.status(f"{icon} {who} · {step['name']}", state="complete", type="step"):
                            st.json(step["result"])
            kg_edges = _kg_edges(msg.get("trace"))
            if kg_edges:
                _render_kg(kg_edges)
            if msg.get("conflict") and msg["conflict"].get("conflict"):
                st.markdown(
                    f'<div class="conflict-box">⚠️ CONFLICT — {msg["conflict"]["message"]}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(f'<div class="answer-card">{_clean_answer(msg["content"])}</div>', unsafe_allow_html=True)

# ── Chat input — + attach · mic · → send ─────────────────────────────────────
if st.session_state.pop("_clear_chat_input", False):
    st.session_state["main_input"] = None
prompt = None
submission = st.chat_input(
    "Type here…",
    accept_file=True,
    file_type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    key="main_input",
    submit_mode="disable",
)
spoken = _mount_voice_mic()

if "_voice_prompt" in st.session_state and st.session_state["_voice_prompt"]:
    prompt = st.session_state.pop("_voice_prompt")
elif spoken:
    prompt = spoken
elif submission:
    if getattr(submission, "audio", None):
        if not groq_key:
            st.error("Paste your Groq API key in the sidebar to use voice.")
            st.stop()
        try:
            from groq import Groq as _G
            _t = _G(api_key=groq_key).audio.transcriptions.create(
                file=("voice.wav", submission.audio.read()),
                model="whisper-large-v3-turbo",
                language="en",
            )
            st.session_state["_voice_prompt"] = _t.text
            st.rerun()
        except Exception as e:
            st.error(f"Voice transcription failed: {e}")
            st.stop()

    if getattr(submission, "files", None):
        f = submission.files[0]
        tmp = Path(tempfile.gettempdir()) / f"agent_sem{Path(f.name).suffix or '.png'}"
        tmp.write_bytes(f.getvalue())
        st.session_state.sem_path = str(tmp)

    typed_text = getattr(submission, "text", "") or ""
    has_file = bool(getattr(submission, "files", None))
    if not typed_text and not has_file:
        st.stop()
    prompt = typed_text or f"[Attached SEM: {submission.files[0].name}] Classify this SEM image and describe the fracture mode."

# Landing only when there is nothing to show yet
if not st.session_state.messages and not prompt:
    greeting = GREETING + (" 🌙" if hour >= 17 or hour < 5 else " ☀️")
    st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-title">{greeting}</div>
  <div class="hero-sub">
    Ask about fracture risk, SEM images, ISO/DIN standards, or motor winding loads.<br>
    Upload an SEM and pick a motor session in the sidebar for the full agentic workflow.
  </div>
</div>
""", unsafe_allow_html=True)

    picked = st.pills(
        "suggestions",
        options=list(SUGGESTIONS.keys()),
        selection_mode="single",
        label_visibility="collapsed",
    )
    if picked:
        prompt = SUGGESTIONS[picked]

if prompt:
    if not groq_key:
        st.error("Paste your Groq API key in the sidebar.")
        st.stop()
    user_msg = {"role": "user", "content": prompt}
    if submission and getattr(submission, "files", None):
        user_msg["has_image"] = True
    st.session_state.messages.append(user_msg)

if st.session_state.messages:
    render_history()

if prompt:
    with st.chat_message("assistant", avatar="⚗️"):
        with st.status(":shimmer[Searching textbooks…]", type="compact") as status:
            result = run_agent(
                question=prompt,
                groq_api_key=groq_key,
                image_path=st.session_state.sem_path,
                motor_profile_id=profile_id,
            )
            n = len(result["tool_trace"])
            status.update(
                label=f":material/check: Done · {n} tool call{'s' if n != 1 else ''}",
                state="complete",
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "trace": result["tool_trace"],
        "conflict": result.get("conflict"),
    })
    _after_prompt_sent()
    st.rerun()
