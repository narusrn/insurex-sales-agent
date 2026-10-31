"""Streamlit chat UI — calls agents.runner directly, no HTTP hop, so this works
standalone on Streamlit Cloud without api/ running anywhere. Same split as
BBLAgent's own streamlit_app.py: the API is for programmatic/SSE clients, this
is for a human to click through.

Deploying on Streamlit Cloud: set OPENAI_API_KEY (or the Azure vars) and
API_KEY in the app's Secrets — no .env file gets uploaded there, so st.secrets
is mirrored into os.environ below, which is what every other module already
reads from.
"""
import asyncio
import os
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # local dev: .env. On Streamlit Cloud there is no .env file, so this is a no-op —
if hasattr(st, "secrets"):  # st.secrets (set via Cloud's Secrets UI) is what's actually used there.
    try:
        for k, v in st.secrets.items():
            os.environ.setdefault(k, str(v))
    except Exception:
        pass  # no secrets.toml locally — fine, load_dotenv() above already covered it

from agents.runner import stream_structured_events  # noqa: E402 (after the env setup above, on purpose)

st.set_page_config(page_title="InsureX Sales Agent", page_icon="💬")

with st.sidebar:
    st.header("InsureX Sales Agent")
    api_key_input = st.text_input("API Key", type="password")
    st.caption("ใส่ API key เพื่อเริ่มใช้งาน — ใช้ตรวจสอบสิทธิ์เท่านั้น ไม่ถูกส่งออกไปที่ไหน")
    if st.button("Reset conversation"):
        st.session_state.clear()
        st.rerun()

expected_key = os.environ.get("API_KEY")
if not expected_key:
    st.error("Server misconfigured: API_KEY is not set (add it to .env locally, or Secrets on Streamlit Cloud).")
    st.stop()
if api_key_input != expected_key:
    st.info("กรอก API key ที่แถบด้านซ้ายเพื่อเริ่มแชทค่ะ")
    st.stop()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def _run_turn(session_id: str, message: str, status):
    """Drains the async event stream synchronously — Streamlit's own script
    execution model is sync top-to-bottom, so this pumps the async generator by
    hand (new event loop, manual __anext__()) instead of needing asyncio.run()
    to own the whole call, which would prevent yielding incrementally for
    st.write_stream below."""
    loop = asyncio.new_event_loop()
    agen = stream_structured_events(session_id, message)
    try:
        while True:
            try:
                event = loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                break
            if event["type"] == "token":
                yield event["content"]
            elif event["type"] == "status":
                status.update(label=f"กำลังเรียกใช้ {event['tool']}...")
    finally:
        loop.close()


if prompt := st.chat_input("พิมพ์ข้อความถึง InsureX Sales Agent..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # status is created but NOT entered via `with` — st.write_stream below must render as a
        # sibling in this chat bubble, not get nested inside the status box, or the reply text
        # disappears from view the moment the status box collapses on completion (confirmed by
        # screenshotting the rendered page: the collapsed "เสร็จแล้ว" status hid the whole reply).
        status = st.status("กำลังประมวลผล...", expanded=False)
        reply = st.write_stream(_run_turn(st.session_state.session_id, prompt, status))
        status.update(label="เสร็จแล้ว", state="complete")
    st.session_state.messages.append({"role": "assistant", "content": reply})
