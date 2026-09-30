import streamlit as st
from src.assistant import CrimeIntelligenceAssistant

st.set_page_config(page_title="Crime Intelligence Assistant", page_icon="🛡️")
st.title("🛡️ Crime Intelligence Assistant")
st.caption("Grounded answers from the existing crime warehouse, analytical RAG index, and forecasting pipeline.")

@st.cache_resource
def get_assistant(): return CrimeIntelligenceAssistant()

if "messages" not in st.session_state: st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.caption(f"Route used: {message['route']}")
            if message["evidence"]:
                with st.expander("Evidence used"):
                    for item in message["evidence"]: st.write(f"**{item['source']}** — {item['text']}")

if question := st.chat_input("Ask about crime patterns, counts, or forecasts"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"): st.markdown(question)
    with st.chat_message("assistant"):
        response = get_assistant().ask(question)
        st.markdown(response.answer)
        st.caption(f"Route used: {response.route}")
        if response.evidence:
            with st.expander("Evidence used"):
                for item in response.evidence: st.write(f"**{item['source']}** — {item['text']}")
    st.session_state.messages.append({"role": "assistant", "content": response.answer, "route": response.route, "evidence": response.evidence})
