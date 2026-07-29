import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

st.title("🤖 Mood-Based AI Chatbot")
st.caption("Type 0 to exit and view the full message history.")

model = ChatMistralAI(model="mistral-small-2506", temperature=0.9)

PERSONALITIES = {
    "Funny": "You are a funny AI agent, crack jokes and have good humour.",
    "Angry": "You are an angry AI agent, talk angrily and act like you are frustrated.",
    "Sad": "You are a sad AI agent, act depressed.",
}

# Let the user pick a personality before the chat starts.
if "personality" not in st.session_state:
    st.session_state.personality = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "ended" not in st.session_state:
    st.session_state.ended = False

if st.session_state.personality is None:
    st.subheader("Choose your chatbot's mood")
    choice = st.radio(
        "Personality",
        list(PERSONALITIES.keys()),
        horizontal=True,
        label_visibility="collapsed",
    )
    if st.button("Start chatting"):
        st.session_state.personality = choice
        st.session_state.messages = [
            SystemMessage(content=PERSONALITIES[choice])
        ]
        st.rerun()
    st.stop()

st.caption(f"Current mood: **{st.session_state.personality}**")

# Display chat history (skip system message)
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

if not st.session_state.ended:
    prompt = st.chat_input("You :")

    if prompt:
        st.session_state.messages.append(HumanMessage(content=prompt))

        if prompt == "0":
            # Same as the original CLI: "0" ends the session
            st.session_state.ended = True
            st.rerun()

        with st.chat_message("user"):
            st.markdown(prompt)

        response = model.invoke(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response.content))

        with st.chat_message("assistant"):
            st.markdown(response.content)

if st.session_state.ended:
    st.info("Session ended (you typed 0). Here's the full raw message history:")

    # Convert LangChain message objects to plain dicts to avoid st.write's
    # internal pandas dataframe-detection check (can trigger DLL/policy issues
    # on some locked-down Windows machines).
    history = [
        {"type": msg.__class__.__name__, "content": msg.content}
        for msg in st.session_state.messages
    ]
    st.json(history)