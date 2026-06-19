import streamlit as st
from rag import get_transcript, create_vector_store, build_chain, ask_question

st.set_page_config(page_title="YouTube RAG Chatbot")

st.title("🎥 YouTube RAG Chatbot")

# ---------------------------
# SESSION STATE INIT
# ---------------------------
if "chain" not in st.session_state:
    st.session_state.chain = None

if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False


# ---------------------------
# INPUT
# ---------------------------
video_url = st.text_input("Enter YouTube Video URL")
question = st.text_input("Ask a question")


# ---------------------------
# EXTRACT VIDEO ID
# ---------------------------
def extract_video_id(url):
    return url.split("v=")[-1]


# ---------------------------
# PROCESS VIDEO
# ---------------------------
if st.button("Process Video"):
    if video_url:
        with st.spinner("Processing video..."):

            video_id = extract_video_id(video_url)

            text = get_transcript(video_id)
            vector_store = create_vector_store(text)

            chain = build_chain(vector_store)

            st.session_state.chain = chain
            st.session_state.video_loaded = True

        st.success("Video processed successfully!")


# ---------------------------
# ASK QUESTION
# ---------------------------
if st.button("Ask Question"):
    if st.session_state.chain and question:
        answer = ask_question(st.session_state.chain, question)

        st.subheader("💡 Answer")
        st.write(answer)

    else:
        st.warning("Please process a video first.")