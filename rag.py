from dotenv import load_dotenv
load_dotenv()
import os
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser


# ---------------------------
# 1. TRANSCRIPT LOADING
# ---------------------------
def get_transcript(video_id: str) -> str:
    transcript_list = YouTubeTranscriptApi().fetch(video_id)
    text = " ".join([t.text for t in transcript_list])
    return text


# 2. VECTOR STORE CREATION
def create_vector_store(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([text])

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store


# ---------------------------
# 3. PROMPT
# ---------------------------
def get_prompt():
    return PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, say you don't know.

{context}

Question: {question}
""",
        input_variables=["context", "question"]
    )


# ---------------------------
# 4. RETRIEVAL FORMATTER
# ---------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------------------
# 5. BUILD RAG CHAIN
# ---------------------------
def build_chain(vector_store):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    prompt = get_prompt()
    parser = StrOutputParser()

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | llm | parser

    return main_chain


# ---------------------------
# 6. MAIN QA FUNCTION
# ---------------------------
def ask_question(chain, question: str):
    return chain.invoke(question)