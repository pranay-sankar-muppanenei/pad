from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"

load_dotenv(BASE_DIR / ".env")


def load_documents() -> list[Document]:
    """Load every Markdown file from the local docs directory."""
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Document directory was not found: {DOCS_DIR}")

    documents: list[Document] = []
    for file_path in sorted(DOCS_DIR.glob("*.md")):
        documents.extend(TextLoader(str(file_path), encoding="utf-8").load())

    if not documents:
        raise FileNotFoundError(f"No Markdown documents were found in {DOCS_DIR}")

    return documents


@st.cache_resource(show_spinner="Indexing documents...")
def create_vectorstore() -> FAISS:
    """Create the searchable FAISS index from the local Markdown documents."""
    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)
    return FAISS.from_documents(chunks, OpenAIEmbeddings())


def answer_question(question: str, vectorstore: FAISS) -> tuple[str, list[str]]:
    """Retrieve relevant context and generate a grounded answer."""
    retrieved_documents = vectorstore.similarity_search(question, k=4)
    context = "\n\n---\n\n".join(
        document.page_content for document in retrieved_documents
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You answer questions using only the supplied document context. "
                "If the answer is not in the context, say you could not find it. "
                "Do not invent facts. Use clear headings when useful.\n\n"
                "Document context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )
    response = (prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0)).invoke(
        {"context": context, "question": question}
    )

    sources = sorted(
        {
            Path(document.metadata.get("source", "Unknown source")).name
            for document in retrieved_documents
        }
    )
    return response.content, sources


st.set_page_config(
    page_title="Personal Knowledge Assistant",
    page_icon="📚",
    layout="centered",
)

st.title("Personal Knowledge Assistant")
st.caption("Ask questions about the Markdown documents in the local docs folder.")

if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "OPENAI_API_KEY is not configured. Copy .env.example to .env and add your key."
    )
    st.stop()

with st.sidebar:
    st.header("About")
    st.write(
        "This app retrieves relevant document chunks with FAISS and uses an "
        "OpenAI model to generate a grounded answer."
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Sources: " + ", ".join(message["sources"]))

question = st.chat_input("Ask something about your documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    vectorstore = create_vectorstore()
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            answer, sources = answer_question(question, vectorstore)
        st.markdown(answer)
        st.caption("Sources: " + ", ".join(sources))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
