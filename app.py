import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PDF_PATH = PROJECT_ROOT / "attention.pdf"
INDEX_PATH = PROJECT_ROOT / "faiss_index"


def read_pdf_text(pdf_path: str | Path) -> str:
    reader = PdfReader(str(pdf_path))
    text_chunks = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            text_chunks.append(page_text)

    return "\n".join(text_chunks)


def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_text(text)


def create_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vector_store(chunks, index_path: str | Path):
    embeddings = create_embeddings()
    documents = [Document(page_content=chunk) for chunk in chunks]
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(str(index_path))
    return vector_store


def load_vector_store(index_path: str | Path):
    embeddings = create_embeddings()
    return FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)


def retrieve_chunks(question: str, index_path: str | Path, k: int = 10):
    vector_store = load_vector_store(index_path)
    return vector_store.similarity_search(question, k=k)


def get_prompt_and_llm():
    prompt_template = """
You are an AI assistant.
Answer the question using only the context below.
If the answer is not present in the context, say exactly:
"THE ANSWER IS NOT AVAILABLE IN THE PROVIDED CONTEXT."
The answer has to be in bullet point wise, each point not exceeding 200 words.
The answer has to be in a way that a CLASS 10TH GRADE STUDENT UNDERSTANDS IT.

Context:
{context}
Question:
{question}

Answer:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=1.9)
    return prompt, llm


def answer_question(question: str, index_path: str | Path):
    docs = retrieve_chunks(question, index_path)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt, llm = get_prompt_and_llm()
    final_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(final_prompt)
    return response.content


def ensure_index(pdf_path: str | Path, index_path: str | Path):
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not (Path(index_path) / "index.faiss").exists():
        text = read_pdf_text(pdf_path)
        chunks = split_text(text)
        build_vector_store(chunks, index_path)


st.set_page_config(page_title="Chat with PDF", page_icon="📄")

st.title("📄 Chat with PDF")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.warning("Add GOOGLE_API_KEY to your .env file before running the app.")
    st.stop()

pdf_path = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
selected_pdf = Path(pdf_path.name) if pdf_path is not None else DEFAULT_PDF_PATH

if pdf_path is not None:
    uploaded_pdf = PROJECT_ROOT / pdf_path.name
    uploaded_pdf.write_bytes(pdf_path.read())
    selected_pdf = uploaded_pdf

try:
    ensure_index(selected_pdf, INDEX_PATH)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

question = st.text_input("Ask a question about the PDF")

if st.button("Ask") and question.strip():
    try:
        answer = answer_question(question, INDEX_PATH)
        st.markdown("### Answer")
        st.write(answer)
    except Exception as exc:
        st.error(f"Something went wrong: {exc}")

if not question.strip():
    st.info("Upload a PDF or use the default one and ask a question.")
