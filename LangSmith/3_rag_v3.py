# pip install -U langchain langchain-groq langchain-community langchain-huggingface faiss-cpu pypdf python-dotenv langsmith langchain-text-splitters
import os
from dotenv import load_dotenv
from langsmith import traceable
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "pdf_rag_demo"

PDF_PATH = "islr.pdf"
FAISS_INDEX_PATH = "islr_faiss_index"

emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ----------------- helpers -----------------
@traceable(name="load_pdf")
def load_pdf(path: str):
    return PyPDFLoader(path).load()

@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(docs)

@traceable(name="build_vectorstore")
def build_vectorstore(splits):
    vs = FAISS.from_documents(splits, emb)
    vs.save_local(FAISS_INDEX_PATH)
    return vs

# ----------------- parent setup (traced) -----------------
@traceable(name="setup_pipeline", tags=["setup"])
def setup_pipeline(pdf_path: str, chunk_size=1000, chunk_overlap=150):
    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading FAISS index from cache...")
        return FAISS.load_local(FAISS_INDEX_PATH, emb, allow_dangerous_deserialization=True)
    print("Building index for the first time (this will take a while)...")
    docs = load_pdf(pdf_path)
    splits = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return build_vectorstore(splits)

# ----------------- model + prompt -----------------
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# ----------------- root traced run -----------------
@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_query(pdf_path: str, question: str):
    vectorstore = setup_pipeline(pdf_path, chunk_size=1000, chunk_overlap=150)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })
    chain = parallel | prompt | llm | StrOutputParser()
    return chain.invoke(question, config={"run_name": "pdf_rag_query"})

# ----------------- CLI -----------------
if __name__ == "__main__":
    print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
    q = input("\nQ: ").strip()
    ans = setup_pipeline_and_query(PDF_PATH, q)
    print("\nA:", ans)