import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import os
import tempfile
import re

load_dotenv()

# initialize models
@st.cache_resource
def init_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    chroma_client = chromadb.PersistentClient(path="./rag_db")
    return embedding_model, llm, chroma_client

embedding_model, llm, chroma_client = init_models()

def process_pdf(uploaded_file, collection_name):
    # save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # load and chunk
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # embed and store
    collection = chroma_client.get_or_create_collection(collection_name)

    if collection.count() == 0:
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedding_model.encode(texts).tolist()
        metadatas = [{"page": chunk.metadata.get("page", 0)} for chunk in chunks]

        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=[f"chunk_{i}" for i in range(len(texts))]
        )

    # cleanup temp file
    os.unlink(tmp_path)
    return collection, len(chunks)

def retrieve_and_answer(query, collection):
    # retrieve
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    context_chunks = results["documents"][0]
    pages = [m.get("page", "?") for m in results["metadatas"][0]]

    # build context
    context = "\n\n".join([
        f"[Page {pages[i]}]: {chunk}"
        for i, chunk in enumerate(context_chunks)
    ])

    # generate answer
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions about documents.
Answer using ONLY the context below.
If the answer is not in the context, say "I couldn't find that in the document."
Always mention which page the information came from.

Context:
{context}"""),
        ("human", "{question}")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": query
    })

    return response.content, context_chunks, pages
# ── UI ──────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📚")
st.title("📚 RAG PDF Chatbot")
st.caption("Upload a PDF and ask questions about it")

# initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# sidebar — PDF upload
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file:
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("Processing PDF..."):
                # use filename as collection name (cleaned)
                collection_name = uploaded_file.name.replace(".pdf", "").lower()

                collection_name = re.sub(r"[^a-z0-9._-]", "_", collection_name)

                collection_name = collection_name[:50]
                
                collection, num_chunks = process_pdf(
                    uploaded_file, collection_name)
                
                st.session_state.collection = collection
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []
                
            st.success(f"✅ Processed {num_chunks} chunks")
            st.info(f"📄 {uploaded_file.name}")

    if st.session_state.pdf_name:
        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# main chat area
if not st.session_state.collection:
    st.info("👈 Upload a PDF from the sidebar to get started")
else:
    # display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📖 View Sources"):
                    for i, (chunk, page) in enumerate(
                            zip(msg["sources"], msg["pages"])):
                        st.markdown(f"**Source {i+1} (Page {page}):**")
                        st.markdown(f"> {chunk[:300]}...")
                        st.divider()

    # chat input
    if prompt := st.chat_input("Ask anything about your PDF..."):
        # show user message
        st.chat_message("user").write(prompt)
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })

        # generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources, pages = retrieve_and_answer(
                    prompt, st.session_state.collection)
            
            st.write(answer)

            with st.expander("📖 View Sources"):
                for i, (chunk, page) in enumerate(zip(sources, pages)):
                    st.markdown(f"**Source {i+1} (Page {page}):**")
                    st.markdown(f"> {chunk[:300]}...")
                    st.divider()

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "pages": pages
        })