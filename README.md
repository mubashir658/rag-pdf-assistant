# 📚 RAG PDF Assistant

An AI-powered PDF Question Answering application built using **Streamlit, LangChain, Gemini, ChromaDB, and Sentence Transformers**. The application enables users to upload PDF documents and interact with them through natural language conversations using Retrieval-Augmented Generation (RAG).

## 🚀 Live Demo

👉 https://rag-pdf-assistant-ali.streamlit.app/

---

## 📌 Overview

RAG PDF Assistant allows users to upload PDF documents and ask questions about their content.

Instead of relying solely on the language model's knowledge, the application retrieves relevant information directly from the uploaded document and generates context-aware answers grounded in the source material.

The system also displays the source passages and page numbers used to generate each response, improving transparency and trustworthiness.

---

## ✨ Features

* Upload and process PDF documents
* Automatic document chunking
* Semantic search using vector embeddings
* Retrieval-Augmented Generation (RAG)
* Context-aware answers powered by Gemini
* Source citations with page references
* Interactive chat interface
* Persistent vector storage using ChromaDB
* Real-time PDF question answering

---

## 🏗️ System Architecture

PDF Upload
↓
Document Loading (PyPDF)
↓
Text Chunking
↓
Sentence Embeddings
↓
ChromaDB Vector Store
↓
Semantic Retrieval
↓
Gemini LLM
↓
Answer Generation + Source References

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### LLM

* Gemini 2.5 Flash

### Frameworks

* LangChain

### Vector Database

* ChromaDB

### Embedding Model

* all-MiniLM-L6-v2 (Sentence Transformers)

### Document Processing

* PyPDFLoader

### Environment Management

* Python Dotenv

---

## ⚙️ How It Works

### 1. Document Processing

* PDF documents are uploaded by the user.
* The document is loaded using PyPDFLoader.
* Text is split into smaller chunks using RecursiveCharacterTextSplitter.

### 2. Embedding Generation

* Each chunk is converted into vector embeddings using Sentence Transformers.
* Embeddings capture semantic meaning rather than exact keywords.

### 3. Vector Storage

* Embeddings are stored in ChromaDB.
* Each chunk is associated with metadata such as page numbers.

### 4. Retrieval

* User queries are converted into embeddings.
* ChromaDB performs similarity search to identify the most relevant document chunks.

### 5. Response Generation

* Retrieved context is provided to Gemini.
* Gemini generates answers based only on the retrieved document content.
* Source references and page numbers are returned alongside the response.

---

## 📂 Project Structure

```text
rag-pdf-assistant/
│
├── app.py
├── requirements.txt
├── .gitignore
├── runtime.txt
└── README.md
```

## 🎯 Key Concepts Demonstrated

* Retrieval-Augmented Generation (RAG)
* Vector Databases
* Semantic Search
* Embeddings
* Document Question Answering
* LangChain Pipelines
* LLM Application Development
* Streamlit Deployment
* Prompt Engineering

---

## 🔮 Future Enhancements

* Conversational memory
* Multi-document support
* Hybrid search (keyword + semantic)
* Chat history persistence
* Document summarization
* Support for DOCX and TXT files
* User authentication

---

## 👨‍💻 Author

**Mohammed Mubashir Ali**

Aspiring AI Engineer passionate about Generative AI, RAG Systems, Agentic AI, and Production AI Applications.

LinkedIn:www.linkedin.com/in/mohammed-mubashir-ali-hyd658
GitHub: https://github.com/mubashir658

---

## ⭐ Why This Project Matters

This project demonstrates the practical implementation of Retrieval-Augmented Generation (RAG), a core architecture used in modern AI applications such as enterprise knowledge assistants, document intelligence systems, customer support copilots, and internal search platforms.
