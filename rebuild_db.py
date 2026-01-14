import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration
PDF_FILE = "data/genai_review.pdf" # Ensure this matches your PDF name exactly
DB_PATH = "vector_db"

def rebuild():
    print(f"--- Rebuilding Vector DB from {PDF_FILE} ---")
    
    # 1. Load PDF
    if not os.path.exists(PDF_FILE):
        print(f"Error: File {PDF_FILE} not found!")
        return

    loader = PyPDFLoader(PDF_FILE)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages.")

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")

    # 3. Embed & Save to Disk
    # using the same model as your eval.py to ensure they match
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("Embedding data... (this might take a moment)")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    
    print("--- Success! Database rebuilt at 'vector_db' ---")

if __name__ == "__main__":
    rebuild()