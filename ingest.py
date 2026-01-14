import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
DATA_PATH = "data/"
DB_PATH = "vector_db"

def create_vector_db():
    print(f"--- Loading PDF from {DATA_PATH} ---")
    
    # 1. Load PDF(s)
    documents = []
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            if file.endswith(".pdf"):
                print(f"Found file: {file}")
                loader = PyPDFLoader(os.path.join(root, file))
                documents.extend(loader.load())

    if not documents:
        print("No PDFs found in 'data/' folder!")
        return

    # 2. Split Text
    # Local models have smaller context windows, so we keep chunks smaller (500)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    # 3. Create Embeddings (The Free Part)
    # This downloads a small, high-performance model from HuggingFace
    print("--- Creating Embeddings (this may take a moment) ---")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Store in Vector DB
    print("--- Inserting into Vector DB ---")
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    print(f"--- Finished! Vector DB saved to '{DB_PATH}' ---")

if __name__ == "__main__":
    create_vector_db()