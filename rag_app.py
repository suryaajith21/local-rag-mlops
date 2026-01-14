from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configuration
DB_PATH = "vector_db"
MODEL_NAME = "llama3.2" # Make sure this matches what you pulled in Ollama

def main():
    # 1. Load the Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    # 2. Create Retriever
    retriever = db.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks

    # 3. Setup Local LLM (Ollama)
    llm = ChatOllama(model=MODEL_NAME)

    # 4. Define the Prompt
    template = """Answer the question based ONLY on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 5. Build the Chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. Interactive Loop
    print(f"--- Chatting with your docs using {MODEL_NAME} (Type 'quit' to stop) ---")
    while True:
        query = input("\nAsk a question: ")
        if query.lower() == "quit":
            break
        
        # Invoke the chain
        print("\nThinking...")
        response = chain.invoke(query)
        print(f"\nAnswer: {response}")

if __name__ == "__main__":
    main()