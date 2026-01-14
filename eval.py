import json
import pandas as pd
from datasets import Dataset 
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# STABLE RAGAS IMPORTS
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Configuration
DB_PATH = "vector_db"
MODEL_NAME = "llama3.2" 

def get_rag_chain():
    # 1. Setup Embeddings & DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    # 2. Retriever
    retriever = db.as_retriever(search_kwargs={"k": 2})
    
    # 3. Local LLM (Ollama)
    llm = ChatOllama(model=MODEL_NAME, temperature=0)

    # 4. Chain
    template = """Answer the question based ONLY on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever, llm, embeddings

def run_evaluation():
    print("--- Starting Evaluation Pipeline ---")
    
    # 1. Load Test Data
    try:
        with open("test_dataset.json", "r") as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return
    
    questions = [item["question"] for item in test_data]
    
    # --- FIX: Flatten the list. Ragas wants Strings, not Lists of Strings ---
    ground_truths = [item["ground_truth"] for item in test_data] 

    # 2. Generate Answers
    chain, retriever, app_llm, app_embeddings = get_rag_chain()
    
    generated_answers = []
    contexts = []

    print("--- Generating Answers for Test Set ---")
    for q in questions:
        print(f"Processing: {q}...")
        
        # Get Contexts (Required for Ragas)
        retrieved_docs = retriever.invoke(q)
        doc_texts = [doc.page_content for doc in retrieved_docs]
        contexts.append(doc_texts)
        
        # Get Answer
        response = chain.invoke(q) 
        generated_answers.append(response)

    # 3. Prepare Data for Ragas
    data_samples = {
        "question": questions,
        "answer": generated_answers,
        "contexts": contexts,
        "ground_truth": ground_truths # Now a simple list of strings
    }
    dataset = Dataset.from_dict(data_samples)

    # 4. Configure Ragas with Local LLM
    print("--- Configuring Ragas to use Local Ollama ---")
    
    ragas_llm = LangchainLLMWrapper(app_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(app_embeddings)

    # 5. Run Evaluation
    print("--- Grading Answers (This uses Ollama as a Judge) ---")
    
    # We pass raise_exceptions=False so one bad math error doesn't crash the whole run
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm, 
        embeddings=ragas_embeddings,
        raise_exceptions=False 
    )

    print("\n--- Evaluation Results ---")
    print(results)
    
    # Save results
    df = results.to_pandas()
    df.to_csv("evaluation_results.csv", index=False)
    print("Results saved to evaluation_results.csv")

if __name__ == "__main__":
    run_evaluation()