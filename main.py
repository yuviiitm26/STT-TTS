from config import LLM
from ingestion import build_vector_db, build_knowledge_graph
from audio import listen_for_question, speak_text
from langchain_core.prompts import PromptTemplate

def main():
    # 1. Build the Databases
    vector_db = build_vector_db()
    graph = build_knowledge_graph()

    # 2. Get User Input via Voice
    question = listen_for_question()
    print(f"\n✅ You asked: '{question}'")

    # 3. Retrieve Context
    print("\n--- Retrieving Context ---")
    vector_results = vector_db.similarity_search(question, k=2)
    vector_context = "\n".join([res.page_content for res in vector_results])
    
    graph_context = "Dependency: order_cache_tier_1 -> Order Service -> Lead Backend Engineer"

    # 4. Generate Answer
    print("\n--- LLM Generation ---")
    prompt_template = PromptTemplate.from_template(
        """You are a site reliability engineering AI. 
        Use the following Vector Context and Graph Context to answer the user's question accurately.
        Keep the answer concise and direct.
        
        Vector Context: {vector_context}
        Graph Context: {graph_context}
        
        Question: {question}
        
        Answer:"""
    )
    
    formatted_prompt = prompt_template.format(
        vector_context=vector_context,
        graph_context=graph_context,
        question=question
    )

    final_answer = LLM.invoke(formatted_prompt)

    # 5. Output Result
    print("\nFINAL HYBRID ANSWER:")
    print(final_answer.content)
    
    speak_text(final_answer.content)

if __name__ == "__main__":
    main()