import os
import pyttsx3
import networkx as nx
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate

# ==========================================
# 1. INITIALIZE TEXT-TO-SPEECH (OFFLINE)
# ==========================================
engine = pyttsx3.init()
engine.setProperty('rate', 175) # Set reading speed (default is usually 200)

# ==========================================
# 2. SETUP AI MODELS (CRUCIAL FIX)
# ==========================================
# Use nomic for embeddings (Vectors) and llama3.2 for chatting (Text Generation)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2", temperature=0)

# ==========================================
# 3. LOAD AND CHUNK DOCUMENTS
# ==========================================
print("Loading documents into FAISS...")
# Assuming your markdown files are in a folder named 'data'
loader = DirectoryLoader('./data', glob="**/*.md", loader_cls=TextLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Build the Vector Database
vector_db = FAISS.from_documents(chunks, embeddings)
print(f"Stored {len(chunks)} chunks in Vector DB.\n")

# ==========================================
# 4. BUILD NETWORKX KNOWLEDGE GRAPH
# ==========================================
print("Building NetworkX Knowledge Graph...")
graph = nx.Graph()

# (Keep your specific graph entity extraction logic here if you have custom nodes/edges)
# For the prototype, we assume the graph is built with your system architecture rules.
graph.add_edge("order_cache_tier_1", "Order Service")
graph.add_edge("Order Service", "Lead Backend Engineer")

print(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.\n")

# ==========================================
# 5. DYNAMIC QUESTION INPUT
# ==========================================
# Instead of hardcoding, the script will now pause and wait for you to type a question
question = input("🤔 Ask a question about your system architecture: ")

# ==========================================
# 6. HYBRID RETRIEVAL (VECTOR + GRAPH)
# ==========================================
print("\n--- Vector Retrieval ---")
vector_results = vector_db.similarity_search(question, k=2)
vector_context = "\n".join([res.page_content for res in vector_results])
print(f"Retrieved {len(vector_results)} chunks.")

print("--- Graph Retrieval ---")
# Mock graph retrieval based on your earlier output logic
graph_context = "Dependency: order_cache_tier_1 -> Order Service -> Lead Backend Engineer"
print("Extracted graph relationships.\n")

# ==========================================
# 7. LLM GENERATION & TTS OUTPUT
# ==========================================
print("--- LLM Generation ---")

prompt_template = PromptTemplate.from_template(
    """You are a site reliability engineering AI. 
    Use the following Vector Context and Graph Context to answer the user's question accurately.
    Keep the answer concise and direct.
    
    Vector Context: {vector_context}
    Graph Context: {graph_context}
    
    Question: {question}
    
    Answer:"""
)

# Format the prompt with our retrieved data
formatted_prompt = prompt_template.format(
    vector_context=vector_context,
    graph_context=graph_context,
    question=question
)

# Generate the final answer
final_answer = llm.invoke(formatted_prompt)

print("\nFINAL HYBRID ANSWER:")
print(final_answer.content)
print("\n🔊 Speaking output...")

# Speak the answer out loud using Windows local TTS
engine.say(final_answer.content)
engine.runAndWait()