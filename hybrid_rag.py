import networkx as nx
import pyttsx3
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
# Initialize the offline voice engine
engine = pyttsx3.init()

# 1. Setup Models (Ensure Ollama is running locally)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2", temperature=0)

# 2. Load Documents into Vector DB
print("Loading documents into FAISS...")
loader = DirectoryLoader('./data', glob="**/*.md", loader_cls=TextLoader)
docs = loader.load()

# Split docs into smaller semantic chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

vector_db = FAISS.from_documents(chunks, embeddings)
print(f"Stored {len(chunks)} chunks in Vector DB.")

# 3. Build the NetworkX Graph (Mocked for Prototype 1)
print("\nBuilding NetworkX Knowledge Graph...")
G = nx.DiGraph()

# Hardcoded triples extracted from our 3 markdown files
triples = [
    ("Kong API Gateway", "routes_to", "Auth Service"),
    ("Auth Service", "depends_on", "users_db (PostgreSQL)"),
    ("Catalog Service", "reads_from", "MongoDB Cluster"),
    ("Order Service", "depends_on", "Catalog Service"),
    ("Order Service", "depends_on", "Inventory Service"),
    ("Order Service", "uses_cache", "order_cache_tier_1"),
    
    ("Squad Alpha", "maintains", "Kong API Gateway"),
    ("Sarah Jenkins", "leads", "Squad Alpha"),
    
    ("Squad Beta", "maintains", "Auth Service"),
    ("Squad Beta", "maintains", "Order Service"),
    ("David Chen", "leads", "Squad Beta"),
    ("Marcus Thorne", "lead_engineer_for", "Order Service"),
    
    ("Squad Gamma", "maintains", "Catalog Service"),
    ("Elena Rostova", "leads", "Squad Gamma")
]

for subject, relation, obj in triples:
    G.add_edge(subject, obj, relation=relation)
print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# 4. The Hybrid Search Function
def hybrid_search(question, seed_entities):
    # A. Vector Search (Find semantically relevant text)
    print("\n--- Vector Retrieval ---")
    vector_results = vector_db.similarity_search(question, k=2)
    vector_context = "\n".join([doc.page_content for doc in vector_results])
    print(f"Retrieved {len(vector_results)} chunks.")

    # B. Graph Traversal (Find relational context)
    print("--- Graph Retrieval ---")
    graph_context = []
    for entity in seed_entities:
        if entity in G:
            # Extract 1-hop neighborhood for the entity
            neighbors = list(G.successors(entity)) + list(G.predecessors(entity))
            for n in neighbors:
                if G.has_edge(entity, n):
                    rel = G[entity][n]['relation']
                    graph_context.append(f"{entity} [{rel}] {n}")
                if G.has_edge(n, entity):
                    rel = G[n][entity]['relation']
                    graph_context.append(f"{n} [{rel}] {entity}")
                    
    graph_text = "\n".join(list(set(graph_context)))
    print(f"Extracted {len(list(set(graph_context)))} graph relationships.")

    # C. Context Fusion (Combine both for the LLM)
    prompt = f"""
    Answer the question based ONLY on the context below.
    
    VECTOR CONTEXT (Document Text):
    {vector_context}
    
    GRAPH CONTEXT (Entity Relationships):
    {graph_text}
    
    Question: {question}
    Answer:
    """
    
    print("\n--- LLM Generation ---")
    response = llm.invoke(prompt)
    return response.content


# 5. Run the "RAG Trap" Test
question = "If the order_cache_tier_1 Redis cluster goes down, who specifically should be paged?"
# In the next phase of the project, we will use the LLM to dynamically extract "order_cache_tier_1" from the question string.
seed_entities = ["order_cache_tier_1", "Order Service"] 

print(f"\nQUESTION: {question}")
answer = hybrid_search(question, seed_entities)
print(f"\nFINAL HYBRID ANSWER:\n{answer}")