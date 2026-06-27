import networkx as nx
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from config import EMBEDDINGS

def build_vector_db(data_path='./data'):
    print("Loading documents into FAISS...")
    loader = DirectoryLoader(data_path, glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    vector_db = FAISS.from_documents(chunks, EMBEDDINGS)
    print(f"Stored {len(chunks)} chunks in Vector DB.\n")
    return vector_db

def build_knowledge_graph():
    print("Building NetworkX Knowledge Graph...")
    graph = nx.Graph()
    
    # Hardcoded logic for the prototype
    graph.add_edge("order_cache_tier_1", "Order Service")
    graph.add_edge("Order Service", "Lead Backend Engineer")
    
    print(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.\n")
    return graph