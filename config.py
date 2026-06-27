from langchain_ollama import OllamaEmbeddings, ChatOllama

# Setup AI Models
EMBEDDINGS = OllamaEmbeddings(model="nomic-embed-text")
LLM = ChatOllama(model="llama3.2", temperature=0)