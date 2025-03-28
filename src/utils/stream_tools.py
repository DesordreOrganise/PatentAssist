from src.system.rag import RAG, Retriever
from langchain_ollama import OllamaEmbeddings, ChatOllama
import pickle
import networkx as nx


def default_rag(model="gemma3:4b", embedder="nomic-embed-text") -> tuple[RAG, nx.MultiGraph]:
    with open("./resources/LegalBases/graph.pkl", "rb") as f:
        g = pickle.load(f)

    texts = []
    metadatas = []
    ids = []
    for id, doc in g.nodes(data=True):
        if doc["data"].page_content:
            ids.append(id)
            metadatas.append({"label": doc["label"]})
            texts.append(doc["data"].page_content)
    articles = {}
    articles["texts"] = texts
    articles["ids"] = ids
    articles["metadatas"] = metadatas

    print("Loading retriever")
    config_path = "./config/retriever_config.yaml"
    local_embeddings = OllamaEmbeddings(model=embedder)
    retriever = Retriever(config_path, articles, local_embeddings)

    print("Loading Rag")
    llm = ChatOllama(model=model)
    rag_config_path = "./config/config.yaml"
    rag = RAG(rag_config_path, llm, retriever, g, None)
    print("Done")

    return rag, g


if __name__ == "__main__":
    rag, g = default_rag()

    print(rag)
    print(g)
