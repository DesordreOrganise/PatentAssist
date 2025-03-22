from abc import ABC, abstractmethod
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import networkx as nx
from os import PathLike
import yaml

class BaseSystem(ABC):

    @abstractmethod
    def run(input:str) -> str:
        pass


class Retriever():

    def __init__(self, config_path: PathLike, articles: list, embeddings: OllamaEmbeddings=OllamaEmbeddings(model="all-minilm")):
        self.local_embeddings = embeddings
        self.vectorstore = Chroma.from_documents(documents=articles, embedding=self.local_embeddings)
        self.config = self._load_config(config_path)


    def _load_config(config_path: PathLike) -> dict:
        with open(config_path, 'r') as ys:
            config = yaml.safe_load(ys)
        return config
    

    def retrieve_documents(self, query: str):
        documents = self.vectorstore.similarity_search(query, k=self.config["k"])
        return documents


class RAG(BaseSystem):

    def __init__(self, LLM, retriever, database: nx.MultiDiGraph, st_memory, lt_memory):
        self.LLM = LLM
        self.retriever = retriever
        self.database = database
        self.st_memory = st_memory
        self.lt_memory = lt_memory


    def run(self, input: str) -> str:
        #documents = retriever.retrieve()
        context = self._format_context(input, documents)
        #documents.get_text()
        
        return self.LLM(context)
    
    
    def _format_context(self, input: str, documents):
        #ici formatter le contexte selon un template?
        pass