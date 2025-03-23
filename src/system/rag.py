from abc import ABC, abstractmethod
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

import networkx as nx
from typing import Tuple
from os import PathLike
import yaml


class BaseSystem(ABC):

    @abstractmethod
    def run(input: str) -> str:
        pass


class Reranker():
    
    def __init__(self, model_name: str, train_mode: bool=False):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if not train_mode:
            self.model.eval()

    def rerank(self, query: str, docs: list[Document], k: int=3) -> list[Document]:
        pairs = [(query, doc.page_content) for doc in docs]

        inputs = self.tokenizer.batch_encode_plus(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze()

        if scores.ndim > 1 and scores.shape[1] > 1:
            scores = F.softmax(scores, dim=1)[:, 1]

        reranked = sorted(zip(docs, scores.tolist()), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in reranked[:k]]


class Retriever():

    def __init__(self, config_path: PathLike,
                articles: dict,
                embeddings: OllamaEmbeddings=OllamaEmbeddings(model="all-minilm"),
                reranker_model: str="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        
        self.local_embeddings = embeddings
        self.vectorstore = Chroma.from_texts(texts=articles["texts"], embedding=self.local_embeddings, ids=articles["ids"], metadatas=articles["metadatas"])
        self.config = self._load_config(config_path)
        self.reranker = Reranker(reranker_model)


    def _load_config(self, config_path: PathLike) -> dict:
        with open(config_path, 'r') as ys:
            config = yaml.safe_load(ys)
        return config
    

    def retrieve_documents(self, query: str, rerank: bool=True):
        documents = self.vectorstore.similarity_search(query, k=self.config["initial_k"])
        if rerank:
            documents = self.reranker.rerank(query, documents, k=self.config["rerank_k"])
        return documents


class RAG(BaseSystem):

    def __init__(self, LLM: ChatOllama, retriever: Retriever, database: nx.MultiDiGraph, lt_memory):
        self.LLM = LLM
        self.retriever = retriever
        self.database = database
        self.system_prompt = SystemMessage(content="You are a helpful assistant that answers based only on the provided context. If the answer is not in the context, say 'I don't know.'")
        self.st_memory = [self.system_prompt]
        self.lt_memory = lt_memory


    def run(self, input: str, rerank: bool=True) -> str:
        documents = self.retriever.retrieve_documents(input, rerank=rerank)
        context = self._format_context(input, documents)
        self.st_memory.append(input)
        response = self.LLM.invoke(self.st_memory)
        self.st_memory.append(response)
        return self.LLM(context)
    
    
    def _format_context(self, input: str, documents: list[Document]):
        #ici formatter le contexte selon un template? CoT
        pass