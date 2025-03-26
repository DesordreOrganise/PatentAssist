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
from typing import Tuple, Optional, Generator
import shutil
import os
from os import PathLike
from pathlib import Path

from src.utils.tools import load_config


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CHROMA_DB_PATH = ROOT_DIR / "resources" / "chroma_db"

class BaseSystem(ABC):

    @abstractmethod
    def run(self, input: str) -> str:
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
        
        self.articles = articles
        self.local_embeddings = embeddings
        self.config = load_config(config_path)
        self.vectorstore = self._load_vectorstore(articles)
        self.reranker = Reranker(reranker_model)


    def _load_vectorstore(self, articles):
        if CHROMA_DB_PATH.exists() and any(CHROMA_DB_PATH.iterdir()):
            print("🔁 Loading existing Chroma vectorstore...")
            vectorstore = Chroma(
                persist_directory=str(CHROMA_DB_PATH),
                embedding_function=self.local_embeddings
            )
        else:
            print("🆕 Building new Chroma vectorstore...")
            vectorstore = Chroma.from_texts(texts=articles["texts"], embedding=self.local_embeddings, ids=articles["ids"], metadatas=articles["metadatas"], persist_directory=str(CHROMA_DB_PATH))
        
        return vectorstore
    

    def retrieve_documents(self, query: str, rerank: bool=True):
        documents = self.vectorstore.similarity_search(query, k=self.config["initial_k"])
        if rerank:
            documents = self.reranker.rerank(query, documents, k=self.config["rerank_k"])
        return documents
    

    def purge_store(self):
        if CHROMA_DB_PATH.exists() and any(CHROMA_DB_PATH.iterdir()):
            for filename in os.listdir(CHROMA_DB_PATH):
                file_path = os.path.join(CHROMA_DB_PATH, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # delete file or symlink
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # delete directory
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
            self.vectorstore = self._load_vectorstore(self.articles)


class RAG(BaseSystem):

    def __init__(self, config_path: PathLike, LLM: ChatOllama, retriever: Retriever, database: nx.MultiDiGraph, lt_memory: Optional[str]):
        self.config = load_config(config_path)
        self.LLM = LLM
        self.retriever = retriever
        self.database = database
        self.system_prompt = SystemMessage(content=self.config["system_prompt"])
        self.st_memory = [self.system_prompt]
        self.lt_memory = lt_memory


    def run_flux(self, input: str, rerank: bool = True) -> Generator:
        documents = self.retriever.retrieve_documents(input, rerank=rerank)
        retrieved_context = self._get_text_from_documents(documents)
        self.st_memory.append(HumanMessage(f"Context: {retrieved_context}\n\n{input}"))

        response = ""
        for chunk in self.LLM.stream(self.st_memory):
            response += chunk.content
            yield str(chunk.content)

        self.st_memory.append(AIMessage(response))


    def run(self, input: str, rerank: bool=True) -> str:

        response = ""
        for chunk in self.run_flux(input, rerank=rerank):
            # print(chunk, end='', flush=True)
            response += chunk

        return response

    def generate_question(self) -> str:
        pass

    def _expand_context(self):
        #navigate in the graph
        pass

    def _get_text_from_documents(self, documents: list[Document]) -> str:
        content = []
        for doc in documents:
            origin = []
            origin.append(self.database.nodes[doc.id]["data"].metadata["book"])
            origin.append(self.database.nodes[doc.id]["data"].metadata["part"])
            origin.append(self.database.nodes[doc.id]["data"].metadata["chapter"])
            origin = " -> ".join(origin)
            content.append(": ".join([origin, doc.page_content]))
        retrieved_context = "\n".join(content)

        return retrieved_context
