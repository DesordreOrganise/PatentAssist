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
import re
from collections import defaultdict

from src.utils.tools import load_config


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CHROMA_DB_PATH = ROOT_DIR / "resources" / "chroma_db"
RESOURCES_PATH = ROOT_DIR / "resources"

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
        self.st_memory = [self.system_prompt.copy()]
        self.lt_memory = lt_memory
        self.categories = self._load_categories()
        self.examples = self._load_examples()
        self.question = ""


    def _load_examples(self) -> dict:
        examples = {}
        if RESOURCES_PATH.exists() and any(RESOURCES_PATH.iterdir()):
            pre_ex_folder = os.path.join(str(RESOURCES_PATH), "EQE_PreEx")
            eqe2022_path = os.path.join(pre_ex_folder, "EQE_2022_PreEx_final_documentLess.json")
            eqe2021_path = os.path.join(pre_ex_folder, "EQE_2021_PreEx_final_documentLess.json")
            eob_path = os.path.join(str(RESOURCES_PATH), "EOB.json")
            examples["EQE_2022_PreEx_final_documentLess.json"] = load_config(eqe2022_path)
            examples["EQE_2021_PreEx_final_documentLess.json"] = load_config(eqe2021_path)
            examples["EOB.json"] = load_config(eob_path)

        return examples
    

    def _load_categories(self) -> dict:
        if RESOURCES_PATH.exists() and any(RESOURCES_PATH.iterdir()):
            categories_folder = os.path.join(str(RESOURCES_PATH), "questions")
            categories_file = os.path.join(categories_folder, "categories2questions.json")
            categories = load_config(categories_file)
            return categories
        else:
            print("Error: resources path don't exist.")
            return {}


    def run_flux(self, input: str, rerank: bool = True) -> Generator:
        assert self.question != "", "You have to generate a question before running the flux."

        documents = self.retriever.retrieve_documents(self.question, rerank=rerank)
        retrieved_context = self._get_text_from_documents(documents)
        self.st_memory.append(SystemMessage(f"Here are some articles that may be related to this question:  use them to source and justify your explanation :\n\n{retrieved_context}"))
        self.st_memory.append(HumanMessage(f"reminders, this is the quesion you asked me : \n\"{self.question}\"\n\nhere is my answer : \"{input}\""))

        response = ""
        for chunk in self.LLM.stream(self.st_memory):
            response += chunk.content
            yield str(chunk.content)

        self.st_memory.append(AIMessage(response))


    def run(self, input: str, rerank: bool=False) -> str:

        response = ""
        for chunk in self.run_flux(input, rerank=rerank):
            # print(chunk, end='', flush=True)
            response += chunk

        return response


    def generate_question(self, topic: str, type:str) -> Generator:
        examples = self._fetch_examples(topic)
        if examples:
            examples_string = "\n\n\t- Question Example :".join(examples)
        else:
            examples_string = ""
        print(examples_string)

        self.st_memory.clear()
        self.st_memory.append(self.system_prompt.copy())
        self.st_memory.append(HumanMessage(f"Ask me a {type} question about the category {topic}. To generate your question, refer to the syntax, subject and difficulty of the following examples : \n\n {examples_string}"))

        self.st_memory.append((f""))
        response = ""
        for chunk in self.LLM.stream(self.st_memory):
            response += chunk.content
            yield str(chunk.content)

        self.st_memory.append(AIMessage(response))
        self.question = response


    def reformulate_question(self, question: str):
        self.st_memory.append(SystemMessage(f"""You're a helpful assistant and have to help a student answer questions related to patent law.
                                            Given a question, think step by step and reformulate the question in terms of which concepts are to be sought to answer. Don't answer to the question."""))
        self.st_memory.append(HumanMessage(question))
        response = self.LLM.invoke(self.st_memory).content
        self.st_memory.append(AIMessage(response))

        return response


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


    def _get_question_exam(self, exam: str, qnum: int):
        exam_content = self.examples[exam]
        if exam == "EOB.json":
            question_build = []
            question = exam_content[qnum-1]
            return question["question_text"]
        else:
            exercices = exam_content["exercices"]
            question = exercices[qnum-1]
            question_build = []
            question_build.append(question["context"])
            for answer in question["questions"]:
                question_build.append(". ".join([answer["question_code"], answer["question_text"]]))
            return "\n".join(question_build)



    def _fetch_examples(self, topic: str, k: int=3) -> str:
        match = re.match(r"^\d+(?:\.\d+)*", topic)
        if match:
            topic = match.group()
        if topic in self.categories.keys():
            subtopics = self.categories[topic]
            # Préparer un pointeur pour chaque sous-catégorie
            pointeurs = defaultdict(int)
            cles = list(subtopics.keys())
            items_recuperes = []

            while len(items_recuperes) < k:
                for cle in cles:
                    liste = subtopics[cle]
                    index = pointeurs[cle]

                    if index < len(liste):  # Si on a encore des items dans cette catégorie
                        items_recuperes.append(liste[index])
                        pointeurs[cle] += 1

                        if len(items_recuperes) == k:
                            break
            examples = []
            for exam, qnum in items_recuperes:
                examples.append(self._get_question_exam(exam, qnum))
            
        #else:
            #examples from the basic questions
        
            return examples


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