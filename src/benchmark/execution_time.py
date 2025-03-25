# file that measure the execution time of Retriever, Ranker and LLM
import time
import logging
from src.system.rag import RAG, Retriever
from langchain_core.messages import AIMessage, HumanMessage
from src.utils.preprocessing import EQE_Dataset_Explaination
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
import numpy as np
import pickle
import json
import os

logging.basicConfig(level=logging.INFO)
# disable logs from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

code_Green = '\033[92m'
code_blue = '\033[94m'
code_end = '\033[0m'

# Load the RAG model


def measure(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"{code_blue}Execution time of {
                     func.__name__}: {end - start}{code_end}")
        return result, end - start
    return wrapper


@measure
def measure_retriever(system: Retriever, question: str):
    return system.retrieve_documents(question, rerank=False)


@measure
def measure_reranker(system: Retriever, question: str, documents):
    documents = system.reranker.rerank(
        question, documents, k=system.config["rerank_k"])

    return documents


@measure
def measure_rag(system: RAG, question: str, documents):

    retrieved_context = system._get_text_from_documents(documents)
    system.st_memory.append(HumanMessage(
        f"Context: {retrieved_context}\n\n{question}"))

    response = ""
    for chunk in system.LLM.stream(system.st_memory):
        # print(chunk.content, end='', flush=True)
        response += chunk.content

    system.st_memory.append(AIMessage(response))

    return response


if __name__ == "__main__":

    logging.info("Loading graph")
    with open("../../resources/LegalBases/graph.pkl", "rb") as f:
        g = pickle.load(f)



    print('loading files')
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
    config_path = "../../config/retriever_config.yaml"
    local_embeddings = OllamaEmbeddings(model="all-minilm")
    retriever = Retriever(config_path, articles, local_embeddings)


    print("Loading Rag")
    # llm = ChatOllama(model="gemma3:1b")
    llm = ChatOllama(model="gemma:2b")
    rag_config_path = "../../config/config.yaml"
    rag = RAG(rag_config_path, llm, retriever, g, None)


    print("Loading dataset")
    dataset = EQE_Dataset_Explaination(
        ["../../resources/EQE_PaperD/EQE_2024_PaperD_final_documentLess.json"])
    dataset = dataset.get_dataset()



    timing = {
        "retriever": [],
        "reranker": [],
        "rag": []
    }

    # for loop with tqdm
    from tqdm import tqdm
    for _, row in tqdm(dataset.iterrows()):
        question = row["X"]

        documents, time_ret = measure_retriever(retriever, question)
        documents, time_rer = measure_reranker(retriever, question, documents)

        _, time_rag = measure_rag(rag, question, documents)

        timing["retriever"].append(time_ret)
        timing["reranker"].append(time_rer)
        timing["rag"].append(time_rag)

    # write in logging in green color

    logging.info(f"{code_Green}Average time for Retriever: {sum(timing['retriever'])/len(timing['retriever'])}{code_end}")
    logging.info(f"{code_Green}Average time for Reranker: {sum(timing['reranker'])/len(timing['reranker'])}{code_end}")
    logging.info(f"{code_Green}Average time for RAG: {sum(timing['rag'])/len(timing['rag'])}{code_end}")

    # save output as json
    output = {
        "embedding_model": local_embeddings .model,
        "reranker_model": retriever.reranker.model,
        "llm_model": llm.model,

        "retriever": {
            "med": np.median(timing['retriever']),
            "avg": sum(timing['retriever'])/len(timing['retriever']),
            "min": min(timing['retriever']),
            "max": max(timing['retriever']),
        },
        "reranker": {
            "med": np.median(timing['reranker']),
            "avg": sum(timing['reranker'])/len(timing['reranker']),
            "min": min(timing['reranker']),
            "max": max(timing['reranker']),
        },
        "llm": {
            "med": np.median(timing['rag']),
            "avg": sum(timing['rag'])/len(timing['rag']),
            "min": min(timing['rag']),
            "max": max(timing['rag']),
        }
    }

    # make output folder
    os.makedirs("output", exist_ok=True)

    # output to file with day as Name
    with open(f"output/benchmark_time_{time.strftime('%Y-%m-%d')}.json", "w") as f:
        json.dump(output, f)
