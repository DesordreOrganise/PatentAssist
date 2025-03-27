from scripts.tools import measure, code_Green, code_end, codecarbone_fr
from src.system.rag import Retriever
from langchain_core.documents import Document
from src.utils.evaluation import EvaluationFramework
from src.system.rag_adapter import Retriever_Adapter
from src.utils.preprocessing import EQE_Dataset_Explaination
from src.utils.metrics_retriever.nDCG import NDCG
from src.utils.metrics_retriever.Precision_K import Precision_K
from src.utils.metrics_retriever.Recall_K import Recall_K

from langchain_ollama import OllamaEmbeddings
from typing import Optional
import numpy as np
from tqdm import tqdm
import time
import pandas as pd
import logging
import pickle
import json
import os

logging.basicConfig(level=logging.INFO)
# disable logs from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.disable(logging.DEBUG)


class Benchmark_retriever(Retriever):

    def __init__(self, retriever: Retriever, rerank: bool = False):
        self.retriever = retriever
        self.execution_time = []
        self.carbone = {
            "emission": [],
            "cpu": [],
            "gpu": []
        }
        self.rerank = rerank

    @codecarbone_fr
    @measure
    def measureable_retrieve_documents(self, query: str) -> list[Document]:
        docs = self.retriever.retrieve_documents(query, self.rerank)
        return docs

    def retrieve_documents(self, query: str, rerank: bool = True) -> list[Document]:
        (docs, time), (emission, cpu, gpu) = self.measureable_retrieve_documents(query)
        self.execution_time.append(time)
        self.carbone["emission"].append(emission)
        self.carbone["cpu"].append(cpu)
        self.carbone["gpu"].append(gpu)

        return docs


def benchmark_retriever(retriever: Retriever, dataset: Optional[pd.DataFrame] = None, rerank: bool = True):
    if dataset is None:
        files = ["../resources/EQE_PaperD/" + file for file in os.listdir(
            "../resources/EQE_PaperD/") if file.endswith("_documentLess.json")]
        files.extend(["../resources/EQE_PreEx/EQE_2021_PreEx_final_documentLess.json",
                     "../resources/EQE_PreEx/EQE_2022_PreEx_final_documentLess.json"])
        print(files)
        dataset_eqe = EQE_Dataset_Explaination(files)
        dataset = dataset_eqe.get_dataset()

    measurable_retriever = Benchmark_retriever(retriever, rerank=rerank)
    adapter = Retriever_Adapter(measurable_retriever)

    benchmark = EvaluationFramework([
        NDCG(),
        Precision_K(5),
        Recall_K(5),
        Precision_K(10),
        Recall_K(10),
        Precision_K(20),
        Recall_K(20),
    ])

    start = time.time()
    output = benchmark.evaluate(adapter, dataset)
    end = time.time()
    dt = end - start

    retrieval_time = np.sum(measurable_retriever.execution_time)

    logging.info(f"{code_Green}Average CPU energy: {np.mean(measurable_retriever.carbone['cpu'])} kWh{code_end}")
    logging.info(f"{code_Green}Average GPU energy: {np.mean(measurable_retriever.carbone['gpu'])} kWh{code_end}")
    logging.info(f"{code_Green}Average emissions: {np.mean(measurable_retriever.carbone['emission'])} kgCO2e{code_end}")
    logging.info(f"{code_Green}Average execution time for the retriever: {np.mean(measurable_retriever.execution_time)} s{code_end}")
    logging.info(f"{code_Green}Average execution time for metrics computation: {dt - retrieval_time} s{code_end}")

    for name, metric in output.items():
        logging.info(f"\t-{code_Green}{name}: {metric}{code_end}")

    output_json = {
        "retriever_model": retriever.local_embeddings.model,

        "number_of_inputs": len(dataset),

        "retriever": {
            "med": np.median(measurable_retriever.execution_time),
            "avg": np.mean(measurable_retriever.execution_time),
            "min": np.min(measurable_retriever.execution_time),
            "max": np.max(measurable_retriever.execution_time),
        },

        "metrics": {},

        "carbone": {
            "emission": {
                "med": str(np.median(measurable_retriever.carbone["emission"])) + " kgCO2e",
                "avg": str(np.mean(measurable_retriever.carbone["emission"])) + " kgCO2e",
                "min": str(np.min(measurable_retriever.carbone["emission"])) + " kgCO2e",
                "max": str(np.max(measurable_retriever.carbone["emission"])) + " kgCO2e",
            },
            "cpu": {
                "med": str(np.median(measurable_retriever.carbone["cpu"])) + " kWh",
                "avg": str(np.mean(measurable_retriever.carbone["cpu"])) + " kWh",
                "min": str(np.min(measurable_retriever.carbone["cpu"])) + " kWh",
                "max": str(np.max(measurable_retriever.carbone["cpu"])) + " kWh",
            },
            "gpu": {
                "med": str(np.median(measurable_retriever.carbone["gpu"])) + " kWh",
                "avg": str(np.mean(measurable_retriever.carbone["gpu"])) + " kWh",
                "min": str(np.min(measurable_retriever.carbone["gpu"])) + " kWh",
                "max": str(np.max(measurable_retriever.carbone["gpu"])) + " kWh",
            }
        },
    }

    for name, metric in output.items():
        output_json['metrics'][name] = metric

    os.makedirs("output", exist_ok=True)

    count = 0
    count = sum([1 for file in os.listdir("output/")
                if file.startswith("benchmark_retriever")])
    with open(f"output/benchmark_retriever_{count}.json", "w") as f:
        json.dump(output_json, f)


if __name__ == "__main__":

    logging.info("Loading graph")
    with open("../resources/LegalBases/graph.pkl", "rb") as f:
        g = pickle.load(f)

    print('loading files')
    texts = []
    metadatas = []
    ids = []
    for id, doc in g.nodes(data=True):
        if doc["data"].page_content and doc["label"] != "Guideline":
            ids.append(id)
            metadatas.append({"label": doc["label"]})
            texts.append(doc["data"].page_content)
    articles = {}
    articles["texts"] = texts
    articles["ids"] = ids
    articles["metadatas"] = metadatas

    print("Loading retriever")
    config_path = "../config/retriever_config.yaml"
    local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    retriever = Retriever(config_path, articles, local_embeddings)
    retriever.purge_store()
    benchmark_retriever(retriever, rerank=False)
