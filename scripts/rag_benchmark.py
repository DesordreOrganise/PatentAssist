
# file that measure the execution time of Retriever, Ranker and LLM

from scripts.tools import measure, code_Green, code_end, codecarbone_fr
from src.system.rag import RAG, Retriever
from src.utils.metrics import Adapteur_Multi_2_Metric
from src.utils.evaluation import EvaluationFramework
from src.utils.preprocessing import EQE_Dataset_Explaination
from langchain_ollama import ChatOllama

from src.utils.metrics_rag.rouge import ROUGE
from src.utils.metrics_rag.bleu import BLEU
from src.utils.metrics_rag.meteor import Meteor
from src.utils.metrics_rag.bert_score import BertScore

from langchain_ollama import OllamaEmbeddings
from typing import Optional
import numpy as np
import time
import pandas as pd
import logging
import pickle
import json
import os

logging.basicConfig(level=logging.INFO)
# disable logs from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)


class Benchmark_rag(RAG):

    def __init__(self, rag: RAG):
        self.rag = rag
        self.execution_time = []
        self.carbone = {
            "emission": [],
            "cpu": [],
            "gpu": []
        }

    @codecarbone_fr
    @measure
    def measurable_run(self, input: str, rerank: bool) -> str:
        response = self.rag.run(input, rerank)

        return response

    def run(self, input: str, rerank: bool = True) -> str:
        (response, time), (emission, cpu, gpu) = self.measurable_run(input, rerank)
        self.execution_time.append(time)
        self.carbone["emission"].append(emission)
        self.carbone["cpu"].append(cpu)
        self.carbone["gpu"].append(gpu)

        return response


def _setup_metrics():
    rouge = ROUGE()
    bert_score = BertScore()

    rouge1 = Adapteur_Multi_2_Metric(rouge, "rouge1")
    rouge2 = Adapteur_Multi_2_Metric(rouge, "rouge2")
    rougeL = Adapteur_Multi_2_Metric(rouge, "rougeL")
    rougeLsum = Adapteur_Multi_2_Metric(rouge, "rougeLsum")

    bert_score_precision = Adapteur_Multi_2_Metric(bert_score, "precision")
    bert_score_recall = Adapteur_Multi_2_Metric(bert_score, "recall")
    bert_score_f1 = Adapteur_Multi_2_Metric(bert_score, "f1")

    return [
        rouge1, rouge2, rougeL, rougeLsum,
        BLEU(),
        Meteor(),
        # bert_score_precision, bert_score_recall, bert_score_f1
    ]


def benchmark_rag(rag: RAG, dataset: Optional[pd.DataFrame] = None):
    if dataset is None:
        dataset_eqe = EQE_Dataset_Explaination(
            ["../../resources/EQE_PaperD/EQE_2024_PaperD_final_documentLess.json"])
        dataset = dataset_eqe.get_dataset()

    measurable_rag = Benchmark_rag(rag)
    # measurable_rag = Benchmark_retriever(retriever)
    # adapter = Retriever_Adapter(measurable_retriever)

    benchmark = EvaluationFramework(_setup_metrics())

    start = time.time()
    output = benchmark.evaluate(measurable_rag, dataset)
    end = time.time()
    dt = end - start

    rag_time = np.sum(measurable_rag.execution_time)

    logging.info(f"{code_Green}Average execution time for the retriever: {
                 np.mean(measurable_rag.execution_time)} s{code_end}")
    logging.info(f"{code_Green}Average execution time for metrics computation: {
                 (dt - rag_time) / len(dataset)} s{code_end}")

    for name, metric in output.items():
        logging.info(f"\t-{code_Green}{name}: {metric}{code_end}")

    output_json = {
        "retriever_model": retriever.local_embeddings.model,

        "number_of_inputs": len(dataset),

        "rag": {
            "total": rag_time,
            "med": np.median(measurable_rag.execution_time),
            "avg": np.mean(measurable_rag.execution_time),
            "min": np.min(measurable_rag.execution_time),
            "max": np.max(measurable_rag.execution_time),
        },

        "metrics": {
            "total_time": dt - rag_time,
            "avg_time": (dt - rag_time) / len(dataset)
        },

        "carbone": {
            "emission": {
                "med": str(np.median(measurable_rag.carbone["emission"])) + " kgCO2e",
                "avg": str(np.mean(measurable_rag.carbone["emission"])) + " kgCO2e",
                "min": str(np.min(measurable_rag.carbone["emission"])) + " kgCO2e",
                "max": str(np.max(measurable_rag.carbone["emission"])) + " kgCO2e",
            },
            "cpu": {
                "med": str(np.median(measurable_rag.carbone["cpu"])) + " kWh",
                "avg": str(np.mean(measurable_rag.carbone["cpu"])) + " kWh",
                "min": str(np.min(measurable_rag.carbone["cpu"])) + " kWh",
                "max": str(np.max(measurable_rag.carbone["cpu"])) + " kWh",
            },
            "gpu": {
                "med": str(np.median(measurable_rag.carbone["gpu"])) + " kWh",
                "avg": str(np.mean(measurable_rag.carbone["gpu"])) + " kWh",
                "min": str(np.min(measurable_rag.carbone["gpu"])) + " kWh",
                "max": str(np.max(measurable_rag.carbone["gpu"])) + " kWh",
            }
        }
    }

    for name, metric in output.items():
        output_json['metrics'][name] = metric

    os.makedirs("output", exist_ok=True)

    count = 0
    count = sum([1 for file in os.listdir("output/")
                if file.startswith("benchmark_rag")])
    with open(f"output/benchmark_rag_{count}.json", "w") as f:
        json.dump(output_json, f)


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

    benchmark_rag(rag)
