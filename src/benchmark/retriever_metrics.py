from src.benchmark.tools import measure, code_Green, code_end
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


class Benchmark_retriever(Retriever):

    def __init__(self, retriever: Retriever, rerank: bool = False):
        self.retriever = retriever
        self.execution_time = []
        self.rerank = rerank

    @measure
    def measureable_retrieve_documents(self, query: str) -> list[Document]:
        docs = self.retriever.retrieve_documents(query, self.rerank)
        return docs

    def retrieve_documents(self, query: str, rerank: bool = True) -> list[Document]:
        docs, time = self.measureable_retrieve_documents(query)
        self.execution_time.append(time)
        return docs


def benchmark_retriever(retriever: Retriever, dataset: Optional[pd.DataFrame] = None, rerank: bool=True):
    if dataset is None:
        dataset_eqe = EQE_Dataset_Explaination(
            ["../../resources/EQE_PaperD/EQE_2024_PaperD_final_documentLess.json"])
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

        "metrics": {}
    }

    for name, metric in output.items():
        output_json['metrics'][name] = metric

    os.makedirs("output", exist_ok=True)

    count = 0
    count = sum([1 for file in os.listdir("output/") if file.startswith("benchmark_retriever")])
    with open(f"output/benchmark_retriever_{count}.json", "w") as f:
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
    retriever.purge_store()
    benchmark_retriever(retriever)
