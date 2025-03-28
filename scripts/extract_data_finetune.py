from src.utils.tools import get_base_id, from_clean_to_id, extract_articles, clean_article, extract_rules, clean_rule
from src.system.rag import Retriever
from src.utils.preprocessing import EQE_Dataset_Explaination, EQE_Dataset_Explaination_before2k19, EOB_dataset, EPAC_dataset

from langchain_ollama import OllamaEmbeddings
from typing import Optional
import numpy as np
from tqdm import tqdm
import pandas as pd
import logging
import pickle
import os
import networkx as nx

logging.basicConfig(level=logging.INFO)
# disable logs from httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.disable(logging.DEBUG)


def extract_dataset_finetune(retriever: Retriever, dataset: Optional[pd.DataFrame] = None, rerank: bool = True, *, g: nx.MultiGraph, limit_per_gt: int = 3):
    if dataset is None:
        files = ["../resources/EQE_PaperD/" + file for file in os.listdir(
            "../resources/EQE_PaperD/") if file.endswith("_documentLess.json")]
        files.extend(["../resources/EQE_PreEx/EQE_2021_PreEx_final_documentLess.json",
                     "../resources/EQE_PreEx/EQE_2022_PreEx_final_documentLess.json"])
        dataset_eqe = EQE_Dataset_Explaination(files)
        dataset = dataset_eqe.get_dataset()

        files2 = ["../resources/EQE_PreEx/" + file for file in os.listdir(
            "../resources/EQE_PreEx/") if file.endswith("_documentLess.json")]
        for file in files:
            if file in files2:
                files2.remove(file)
        dataset_eqe2 = EQE_Dataset_Explaination_before2k19(files2)
        dataset2 = dataset_eqe2.get_dataset()

        dataset_eob = EOB_dataset(["../resources/EOB/EOB.json"])
        dataset3 = dataset_eob.get_dataset()

        dataset_EPAC = EPAC_dataset(["../resources/EPAC/mcq_EPAC.json"])
        dataset4 = dataset_EPAC.get_dataset()

        dataset = pd.concat([dataset, dataset2, dataset3, dataset4])

    output_dataset = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):

        docs = retriever.retrieve_documents(row["X"], rerank=False)
        docs = retriever.reranker.rerank(row["X"], docs, 5)

        gt_rules = extract_rules(row["Y"])
        gt_rules = [clean_rule(rule) for rule in gt_rules]
        gt_rules = [from_clean_to_id(rule) for rule in gt_rules]

        gt_articles = extract_articles(row["Y"])
        gt_articles = [clean_article(article) for article in gt_articles]
        gt_articles = [from_clean_to_id(article) for article in gt_articles]

        gt = gt_rules + gt_articles
        gt_docs = [g.nodes[id]["data"] for id in gt]

        for item in docs:
            if item.id not in gt:
                limite = limit_per_gt
                for ground_doc in gt_docs:
                    if limite == 0:
                        break
                    limite -= 1

                    id = np.random.choice(range(len(docs)))
                    non_pert = docs[id]

                    output_dataset.append({
                        "query": row["X"],
                        "pertinent": ground_doc.page_content,
                        "non_pertinent": non_pert.page_content,
                    })

    output_dataset = pd.DataFrame(output_dataset)

    output_dataset.to_csv("output_dataset.csv", index=False)


if __name__ == "__main__":

    logging.info("Loading graph")
    with open("../resources/LegalBases/graph.pkl", "rb") as f:
        g = pickle.load(f)

    print('loading files')
    texts = []
    metadatas = []
    ids = []
    for id, doc in g.nodes(data=True):

        try:
            id = get_base_id(doc["data"].id)
        except Exception:
            id = doc["data"].id

        # or (doc["label"] == "Guideline" and g.edges(id))):
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
    # local_embeddings = OllamaEmbeddings(model="all-minilm")
    local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # local_embeddings = OllamaEmbeddings(model="bge-m3")
    retriever = Retriever(config_path, articles, local_embeddings,"cross-encoder/ms-marco-MiniLM-L-12-v2", g)

    retriever.purge_store()
    extract_dataset_finetune(retriever, rerank=True, g=g, limit_per_gt=3)
