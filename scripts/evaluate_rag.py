from src.system.rag import Retriever, RAG
from src.utils.preprocessing import EQE_Dataset_Explaination

from src.utils.metrics_rag.rouge import rouge1, rouge2, rougeL, rougeLsum
from src.utils.metrics_rag.bleu import bleu
from src.utils.metrics_rag.meteor import meteor_metric
from src.utils.metrics_rag.bert_score import bert_score_acc, bert_score_recall, bert_score_f1

from src.utils.evaluation import EvaluationFramework

import os
import networkx as nx
import pickle
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_ollama import ChatOllama


def framework_RAG_exemple(rag: RAG):
    # loading dataset
    Folder_Path = ["../../resources/EQE_PaperD/", "../../resources/EQE_PreEx/"]
    files = []
    files.extend([Folder_Path[0] + file for file in os.listdir(Folder_Path[0]) if file.endswith("_documentLess.json")])

    files = [files[3]]
    # files = [Folder_Path[0] + file for file in os.listdir(Folder_Path[0]) if file.endswith("_documentLess.json")]

    # files.extend(["../../resources/EQE_PreEx/EQE_2021_PreEx_final_documentLess.json", "../../resources/EQE_PreEx/EQE_2022_PreEx_final_documentLess.json"])
    dataset = EQE_Dataset_Explaination(files)
    test_data = dataset.get_dataset()
    
    # initializing metrics
    metrics = [
        rouge1,
        rouge2,
        rougeL,
        rougeLsum,
        bleu,
        meteor_metric,
        bert_score_acc,
        bert_score_recall,
        bert_score_f1
    ]


    for _, sample in test_data.iterrows():
        print(sample)
        break

    # initializing framework
    fr = EvaluationFramework(metrics)
    
    print("Evaluating RAG")
    # initializing retriever
    fr.evaluate(rag, test_data)


if __name__ == "__main__":

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
    config_path = "../../config/retriever_config.yaml"
    print("Loading embeddings")
    local_embeddings = OllamaEmbeddings(model="all-minilm")
    print("Loading retriever")
    retriever = Retriever(config_path, articles, local_embeddings)

    llm = ChatOllama(model="llama3")

    rag_config_path = "../../config/config.yaml"
    rag = RAG(rag_config_path, llm, retriever, g, None)

    framework_RAG_exemple(rag)