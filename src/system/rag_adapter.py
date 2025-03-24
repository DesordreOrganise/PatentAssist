from src.system.rag import BaseSystem, Retriever


class Retriever_Adapter(BaseSystem):

    def __init__(self, retriever: Retriever, rerank: bool=True):
        self.retriever = retriever
        self.rerank = rerank

    def run(self, input: str) -> str:
        documents = self.retriever.retrieve_documents(input, rerank=self.rerank)

        print(documents)

        return ""



