from abc import ABC, abstractmethod
import os
import json
from typing import List
import pandas as pd

class Dataset(ABC):

    # return a dataFrame with 'input' and 'ground_truth' columns
    @abstractmethod
    def get_dataset(self) -> pd.DataFrame:
        pass


class EQE_Dataset_Explaination(Dataset):
    def __init__(self, json_datas: List[str]):
        self.data = json_datas
        self.dataset = self._create_dataset(json_datas)

    def _create_dataset(self, json_datas: List[str]) -> pd.DataFrame:
        # Liste pour stocker les données extraites
        data_list = []

        # Parcourir les exercices et extraire les informations
        for file in json_datas:
            # check if file exists
            if not os.path.exists(file):
                raise FileNotFoundError(f"File {file} not found")

            with open(file, 'r') as f:
                data = json.load(f)

            # data = json.loads(file)

            for exercice in data['exercices']:
                context = exercice['context']
                for question in exercice['questions']:
                    question_text = question['question_text']
                    possible_answers = question['answer']

                    if 'examiner_note' in question:
                        possible_answers = question['examiner_note']

                    # Ajouter les informations extraites à la liste
                    data_list.append({
                        'context': context,
                        'question_text': question_text,
                        'answer': possible_answers
                    })

        # Créer un DataFrame à partir de la liste
        return pd.DataFrame(data_list)

    def get_dataset(self):
        return self._format_dataset_joins()


    def _format_dataset_joins(self) -> pd.DataFrame:
        output = []

        for _, row in self.dataset.iterrows():
            # make sure question ends with a question mark
            if row['question_text'][-1] != '?':
                row['question_text'] += '?'

            output.append({
                'X': row['context'] + ' ' + row['question_text'],
                'Y': row['answer']
            })

        return pd.DataFrame(output)

