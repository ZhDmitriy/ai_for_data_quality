"""
    Применяем AI в ETL/ELT процессах для обеспечения высокого уровня качества данных.

    Пример рассмотрен на предметной области торговли на маркетплейсах - торговля в общем смысле. 
"""

import pandas as pd 
import os
from dotenv import load_dotenv


load_dotenv()


class DataSourceClient:
    """ Клиент управления источником данных. 
        В этом классе созданы синтетические данные из 100 строк: 
        - 80% корректные 
        - 20% неккоректные 
    """

    def create_dataset(self) -> pd.DataFrame: 
        """ Читаем датасет с данными """
        data_ecom = pd.read_csv("data_ecom.csv", sep=';')
        print(data_ecom)


class AiModelClient: 
    """ Клиент по взаимодействию с LLM моделью DeepSeek-V3 """

    def _init__(self):
        self.api_key = os.environ.get('DEEPSEEK_API_KEY')


class PrecisionAiAgent:
    """ ИИ агент для оценки точности данных """

    def __init__(self):
        pass 

    def get_prompt_requests(self) -> pd.DataFrame: 
        """ Получаем промпт запросы для оценки точности данных """
        pass 


class FulnessAiAgent: 
    """ ИИ агент для оценки полноты данных """

    def __init__(self):
        pass 


    def get_prompt_requests(self) -> pd.DataFrame: 
        """ Получаем промпт запрос для оценки полноты данных """
        pass 


class ConsistencyAiAgent:
    """ ИИ агент для оценки согласованности данных """

    def __init__(self): 
        pass 

    def get_prompt_requests(self) -> pd.DataFrame: 
        """ Получаем промпт запрос для оценки согласованности данных """
        pass 


class ValidityAiAgent:
    """ ИИ агент для оценки достоверности данных """ 

    def __init__(self):
        pass 

    def get_prompt_requests(self) -> pd.DataFrame: 
        """ Получаем промпт запрос для оценки достоверности данных """
        pass 


class RelevantAiAgent:
    """ ИИ агент для оценки своевременности данных """

    def __init__(self):
        pass 

    def get_prompt_requests(self) -> pd.DataFrame: 
        """ Получаем промпт запрос для оценки своерменности данных """
        pass 


if __name__ == '__main__': 
    ai_model = AiModelClient()

    data_source = DataSourceClient()
    data_source.create_dataset()
