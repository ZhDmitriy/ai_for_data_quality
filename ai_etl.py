"""
    Применяем AI в ETL/ELT процессах для обеспечения высокого уровня качества данных.

    Пример рассмотрен на предметной области торговли на маркетплейсах. Может быть применено на торговле в общем смысле.

"""

import pandas as pd 
from langchain_gigachat import GigaChat
from langchain_core.tools import tool 
from langchain_community.tools import DuckDuckGoSearchResults
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
        data_ecom = pd.read_csv("data_ecom.csv", sep=',')
        return data_ecom


class AiModelClient: 
    """ Клиент по взаимодействию с LLM моделью GigaChat """

    BASE_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

    def create_api_token(self) -> str:
        """ Создаем экземпляр класса для работы с моделями GigaChat """

        giga = GigaChat(credentials=os.environ.get('GIGACHAT_AUTHORIZATION_KEY'), 
                        model='GigaChat-preview', 
                        verify_ssl_certs=False, 
                        )
        
        return giga 


class PrecisionAiAgent:
    """ ИИ агент для оценки точности данных """

    def __init__(self): 
        self.data_error_standart = {
            "date": None, 
            "product_id": None, 
            "product_name": None, 
            "category": None, 
            "price": 0.15, 
            "cost": 0.1, 
            "quantity_sold": 0.2, 
            "marketing_spend": 0.1, 
            "warehouse_cost": 0.1, 
            "conversion_rate": 0.02, 
            "gross_margin": 0.05, 
            "net_margin": 0.03, 
            "timestamp": None
        }

    @tool
    def read_dataset_sales(self) -> pd.DataFrame: 
        """ Читаем датасет данных о продажах на маркетплейсе. Датасет содержит поля: 
            - date - дата продажи
            - product_id - id продукта 
            - product_name - наименование продукта/товара
            - category - категория товара
            - price - цена продажи товара, установленная продавцом
            - cost - себестоимость продажи 1 товара
            - quantity_sold - количество продаж
            - marketing_spend - маркетинговые расходы (внутренние и внешние)
            - warehouse_cost - расходы склада на содержание товара
            - conversion_rate - конверсия в покупку
            - gross_margin - гросс маржа
            - net_margin - нетто маржа
            - net_profit - чистая прибыль
            - timestamp - дата загрузки в базу данных

            Верни далее по по цепочке этот датафрейм.
        """
        ds = DataSourceClient()
        data_sales = ds.create_dataset()
    
    @tool 
    def check_dataset_correct(self) -> pd.DataFrame: 
        f""" Проверяем датасет переданный ранее на корректность по следующим признакам: 

            {self.data_error_standart} - здесь представлены нормы отклонения от среднего значения для каждого поля в процентах (если указано
            значение None, то не применяй сравнение для этого поля), например, чтобы понять норму отклонения для себестоимости, мы должны 
            посчитать среднее значений для этой категории товара без учета выбросов и сравнить с текущим значением строки это значение, 
            если значение строки отличается больше чем на заданный показатель, то данные скорее неверные. 

            Добавь в датафрейм новое поле "is_valid", если строка прошла валидацию по этим критерия, то верни 1, иначе 0

            Верни далее по цепочке обновленный датафрейм.
        """
        ds = DataSourceClient()
        data_sales = ds.create_dataset()

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
    test = ai_model.create_api_token()
    print(test)

    data_source = DataSourceClient()
    data_source.create_dataset()
