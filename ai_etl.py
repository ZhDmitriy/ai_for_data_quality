"""
    Применяем AI в ETL/ELT процессах для обеспечения высокого уровня качества данных.

    Пример рассмотрен на предметной области торговли на маркетплейсах. Может быть применено на торговле в общем смысле.

"""

import pandas as pd 
from langchain_gigachat import GigaChat
from langchain_core.tools import tool 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.schema import SystemMessage
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

    def create_model(self) -> str:
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
        return data_sales
    
    @tool 
    def check_dataset_correct(self, data_sales: pd.DataFrame) -> pd.DataFrame: 
        """ Проверяем датасет переданный ранее на корректность по следующим признакам: 

            data_error_standart (переменная)- здесь представлены нормы отклонения от среднего значения для каждого поля в процентах (если указано
            значение None, то не применяй сравнение для этого поля), например, чтобы понять норму отклонения для себестоимости, мы должны 
            посчитать среднее значений для этой категории товара без учета выбросов и сравнить с текущим значением строки это значение, 
            если значение строки отличается больше чем на заданный показатель, то данные скорее неверные. 

            Добавь в датафрейм новое поле "is_valid", если строка прошла валидацию по этим критерия, то верни 1, иначе 0

            Верни далее по цепочке обновленный датафрейм.
        """
        data_error_standart = self.data_error_standart
        data_sales['is_valid'] = 1
        return data_sales

    @tool
    def add_your_solution(self, data_sales: pd.DataFrame) -> pd.DataFrame: 
        """
            На основе полученного датафрейма, для каждой строки сделай свой вердик относительно корректности по 4 оценкам: 
            0 - не похоже на правду  
            0.25 - похоже на правду на 25% 
            0.5 - верно наполовину
            1 - абсолютная правда  

            Добавь новое поле is_valid_gpt и расставь эти значения для каждой строки. 

            Верни далее по цепочке обновленный датафрейм. 
        """
        data_sales['is_valid_gpt'] = 0
        return data_sales

    def create_agent_chain(self) -> pd.DataFrame: 
        """ Запускаем всю цепочку обработки данных """
        ai = AiModelClient()
        giga_model = ai.create_model()

        tools = [
            self.read_dataset_sales, 
            self.check_dataset_correct, 
            self.add_your_solution
        ]

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
                Ты AI агент для проверки точности данных. Выполни последовательно все этапы проверки.
            """), 
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agent = create_tool_calling_agent(giga_model, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        return agent_executor
    
    def run_chain(self):
        """ Запуска всю цепочку обработки """   
        agent_executor = self.create_agent_chain()
        result = agent_executor.invoke({
            "input": "Запусти полную проверку данных: прочитай датасет, проверь корректность и добавь AI оценку"
        })

        return result["output"]


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

    agent = PrecisionAiAgent()
    final_data = agent.run_chain()

