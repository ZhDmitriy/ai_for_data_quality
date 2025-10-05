"""
    
    Применяем AI в ETL/ELT процессах для обеспечения высокого уровня качества данных.

    Пример рассмотрен предметной области образования. 

"""

import pandas as pd 
import json
import uuid
import requests
from langchain_gigachat import GigaChat
import os
from dotenv import load_dotenv

load_dotenv()


class DataSourceClient:
    """ Клиент по управлению источниками данных. 
        В этом классе созданы синтетические данные из 100 строк: 
        - 80% корректные 
        - 20% неккоректные 
        которые преобразуются в json структуру
    """

    def create_dataset(self) -> pd.DataFrame: 
        """ Читаем датасет с данными """
        data_edtech = pd.read_csv("data_edtech.csv", sep=',')
        return data_edtech
    
    def dataset_from_dataframe_to_json(self, dataset_edtech: pd.DataFrame) -> json: 
        """ Преобразовываем DataFrame в json структуру """
        records = dataset_edtech.to_dict('records')
        json_string = json.dumps(records, indent=2)
        return json_string


class AiModelClient: 
    """ Клиент по взаимодействию с LLM моделью GigaChat """

    BASE_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

    def create_model(self) -> str:
        """ Создаем экземпляр класса для работы с моделями GigaChat """

        giga = GigaChat(credentials=os.environ.get('GIGACHAT_AUTHORIZATION_KEY'), 
                        model='GigaChat-preview', 
                        verify_ssl_certs=False
                        )
        
        return giga 
    
    def get_token_auth(self) -> str: 
        """ Получаем токен доступа для отправки запросов """
        rqUID = uuid.uuid4()
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'RqUID': str(rqUID),
            'Authorization': f'Basic {os.environ.get('GIGACHAT_AUTHORIZATION_KEY')}',
        }

        data = {
            'scope': 'GIGACHAT_API_PERS',
        }

        response = requests.post('https://ngw.devices.sberbank.ru:9443/api/v2/oauth', headers=headers, data=data, verify=False)
        if response.status_code == 200:
            return response.json()['access_token']
        else: 
            raise Exception(f"Ошибка получения токена: {response.status_code}")
        

class PrecisionAiAgent:
    """ ИИ агент для оценки точности данных """

    def __init__(self): 
        # Задаём норму отклонения для оценки точности данных для некоторых числовых переменных
        self.data_error_standart = { 
            "scholarship_amount": 1500, 
            "average_study_hours": 15, 
            "library_visits_per_month": 10
        }

    def send_gigachat_request_precision(self, token: str, message: str) -> json:
        """Отправка запроса к GigaChat API"""
        url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}'
        }
        
        data = {
            "model": "GigaChat",
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ],
            "temperature": 0.7
        }
        
        response = requests.post(url, headers=headers, json=data, verify=False)
        return response.json()

    def read_dataset_edtech_data_prompt(self, data_edtech: json) -> str: 
        """ Отправляем промпт запрос на чтение датасета """

        prompt_message = """ Читаем датасет данных о студентах. Датасет содержит поля: 
            - student_id - уникальный идентификатор студента
            - email - электронная почта студента
            - phone_number - мобильный номер телефона студента
            - date_of_birth - дата рождения студента
            - age - возраст студента
            - admission_year - год поступления в учебное заведение
            - faculty - факультет, на котором учится студента
            - group_name - наименование группы обучения
            - gpa - средний балл студента 
            - last_test_score -  балл за последний сданный экзамен или тест
            - attendance_percent - процент посещаемости занятий
            - scholarship_amount - размер стипендии в рублях
            - extracurricular_activities - внеучебные активности
            - on_probation - находится ли студент на академическом испытательном сроке 
            - has_dormitory - проживает ли студент в общежитии
            - enrollment_status - текущий статус обучения студента
            - preferred_language - предпочитаемый язык для обучения или язык, на котором обучается студент
            - mentor_id - идентификатор ментора
            - average_study_hours - среднее количество часов, которое тратит на самостоятельное обучение студент
            - library_visits_per_month - среднее количество посещений библиотеки в месяц
       
            Изучи датасет и верни "ОК", если ты изучил его.
        """ + str(data_edtech)

        return prompt_message
 
    def check_dataset_correct_prompt(self, data_edtech: json) -> str: 
        
        prompt_message = """ Проверяем датасет переданный ранее на корректность по следующим признакам: 

            data_error_standart (переменная) - здесь представлены нормы отклонения от среднего значения для каждого поля в процентах (если указано
            значение None или Null, то не применяй сравнение для этого поля), например, чтобы понять норму отклонения для себестоимости, мы должны 
            посчитать среднее значений для этой категории товара без учета выбросов и сравнить с текущим значением строки это значение, 
            если значение строки отличается больше чем на заданный показатель, то данные скорее неверные. 

            Добавь в структуру новое поле "is_valid", если строка прошла валидацию по этим критерия, то верни 1, иначе 0

            Верни обновленную структуру данных только в формате JSON, не возвращай больше ничего.
        """ + str(data_edtech) + "data_error_standart = " + str(self.data_error_standart)
        
        return prompt_message

    def add_solution_gpt(self, data_edtech: json) -> str: 
        
        prompt_message = """
            На основе полученного датафрейма, для каждой строки сделай свой вердик относительно корректности по 4 оценкам: 
            0 - не похоже на правду  
            0.25 - похоже на правду на 25% 
            0.5 - верно наполовину
            1 - абсолютная правда  

            Добавь новое поле is_valid_gpt и расставь эти значения для каждой строки. 

            Верни JSON структуру, не добавляй больше в сообщение ничего.
        """ + str(data_edtech)
        
        return prompt_message

    def get_accuracy_assessment(self, data_edtech: json) -> str: 
        """ Получаем оценку качества данных на основе анализа данных от ИИ модели """

        prompt_message = """
            Ты изучил переданные. Поставь оценку качества данных от 0 до 1. Оцени параметр точности данных. 

            Точность - оценка отклонения данных от ожидаемых или реальных показателей. 

            Верни просто 1 цифру.

        """ + str(data_edtech)

        return prompt_message

    def create_agent_chain(self, token: str) -> str: 
        """ Запускаем всю цепочку обработки данных """

        ds = DataSourceClient()
        data_edtech = ds.create_dataset()
        data_edtech = ds.dataset_from_dataframe_to_json(dataset_edtech=data_edtech)

        # Изучаем данные 
        learning_data_prompt = self.read_dataset_edtech_data_prompt(data_edtech=data_edtech)
        learning_data = self.send_gigachat_request_precision(token=token, message=learning_data_prompt)['choices'][0]['message']['content']

        check_data_prompt = self.check_dataset_correct_prompt(data_edtech=data_edtech)
        add_solution_gpt_prompt = self.add_solution_gpt(data_edtech=data_edtech)
        get_accuracy_prompt = self.get_accuracy_assessment(data_edtech=data_edtech)

        if "ОК" in learning_data: 

            # Проверяем датасет по первому критерию
            checking_data = self.send_gigachat_request_precision(token=token, message=check_data_prompt)['choices'][0]['message']['content'] 
            
            # Добавляем решение ии модели оценки качества данных в каждой строке
            add_solution_gpt = self.send_gigachat_request_precision(token=token, message=add_solution_gpt_prompt)['choices'][0]['message']['content'] 

            # Оцениваем точность, как параметр качества данных
            accuracy_score = self.send_gigachat_request_precision(token=token, message=get_accuracy_prompt)['choices'][0]['message']['content']

            print(accuracy_score)

        else: 
            print("Модель не смогла изучить переданный датасет")

        return {
            "checking_data_first_sign": checking_data,
            "processed_data": add_solution_gpt, 
            "accuracy_score": accuracy_score
        }


class FulnessAiAgent: 
    """ ИИ агент для оценки полноты данных """

    def send_gigachat_request_fulness(self, token: str, message: str) -> json:
        """Отправка запроса к GigaChat API"""
        url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}'
        }
        
        data = {
            "model": "GigaChat",
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ],
            "temperature": 0.7
        }
        
        response = requests.post(url, headers=headers, json=data, verify=False)
        return response.json()

    def read_dataset_edtech_data_prompt(self, data_edtech: json) -> str: 
        """
            Получаем промпт запрос для оценки полноты данных 
            data_edtech: исходные данные
        """

        prompt_message = """
            На вход ты получаешь датасет с образовательными данными о студентах. 

            Поля в датасете следующие: 
                - student_id - уникальный идентификатор студента
                - email - электронная почта студента
                - phone_number - мобильный номер телефона студента
                - date_of_birth - дата рождения студента
                - age - возраст студента
                - admission_year - год поступления в учебное заведение
                - faculty - факультет, на котором учится студента
                - group_name - наименование группы обучения
                - gpa - средний балл студента 
                - last_test_score -  балл за последний сданный экзамен или тест
                - attendance_percent - процент посещаемости занятий
                - scholarship_amount - размер стипендии в рублях
                - extracurricular_activities - внеучебные активности
                - on_probation - находится ли студент на академическом испытательном сроке 
                - has_dormitory - проживает ли студент в общежитии
                - enrollment_status - текущий статус обучения студента
                - preferred_language - предпочитаемый язык для обучения или язык, на котором обучается студент
                - mentor_id - идентификатор ментора
                - average_study_hours - среднее количество часов, которое тратит на самостоятельное обучение студент
                - library_visits_per_month - среднее количество посещений библиотеки в месяц

            Оцени полноту данных. Данные являются полными, если сущность имеет достаточное кол-во атрибутов для анализа. 
            Здесь имеется ввиду, что все необходимые поля для этого присутствуют. Ненужно проверять корректность данных.
            Моя цель - понять факторы от которых зависит успех в обучении. 

            Верни итоговую оценку от 0 до 1 и комментария свои. Больше ничего не возвращай в ответ. 
        """ + str(data_edtech)

        return prompt_message

    def create_agent_chain(self, token: str) -> dict: 

        ds = DataSourceClient()
        data_edtech = ds.create_dataset()
        data_edtech = ds.dataset_from_dataframe_to_json(dataset_edtech=data_edtech)

        learning_data_prompt = self.read_dataset_edtech_data_prompt(data_edtech=data_edtech)
        fulness_data = self.send_gigachat_request_fulness(token=token, message=learning_data_prompt)['choices'][0]['message']['content']

        return fulness_data


class ValidityAiAgent:
    """ ИИ агент для оценки достоверности данных """ 

    def send_gigachat_request_validity(self, token: str, message: str) -> json:
        """Отправка запроса к GigaChat API"""
        url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}'
        }
        
        data = {
            "model": "GigaChat",
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ],
            "temperature": 0.7
        }
        
        response = requests.post(url, headers=headers, json=data, verify=False)
        return response.json()

    def read_dataset_edtech_data_prompt(self, data_edtech: json) -> str: 
        """ Получаем промпт запрос для оценки достоверности данных """
        
        prompt_message = """
            На вход ты получаешь датасет, который содержит следующие поля: 

            - student_id - уникальный идентификатор студента
            - email - электронная почта студента
            - phone_number - мобильный номер телефона студента
            - date_of_birth - дата рождения студента
            - age - возраст студента
            - admission_year - год поступления в учебное заведение
            - faculty - факультет, на котором учится студента
            - group_name - наименование группы обучения
            - gpa - средний балл студента 
            - last_test_score -  балл за последний сданный экзамен или тест
            - attendance_percent - процент посещаемости занятий
            - scholarship_amount - размер стипендии в рублях
            - extracurricular_activities - внеучебные активности
            - on_probation - находится ли студент на академическом испытательном сроке 
            - has_dormitory - проживает ли студент в общежитии
            - enrollment_status - текущий статус обучения студента
            - preferred_language - предпочитаемый язык для обучения или язык, на котором обучается студент
            - mentor_id - идентификатор ментора
            - average_study_hours - среднее количество часов, которое тратит на самостоятельное обучение студент
            - library_visits_per_month - среднее количество посещений библиотеки в месяц

            Для каждого поля оцени: возможен ли такой показатель исходя из текущей ситуации в образовании. 
            Добавь дополнительное поле is_valid в датасет и добавь туда одно из трех значений: точный, не похож 
            на правду, похож на правду. 

            Верни измененный датасет, больше ничего не возвращай.  
        """ + str(data_edtech)

        return prompt_message

    def get_validity_assessment(self, data_edtech: json) -> str: 
        """ Получаем оценку достоверности данных """

        prompt_message = """
            Оцени размеченный датасет на предмет достоверности данных. Верни оценку от 0 до 1 
            и добавь небольшие комментария.

            Оценка достоверности = (кол-во правдивых значений + точных)/(кол-во всех значений)
        """ + str(data_edtech)

        return prompt_message

    def create_agent_chain(self, token: str) -> dict: 
        """ Создаем цепочку вызовов """

        ds = DataSourceClient()
        data_edtech = ds.create_dataset()
        data_edtech = ds.dataset_from_dataframe_to_json(dataset_edtech=data_edtech)    

        # Изучаем данные 
        learning_data_prompt = self.read_dataset_edtech_data_prompt(data_edtech=data_edtech)
        learning_data = self.send_gigachat_request_validity(token=token, message=learning_data_prompt)['choices'][0]['message']['content']

        # Оцениваем достоверность данных
        validity_score_prompt = self.get_validity_assessment(data_edtech=learning_data)
        validity_score = self.send_gigachat_request_validity(token=token, message=validity_score_prompt)['choices'][0]['message']['content']

        return {
            "learning_data": learning_data, 
            "validity_score": validity_score
        }



if __name__ == '__main__': 
        
    # читаем данные 
    ds = DataSourceClient()
    data_edtech = ds.create_dataset()
    data_edtech_json = ds.dataset_from_dataframe_to_json(dataset_edtech=data_edtech)

    # получаем токен доступа к API GigaChat 
    ai_model = AiModelClient()
    token = ai_model.get_token_auth()

    # определяем точность данных 
    # precision_agent = PrecisionAiAgent()
    # data_result_accuracy = precision_agent.create_agent_chain(token=token)

    # определяем полноты данных 
    # fulness_agent = FulnessAiAgent()
    # data_result_fulness = fulness_agent.create_agent_chain(token=token)

    # определяем достоверность данных
    validity_agent = ValidityAiAgent()
    validity_agent_result = validity_agent.create_agent_chain(token=token)

    print(validity_agent_result['validity_score'])
