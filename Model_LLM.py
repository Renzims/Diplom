from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline, BitsAndBytesConfig,AutoProcessor, AutoModelForPreTraining
import torch
from langchain_huggingface.llms import HuggingFacePipeline
import os
from huggingface_hub import login
from langchain_core.output_parsers import BaseOutputParser
import gc
from typing import Optional, List, Union
#from langchain.llms.base import BaseLLM
from PIL import Image
from pymongo import MongoClient

os.environ["HF_HOME"] = "D:/Diplom/New folder/models"
cache_dir = "D:/Diplom/New folder/models"
huggingfacehub_api_token = "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"
login(huggingfacehub_api_token)
'''
model_name="meta-llama/Llama-3.1-8B-Instruct"
bnb_config= BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name,cache_dir=cache_dir,quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)
model.generation_config.pad_token_id = tokenizer.eos_token_id

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=650,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
)

# Интеграция с LangChain
llm = HuggingFacePipeline(pipeline=pipe)
template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant who provides clear and concise answers to the user's questions.
You will be provided with a chat history with user:
{history}

<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
class OutputParser(BaseOutputParser[str]):
    def parse(self, response: str) -> str:
        """
        Извлекает финальный ответ из строки, предполагая, что ответ начинается с "assistant<|end_header_id|>"
        """
        answer_start = response.rfind("assistant<|end_header_id|>")
        if answer_start != -1:
          return response[answer_start + len("assistant<|end_header_id|>"):]
        return response

    @property
    def _type(self) -> str:
        return "str_output_parser"
prompt_template = PromptTemplate(input_variables=["history", "query"], template=template)


class ChatHistory:
    def __init__(self, max_records=5):
        self.messages = []  # Список для хранения сообщений
        self.max_records = max_records  # Максимальное количество записей

    def add_message(self, role: str, content: str):
        """Добавляет сообщение и поддерживает максимальное количество записей"""
        # Добавляем новое сообщение в конец списка
        self.messages.append((role, content))

        # Если длина списка превышает допустимое количество записей, удаляем старые
        if len(self.messages) > self.max_records * 2:  # Умножаем на 2, т.к. пара "User -> Assistant"
            self.messages = self.messages[-self.max_records * 2:]

    def get_history(self) -> str:
        """Возвращает историю сообщений в формате строк"""
        return "\n".join([f"{role}: {content}" for role, content in self.messages])


# Инициализация истории
chat_history = ChatHistory()
# Создание LLMChain
chain = prompt_template | llm | OutputParser()

# Функция для извлечения финального ответа

# Функция для взаимодействия с моделью
def ask_question(question):
    # Получаем текущую историю
    history = chat_history.get_history()

    # Получаем ответ от модели
    response = chain.invoke({'history': history, 'query': question})
    # Извлекаем только финальный ответ
    #final_answer = extract_final_answer(response)

    # Добавляем вопрос и ответ в историю
    chat_history.add_message("User", question)
    chat_history.add_message("Assistant", response)
    gc.collect()
    torch.cuda.empty_cache()
    return response
'''
class ChatHistory:
    def __init__(self, max_records=5):
        self.messages = []  # Список для хранения сообщений
        self.max_records = max_records  # Максимальное количество записей

    def add_message(self, role: str, content: str):
        """Добавляет сообщение и поддерживает максимальное количество записей"""
        # Добавляем новое сообщение в конец списка
        self.messages.append((role, content))

        # Если длина списка превышает допустимое количество записей, удаляем старые
        if len(self.messages) > self.max_records * 2:  # Умножаем на 2, т.к. пара "User -> Assistant"
            self.messages = self.messages[-self.max_records * 2:]

    def get_history(self) -> str:
        """Возвращает историю сообщений в формате строк"""
        return "\n".join([f"{role}: {content}" for role, content in self.messages])
    def clear(self):
        self.messages=[]
class OutputParser(BaseOutputParser[str]):
    def parse(self, response: str) -> str:
        answer_start = response.rfind("\nassistant\n")
        if answer_start != -1:
          return response[answer_start + len("\nassistant\n"):]
        return response

    @property
    def _type(self) -> str:
        return "str_output_parser"

bnb_config= BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
class CustomLLM:
  def __init__(self, model_name, device = 'auto',
               bnb_config=bnb_config,
               chat_history = ChatHistory(),
               parser=OutputParser(),
               db_uri="mongodb://localhost:27017/",
               db_name="chatbot_database",
               collection_name="chat_history"):
    self.processor = AutoProcessor.from_pretrained(model_name, device_map=device, trust_remote_code=True,cache_dir=cache_dir)
    self.model = AutoModelForPreTraining.from_pretrained(model_name, device_map=device, trust_remote_code=True,quantization_config=bnb_config,cache_dir=cache_dir)
    self.chat_history = chat_history
    self.parser=parser
    self.generation_params = {
    'max_new_tokens': 650,
    'temperature': 0.7,
    'top_k': 50,
    'top_p': 0.95
}
    self.template_text = PromptTemplate(
            input_variables=["history", "query"],
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant who provides clear and concise answers to the user's questions.
You will be provided with a chat history with user:
{history}

<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        )
    self.template_image = PromptTemplate(
            input_variables=["history", "query"],
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant who provides clear and concise answers to the user's questions.
You will be provided with a chat history with user:
{history}

<|eot_id|><|start_header_id|>user<|end_header_id|>

<|image|>
{query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        )
    self.db_uri = db_uri
    self.db_name = db_name
    self.collection_name = collection_name
  def custom_response_function(self, question: str, image: Optional[Image.Image] = None) -> str:
    if image:
      inputs = self.processor(images=image, text=question, return_tensors="pt", padding=True)
    else:
      inputs = self.processor(text=question, return_tensors="pt", padding=True)
    inputs = inputs.to(self.model.device)
    outputs = self.model.generate(**inputs,**self.generation_params)
    response_text = self.processor.decode(outputs[0],skip_special_tokens=True)
    return response_text

  def fetch_chat_history_from_mongo(self, max_records=5):
      """Получает последние 5 пар запросов и ответов из MongoDB."""
      client = MongoClient(self.db_uri)
      db = client[self.db_name]
      collection = db[self.collection_name]

      user_messages = reversed(list(collection.find({"role": "user"}).sort("timestamp", -1).limit(max_records)))
      assistant_messages = reversed(list(collection.find({"role": "assistant"}).sort("timestamp", -1).limit(max_records)))

      # Добавляем пары "User -> Assistant" в ChatHistory
      self.chat_history.messages.clear()  # Очищаем старую историю перед добавлением новых
      for user_msg, assistant_msg in zip(user_messages, assistant_messages):
          self.chat_history.add_message(user_msg['role'], user_msg['content'])
          self.chat_history.add_message(assistant_msg['role'], assistant_msg['content'])

  def _call(self, prompt: str, image: Optional[Image.Image] = None, stop: Optional[List[str]] = None) -> str:

    self.fetch_chat_history_from_mongo()

    selected_template = self.template_image if image else self.template_text
    formatted_prompt = selected_template.format(history=self.chat_history.get_history(), query=prompt)
    response=self.custom_response_function(formatted_prompt, image=image)
    response=self.parser.parse(response)
    self.chat_history.add_message("User", prompt)
    self.chat_history.add_message("Assistant", response)
    gc.collect()
    torch.cuda.empty_cache()
    return response
  def _generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:
    return self._call(prompt, stop=stop)
  @property
  def _identifying_params(self):
    return {"model_name": "Custom LLM with Vision"}
  @property
  def _llm_type(self) -> str:
    return "custom_llm_with_vision"
model=CustomLLM("meta-llama/Llama-3.2-11B-Vision-Instruct",device="cuda")