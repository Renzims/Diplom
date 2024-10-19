from langchain.prompts import ChatPromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline, BitsAndBytesConfig
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
import torch
from langchain_huggingface.llms import HuggingFacePipeline
import os
from huggingface_hub import login

os.environ["HF_HOME"] = "D:/Diplom/New folder/models"
cache_dir = "D:/Diplom/New folder/models"
huggingfacehub_api_token = "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"
login(huggingfacehub_api_token)
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


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if tokenizer.decode(stop) == tokenizer.decode(last_token):
                return True
        return False

stop_words = [tokenizer.decode(tokenizer.eos_token_id)]
stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

# Создаём pipeline с использованием модели и токенизатора
'''
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=400,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
)
'''
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=400,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    stopping_criteria=stopping_criteria
)

# Интеграция с LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Создаём шаблон для диалогов
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant who provides clear and concise answers to the user's questions. Also, you should not change the user's question in any way."),
        ("user", "Q:{question}: <|eot_id|>"),
    ]
)

# Создаём цепочку
chain = prompt_template | llm
