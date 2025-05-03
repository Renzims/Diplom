from huggingface_hub import login
import gc
import torch
from typing import Optional, List, Dict, Any
from langchain.llms.base import LLM
from transformers import BitsAndBytesConfig, AutoProcessor, AutoModelForPreTraining
from PIL import Image
from pydantic import Field, PrivateAttr
from langchain.prompts import PromptTemplate
import os
import shutil
import time
import uuid

SAVE_FOLDER = "D:\Diplom\pycharm\\try_2\ReAct\\temp_photo"

def generate_filename():
    timestamp = int(time.time())
    unique_id = uuid.uuid4().hex[:8]
    return f"image_{timestamp}_{unique_id}.png"

def save_image(image: Image.Image):
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    filename = generate_filename()
    file_path = os.path.join(SAVE_FOLDER, filename)
    image.save(file_path)
    return file_path

def clear_images():
    if os.path.exists(SAVE_FOLDER):
        shutil.rmtree(SAVE_FOLDER)


os.environ["HF_HOME"] = "D:/Diplom/New folder/models"
cache_dir = "D:/Diplom/New folder/models"
huggingfacehub_api_token = "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"
login(huggingfacehub_api_token)
bnb_config= BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

template=PromptTemplate(
            input_variables=["query"],
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a useful assistant who should analyze images based on the user's request.
<|eot_id|><|start_header_id|>user<|end_header_id|>
User request:
{query}

User images
<|image|>

<|eot_id|><|start_header_id|>assistant_answer_final<|end_header_id|>"""
        )

class CustomLLM(LLM):
    model_name: str = Field(default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=650)
    top_k: int = Field(default=50)
    top_p: float = Field(default=0.95)

    _processor: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct", device: str = "cuda:1", **kwargs):
        super().__init__(**kwargs)
        self._processor = AutoProcessor.from_pretrained(model_name, device_map=device, trust_remote_code=True, cache_dir=cache_dir)
        self._model = AutoModelForPreTraining.from_pretrained(model_name, device_map=device, trust_remote_code=True, quantization_config=bnb_config, cache_dir=cache_dir)

    def _call(self, prompt: str, stop: Optional[List[str]] = None, image: Optional[Image.Image] = None, **kwargs) -> str:
        inputs = self._processor(images=image, text=prompt, return_tensors="pt", padding=True) if image else self._processor(text=prompt, return_tensors="pt", padding=True)
        inputs = inputs.to(self._model.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p
        )
        response_text = self._processor.decode(outputs[0], skip_special_tokens=True)
        final_output=self.parse_assistant_response(response_text)
        gc.collect()
        torch.cuda.empty_cache()
        return final_output
    def parse_assistant_response(self,full_response: str) -> str:
        keyword = "assistant_answer_final"
        keyword_index = full_response.find(keyword)
        if keyword_index != -1:
            assistant_text = full_response[keyword_index + len(keyword):].strip()
            return assistant_text
        else:
            return full_response

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return self.model_dump()


    @property
    def _llm_type(self) -> str:
        return "custom_llm"

model = CustomLLM()
