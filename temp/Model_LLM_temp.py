from huggingface_hub import login
import gc
import torch
from typing import Optional, List, Dict, Any
from langchain.llms.base import LLM  # ✅ Наследуем LangChain LLM
from transformers import BitsAndBytesConfig, AutoProcessor, AutoModelForPreTraining
from PIL import Image
import os
from pydantic import Field, PrivateAttr

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

class CustomLLM(LLM):  # ✅ Теперь это полноценный LangChain LLM
    model_name: str = Field(default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=650)
    top_k: int = Field(default=50)
    top_p: float = Field(default=0.95)

    _processor: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct", device: str = "cuda", **kwargs):
        """Конструктор, совместимый с LangChain"""
        super().__init__(**kwargs)  # ✅ Передаем параметры в родительский LLM
        self._processor = AutoProcessor.from_pretrained(model_name, device_map=device, trust_remote_code=True, cache_dir=cache_dir)
        self._model = AutoModelForPreTraining.from_pretrained(model_name, device_map=device, trust_remote_code=True, quantization_config=bnb_config, cache_dir=cache_dir)

    def _call(self, prompt: str, stop: Optional[List[str]] = None, image: Optional[Image.Image] = None, **kwargs) -> str:
        """Метод для вызова LLM, теперь совместим с LangChain"""
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

        gc.collect()
        torch.cuda.empty_cache()
        return response_text

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """✅ Теперь это свойство, а не метод"""
        return self.model_dump()  # ✅ Pydantic v2 заменяет `self.dict()` на `self.model_dump()`


    @property
    def _llm_type(self) -> str:
        """Обязательный метод для LangChain LLM"""
        return "custom_llm"

# ✅ Теперь `CustomLLM` полностью совместим с LangChain
model = CustomLLM()
