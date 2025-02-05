import torch
from diffusers import StableDiffusionPipeline,DiffusionPipeline
from transformers import BitsAndBytesConfig
import sys
import random
cache_dir = "D:/Diplom/New folder/models"

# Проверяем, доступна ли GPU
device = "cuda:1" if torch.cuda.is_available() else "cpu"

# Конфигурация для использования 8-битных весов
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)
#'''
# Загружаем модель с использованием 8-битных весов
def load_model(model_id):
    model = StableDiffusionPipeline.from_pretrained(
        model_id,
        revision="fp16",  # Используем FP16 для снижения потребления памяти
        torch_dtype=torch.float16,
        bitsandbytes_config=bnb_config,
        cache_dir= cache_dir,
    )
    model.to(device)
    return model
model_id="stabilityai/stable-diffusion-2-1"
model = load_model(model_id)
'''

common_params = {
    "torch_dtype": torch.float16,
    "use_safetensors": True,
    "variant": "fp16",
    "cache_dir" : cache_dir,
}
device = "cuda:1"

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", **common_params)
pipe = pipe.to(device)
'''