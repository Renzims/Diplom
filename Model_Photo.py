import torch
from diffusers import StableDiffusionPipeline
from transformers import BitsAndBytesConfig

# Проверяем, доступна ли GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Конфигурация для использования 8-битных весов
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)
cache_dir = "D:/Diplom/New folder/models"
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
