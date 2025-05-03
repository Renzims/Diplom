import torch
from diffusers import StableDiffusionPipeline
from transformers import BitsAndBytesConfig
import sys
import random
cache_dir = "D:/Diplom/New folder/models"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)
def load_model(model_id):
    model = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        cache_dir= cache_dir,
    )
    model.to(device)
    return model
model_id="stabilityai/stable-diffusion-2-1"
model = load_model(model_id)