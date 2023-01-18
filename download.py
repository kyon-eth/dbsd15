# In this file, we define download_model
# It runs during container build time to get model weights built into the container

import os
import torch
import requests
from diffusers import StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from huggingface_hub import HfFolder
from transformers import pipeline


def download_file(url, local_filename):
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
    return local_filename

# Download the weights
def download_model():
    # do a dry run of running models, which will download weights
    TOKEN = os.getenv('HF_AUTH_TOKEN')
    HfFolder.save_token(TOKEN)
    
    print("downloading model from civitai")
    
    model = download_file("https://civitai.com/api/download/models/4007?type=Model&format=PickleTensor", "models/model.ckpt")
    
    print("downloading model: stabilityai/sd-vae-ft-mse & runwayml/stable-diffusion-v1-5...")
    
    # model = "runwayml/stable-diffusion-v1-5"
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    pipe = StableDiffusionPipeline.from_pretrained(model, vae=vae)

    print("done")


if __name__ == "__main__":
    download_model()