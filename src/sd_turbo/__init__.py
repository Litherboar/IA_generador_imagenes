import torch
from diffusers import AutoPipelineForText2Image

print("cargando modelo...")

modelo = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float32
    #float 32 o 16 me indica la cantidad de
    #decimales con el cual el modelo creara
    #la imagen, si mi pc genera una iamgen negra paso a 16
    )

modelo = modelo.to("cpu")
# cuda - nvidia
# mps - mac com procesadores m 1 . 2 . 3
# xpu - grafica intel iris#

prompt = input("Escribe el prompt de la imagen: ")
negative_prompt = "low quality, worst quality, blurry, jpeg artifacts, bad anatomy, bad hands, extra fingers, mutated hands, disfigured face, deformed eyes, long neck, watermark, text, logo, signature"



print("Generando la imagen...")

imagen = modelo(
    prompt=prompt,
    num_inference_steps=10, # Numero de veces que corregira o creara la imagen
    guidance_scale=2.0, #que tan fiel sera al prompt
    height=1024,
    width= 1024,     
).images[0]

imagen.save("imagen.png")
print("Imagen guardada !")
