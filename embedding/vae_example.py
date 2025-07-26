import os
import torch
from vae import VAE
from VAE.v2x_sim_mini_proc import load_bin, load_image, load_npz

model_dir = os.path.join("VAE", "my_models")

vae = VAE(model_root_dir=model_dir, visualize=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

input_file = "/home/peh324/Codes/V2X-Sim-2.0-mini/v2x_sim_2.0_mini/sweeps/CAM_FRONT_id_1/scene_5_000006.jpg"
input = load_image(input_file) / 255.0
input = torch.from_numpy(input).unsqueeze(0).float().to(device)

recon_tensor = vae.reconstruct(input, "img")
print(f"reconstruction shape {recon_tensor.shape}")

embedding = vae.embedding(input, "img")
print(f"embedding shape {embedding.shape}")

decode_sensor = vae.decode(embedding, "img")
print(f"decode shape {decode_sensor.shape}")