import torch
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from pythae.models import AutoModel
from pythae.models import VAE, VAEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
from Encoder_Conv_VAE_V2X import Encoder_Conv_VAE_V2X as Encoder_VAE
from Decoder_ResNet_VAE_V2X import Decoder_ResNet_VAE_V2X as Decoder_AE   
from pythae.data.preprocessors import DataProcessor
from v2x_sim_mini_proc import load_bin, load_image, load_npz

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
# PATH = "data/v2x_sim_mini/img_train.npz"
# eval_dataset = np.load(PATH)["data"] / 255.0  # normalize to [0, 1]
# #print(f"Original numpy dataset shape: {eval_dataset.shape}")

one_Path = "/home/peh324/Codes/V2X-Sim-2.0-mini/v2x_sim_2.0_mini/sweeps/CAM_FRONT_id_1/scene_5_000006.jpg"
eval_dataset = load_image(one_Path) / 255.0
eval_dataset = torch.from_numpy(eval_dataset).unsqueeze(0).float().to(device)
#print(eval_dataset.shape)

#print(f"Torch tensor dataset shape: {eval_dataset.shape}")

# Load trained model
model_dir = Path("my_models_on_mnist/VAE_training_2025-05-23_12-17-17/final_model")
trained_model = AutoModel.load_from_folder(model_dir).to(device)
#print("Loaded model:")
#print(trained_model)

# Reconstruct images
reconstructions = trained_model.reconstruct(eval_dataset).detach().cpu()
#print(f"Reconstructed image tensor shape: {reconstructions.shape}")

# Display utility
def imshow_tensor(ax, tensor):
    img = tensor.detach().cpu()
    if img.shape[0] == 1:
        img = img.squeeze(0)
        ax.imshow(img, cmap='gray')
    elif img.shape[0] == 3:
        img = img.permute(1, 2, 0)
        ax.imshow(img)
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")
    ax.axis('off')

# Show reconstructions
fig, ax = plt.subplots(figsize=(4, 4))  
imshow_tensor(ax, reconstructions[0])   
ax.axis('off')  
plt.title("VAE Reconstruction", fontsize=16)
plt.tight_layout()
plt.show()


# Show original images
fig, ax = plt.subplots(figsize=(4, 4))  
imshow_tensor(ax, eval_dataset[0])   
ax.axis('off')  
plt.suptitle("Original Images", fontsize=16)
plt.tight_layout()
plt.show()
