# ConvVAE model

import numpy as np
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pythae.models import AutoModel
from pythae.models.base.base_utils import ModelOutput
from VAE.utils import imshow_tensor


class VAE:
    def __init__(self, model_root_dir, visualize = False):
        self.model_root_dir = model_root_dir
        self.visualize = visualize
        self.type = ["img", "pcd", "arr"]
        self.models = {"img": self.load_model("img"), \
                       "pcd": self.load_model("pcd"), \
                        "arr": self.load_model("arr")}

    def load_model(self, model_type):
        assert model_type in self.type, f"unsupported model type {model_type}"
        models = {}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        type_file = model_type + "_model"
        model_path = os.path.join(self.model_root_dir, type_file)
        model = AutoModel.load_from_folder(model_path).to(device)
        models["model_name"] = model.model_name
        models["model"] = model
        models["encoder"] = model.encoder
        models["decoder"] = model.decoder
        
        return models
        

    def reconstruct(self, dataset, type, visul_num = 5):
        assert type in self.type, f"unsupported model type {type}"
        model = self.models[type]["model"]
        #print(f"dataset shape {dataset.shape}")

        length = len(dataset) 
        col = visul_num # math.ceil(length / 5.0)
        row = 2
        rescontructions = model.reconstruct(dataset[:length]).detach().cpu()

        if self.visualize:
            fig, axes = plt.subplots(nrows=row, ncols=col, figsize=(15, 6))
            for i in range(visul_num):
                if i >= length:
                    break
                imshow_tensor(axes[0][i], rescontructions[i])
                imshow_tensor(axes[1][i], dataset[i])
            
            fig.suptitle("Reconstructions VS Orignal", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95]) 
            plt.show()
        
        return rescontructions
    
    def embedding(self, inputs: torch.Tensor, type):
        assert type in self.type, f"unsupported model type {type}"
        model = self.models[type]["model"]

        outputs = model.embed(inputs)    # inputs: [Batch * input_dim]  # self.encoder(inputs).embedding
        return outputs                   # outputs: [Batch * latent_dim]


    def decode(self, z, type):
        assert type in self.type, f"unsupported model type {type}"
        model = self.models[type]["model"]
        out = model.decoder(z)["reconstruction"]
        return out
    
    def predict(self, inputs : torch.Tensor) -> ModelOutput:
        """
        output = ModelOutput(
            recon_x=recon_x,
            embedding=z,
        )
        """
        assert type in self.type, f"unsupported model type {type}"
        model = self.models[type]["model"]

        output : ModelOutput = model.predict(inputs)
        return output


