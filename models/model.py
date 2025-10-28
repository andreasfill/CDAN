import os
from os.path import join

import torch
from torch import cat
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torchvision.models as models

from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from models.base import BaseModel
from utils.post_processing import enhance_color, enhance_contrast

from einops import rearrange


class Model(BaseModel):
    def __init__(self, network, **kwargs):
        """Must to init BaseModel with kwargs."""
        super(Model, self).__init__(**kwargs)

        self.network = network.to(self.device)
        self.optimizer = Adam(self.network.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def composite_loss(self, outputs, targets):
        """Computes the composite loss by combining L2 loss and perceptual (VGG) loss."""
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:20].to(self.device)
        perceptual_loss_weight = 0.25
        loss = self.criterion(outputs, targets)
        perceptual_loss = perceptual_loss_weight * F.mse_loss(vgg(outputs), vgg(targets))

        return loss + perceptual_loss
    
    def generate_output_images(self, outputs, names, save_dir):
        """Generates and saves output images to the specified directory."""
        # Turn list of tensors, where each tensor is for one patch (CHW) into a single
        # tensor again (BCHW).
        outputs = cat(outputs)
        outputs = rearrange(outputs, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=self.dataset.num_vertical_patches, b2=self.dataset.num_horizontal_patches, h=self.dataset.patch_size, w=self.dataset.patch_size)

        os.makedirs(save_dir, exist_ok=True)
        # for output_image, image_name in zip(outputs, names):
        output_image = outputs.detach().cpu().permute(1, 2, 0).numpy()
        # output_image = outputs.detach().cpu().permute(2, 1, 0).numpy()
        output_image = (output_image * 255).astype(np.uint8)
        # output_image = Image.fromarray(output_image)

        output_path = join(save_dir, f"{self.dataset.patch_size}_{names[0]}") 

        # output_image.save(output_path)
        cv.imwrite(output_path, output_image) 

        fig, sp = plt.subplots(nrows=1, ncols=3, figsize=((30, 10)), layout="constrained")
        for i, color_channel in enumerate(["Red", "Green", "Blue"]):
            color_bar = sp[i].imshow(output_image[:, :, i], cmap="viridis", vmin=0, vmax=255)
            sp[i].set_title(f"{color_channel} Color Channel")
            sp[i].axis("off")
            fig.colorbar(color_bar, ax=sp[i], shrink=0.65, label="Pixel Intensity")

        plt.savefig(join(save_dir, f"{self.dataset.patch_size}_ColorChannels_{names[0]}"))
        
        print(f'{len(names)} output images generated and saved to {output_path}')

        # os.makedirs(save_dir, exist_ok=True)
        # for i, output_image in enumerate(outputs):
        #     output_image = output_image.detach().cpu().permute(1, 2, 0).numpy()
        #     output_image = (output_image * 255).astype(np.uint8)
        #     output_image = Image.fromarray(output_image)

        #     output_path = os.path.join(save_dir, f'output_{i + 1}.png')

        #     output_image.save(output_path)
        # print(f'{len(outputs)} output images generated and saved to {save_dir}')


    def train_step(self):
        """Trains the model."""
        train_losses = np.zeros(self.epoch)
        best_loss = float('inf')
        self.network.to(self.device)

        for epoch in range(self.epoch):
            train_loss = 0.0
            dataloader_iter = tqdm(
                self.dataloader, desc=f'Training... Epoch: {epoch + 1}/{self.epoch}', total=len(self.dataloader))
            for inputs, targets in dataloader_iter:
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.network(inputs)
                loss = self.composite_loss(outputs, targets)
                train_loss += loss.item()

                loss.backward()
                self.optimizer.step()

                dataloader_iter.set_postfix({'loss': loss.item()})

            train_loss = train_loss / len(self.dataloader)

            if train_loss < best_loss:
                best_loss = train_loss
                self.save_model(self.network)

            train_losses[epoch] = train_loss

            print(f"Epoch [{epoch + 1}/{self.epoch}] Train Loss: {train_loss:.4f}")


    def test_step(self):
        """Test the model."""
        # path = os.path.join(self.model_path, self.model_name)
        # self.network.load_state_dict(torch.load(path))
        self.network.eval()

        psnr = PeakSignalNoiseRatio().to(self.device)
        ssim = StructuralSimilarityIndexMeasure().to(self.device)
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(self.device)

        with torch.no_grad():
            test_loss = 0.0
            test_psnr = 0.0
            test_ssim = 0.0
            test_lpips = 0.0
            self.network.eval()
            self.optimizer.zero_grad()
            if self.is_dataset_paired:
                for inputs, targets in tqdm(self.dataloader, desc='Testing...'):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    outputs = self.network(inputs)
                    if self.apply_post_processing:
                        outputs = enhance_contrast(outputs, contrast_factor=1.12)
                        outputs = enhance_color(outputs, saturation_factor=1.35)
                    loss = self.criterion(outputs, targets)
                    test_loss += loss.item()
                    test_psnr += psnr(outputs, targets)
                    test_ssim += ssim(outputs, targets)
                    test_lpips += lpips(outputs, targets)
            else:
                for image_names, inputs in tqdm(self.dataloader, desc='Testing...'):
                    # Remove leading dimension of size 1 added by Pytorch Dataloader.
                    # The number of patches is the batch size.
                    if self.dataset.create_patches:
                        img_outputs = list()
                        inputs = inputs.squeeze(dim=0)
                        for img in tqdm(inputs):
                            img = img.unsqueeze(dim=0)
                            img = img.to(self.device)
                            img_outputs.append(self.network(img))
                        outputs = img_outputs
                    else:
                        inputs = inputs.to(self.device)
                        outputs = self.network(inputs)
                    names = image_names
                    break

            test_loss = test_loss / len(self.dataloader)
            test_psnr = test_psnr / len(self.dataloader)
            test_ssim = test_ssim / len(self.dataloader)
            test_lpips = test_lpips / len(self.dataloader)

            if self.is_dataset_paired:
                print(
                    f'Test Loss: {test_loss:.4f}, Test PSNR: {test_psnr:.4f}, Test SSIM: {test_ssim:.4f}, Test LIPIS: {test_lpips:.4f}')

            self.generate_output_images(outputs, names, self.output_images_path)
