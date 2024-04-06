from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np

def calculate_ssim(tensor1, tensor2):
    ssim_values = []
    for i in range(tensor1.size(0)):  # Iterate over the batch
        img1 = tensor1[i].cpu().detach().numpy().squeeze()  # Convert to numpy and remove channel dimension
        img2 = tensor2[i].cpu().detach().numpy().squeeze()  # Same for the second tensor

        # Calculate SSIM, assuming the images are grayscale so channel_axis is None
        ssim_value = ssim(img1, img2, data_range=img1.max() - img1.min())
        ssim_values.append(ssim_value)

    return np.mean(ssim_value)