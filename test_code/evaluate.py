import scipy.io as scio
import numpy as np
from math import log10
from skimage.metrics import structural_similarity as ssim
from sklearn.preprocessing import normalize
import os


class Metric(object):
    def __init__(self, hsi_image, recon_image):
        self.oringin = hsi_image
        self.reconstruct = recon_image
        self.mse_value = 0
        self.rame_value = 0
        self.ssim_value = 0
        self.psnr_value = 0
        self.sam_value = 0

    def psnr(self, original_image, approximation_image):
        return 20*log10(np.amax(original_image)) - 10*log10(pow(np.linalg.norm(original_image - approximation_image), 2)
                                                            / approximation_image.size)

    # MSE: 均方误差
    def mse(self, x, y):
        return np.mean((x - y) ** 2)

    # SSIM: 结构相似性指数
    def ssim_index(self, x, y):
        # SSIM 计算时需要确保输入为灰度图像，即二维数组
        return ssim(x, y, data_range=y.max() - y.min(), multichannel=True)

    # SAM: 光谱角度映射 (Spectral Angle Mapper)
    def sam(self, x, y):
        # 计算每个像素的光谱角度
        dot_product = np.sum(x * y, axis=2)
        norm_x = np.linalg.norm(x, axis=2)
        norm_y = np.linalg.norm(y, axis=2)
        cos_angle = dot_product / (norm_x * norm_y)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)  # 返回的是弧度值
        angle_deg = np.degrees(angle)  # 转换为度
        return np.mean(angle_deg)

    def mrae(self, original_image, reconstructed_image):
        # 确保两个图像具有相同的维度
        if original_image.shape != reconstructed_image.shape:
            raise ValueError("The dimensions of the original and reconstructed images must match.")

        epsilon = 1e-10
        # 计算绝对误差
        absolute_errors = np.abs(original_image - reconstructed_image)
        # 计算相对绝对误差
        relative_errors = absolute_errors / (original_image + epsilon)
        # 计算平均相对绝对误差
        mask = (original_image > 0)
        mrae = np.mean(relative_errors[mask])  # 排除分母为零的情况
        return mrae

    def fit(self):
        self.psnr_value = self.psnr(self.oringin, self.reconstruct)
        self.mse_value = self.mse(self.oringin, self.reconstruct)
        self.ssim_value = self.ssim_index(self.oringin, self.reconstruct)
        self.sam_value = self.sam(self.oringin, self.reconstruct)
        self.mrae_value = self.mrae(self.oringin, self.reconstruct)
        return self.psnr_value, self.mse_value, self.ssim_value, self.sam_value, self.mrae_value

if __name__ == '__main__':
    gt_path = ''
    recon_path = ''
    gt_file = [file for file in os.listdir(gt_path) if '.mat' in file]
    recon_file = [file for file in os.listdir(recon_path) if '.mat' in file]
    for i in range(len(gt_file)):
        print(gt_file[i], recon_file[i])
        gt_hsi = np.array(scio.loadmat(os.path.join(gt_path, gt_file[i]))['recon'])
        recon_hsi = np.array(scio.loadmat(os.path.join(recon_path, recon_file[i]))['res'])
        metric = Metric(gt_hsi, recon_hsi)
        psnr_value, mse_value, ssim_value, sam_value, mrae_value = metric.fit()
        print(f'重建对象{i}之间误差为：PSNR:{psnr_value}dB, SSIM:{ssim_value}, SAM:{sam_value}')