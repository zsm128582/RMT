import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider

def showFFt(inputdata):
    fft_reault = np.fft.fft2(inputdata , axes=(2,3))
    # 获取频谱的幅度 (可选)
    fft_magnitude = np.abs(fft_reault)
    fft_magnitude_shifted = np.fft.fftshift(fft_magnitude, axes=(2, 3))

    # 如果需要频谱的相位
    fft_phase = np.angle(fft_reault)


    plt.imshow(np.log(fft_magnitude_shifted[0, 0, :, :, 12] + 1), cmap='viridis')
    plt.title("Centered Frequency Spectrum (Batch 0, Head 0, Channel 0)")
    plt.colorbar(label="Log Magnitude")
    plt.show()

input_way="/home/zengshimao/code/RMT/classfication_release/testFFT/module_input.npy"
output_way="/home/zengshimao/code/RMT/classfication_release/testFFT/module_output.npy"

inputdata = np.load(input_way)
outputdata = np.load(output_way)

# 计算 FFT 和中心化幅度谱
def compute_fft(data):
    fft_result = np.fft.fft2(data, axes=(2, 3))  # 对 H, W 维度进行 2D FFT
    fft_magnitude = np.abs(fft_result)           # 计算幅度谱
    fft_magnitude_shifted = np.fft.fftshift(fft_magnitude, axes=(2, 3))  # 零频率移到中心
    return fft_magnitude_shifted

B,heads,H,W,C = inputdata.shape
# 预计算 FFT 结果
fft_before = compute_fft(inputdata)
fft_after = compute_fft(outputdata)

# 可视化函数
def plot_spectrum(batch, head, channel):
    # 创建画布，左右两幅子图
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 编码前频谱图
    im_before = axes[0].imshow(np.log(fft_before[batch, head, :, :, channel] + 1), cmap='viridis')
    axes[0].set_title(f"before\nBatch: {batch}, Head: {head}, Channel: {channel}")
    axes[0].axis('off')  # 隐藏坐标轴
    plt.colorbar(im_before, ax=axes[0])
    
    # 编码后频谱图
    im_after = axes[1].imshow(np.log(fft_after[batch, head, :, :, channel] + 1), cmap='viridis')
    axes[1].set_title(f"after\nBatch: {batch}, Head: {head}, Channel: {channel}")
    axes[1].axis('off')  # 隐藏坐标轴
    plt.colorbar(im_after, ax=axes[1])
    
    plt.tight_layout()
    plt.show()

# 添加交互式控件
interact(plot_spectrum,
         batch=IntSlider(min=0, max=B-1, step=1, value=0, description='Batch'),
         head=IntSlider(min=0, max=heads-1, step=1, value=0, description='Head'),
         channel=IntSlider(min=0, max=C-1, step=1, value=0, description='Channel'))

# # 读取 .npy 文件
# data = np.load('文件路径.npy')

# # 打印数据
# print(data)

