import os
import numpy as np
from scipy import signal

SAMPLING_RATE = 2500   #采样频率
NPERSEG = 1024 
NFFT = NPERSEG         # 默认为 nperseg，决定了频率点 M
OVERLAP = NPERSEG // 2 # 50% 重叠是常用设置

def calculate_energy(data, window_size):
    """计算能量：对每个窗口内的幅度平方求和"""
    energy = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i+window_size]
        energy.append(np.sum(window**2))  # 每一帧幅度的平方和
    return np.array(energy)

def find_max_energy_segment(data, window_size):
    """寻找能量最大的窗口时间段"""
    energy = calculate_energy(data, window_size)
    max_energy_index = np.argmax(energy)  # 找到能量最大的窗口索引
    start_index = max_energy_index  # 窗口的起始索引
    end_index = max_energy_index + window_size  # 窗口的结束索引
    return data[start_index:end_index], (start_index, end_index)

# 示例数据（假设data是一个长度为200000的数组）
data = np.random.rand(200000)  # 示例数据，你可以替换成你的实际数据

window_size = 2500  # 设置窗口大小为2500
max_energy_segment, (start_idx, end_idx) = find_max_energy_segment(data, window_size)

def compute_csm(data_matrix, fs, nperseg, overlap):#计算复数？

    data_matrix = data_matrix.T #转置
    num_channels = data_matrix.shape[0] #测点数
    
    # 预计算，获取频率点 M
    freqs, Pxy_dummy = signal.csd(
        data_matrix[0, :], data_matrix[0, :], fs=fs, nperseg=nperseg, noverlap=overlap, nfft=NFFT
    )
    M = len(freqs)
    
    # 初始化 CSM 矩阵 (125 x 125 x M)，存储复数结果
    csm_matrix = np.zeros((num_channels, num_channels, M), dtype=np.complex64)

    # 循环计算所有测点对的互谱
    for i in range(num_channels):
        for j in range(i, num_channels): # 只需要计算上三角，因为 P_ji = P*_ij
            
            # 使用 Welch 法计算互谱密度 (CSD)
            f, Pxy = signal.csd(
                data_matrix[i, :], data_matrix[j, :], 
                fs=fs, nperseg=nperseg, noverlap=overlap, nfft=NFFT
            )
            
            csm_matrix[i, j, :] = Pxy
            
            # 利用共轭对称性 P_ji = P*_ij
            if i != j:
                csm_matrix[j, i, :] = np.conjugate(Pxy)
 
    reference_matrix = np.mean(csm_matrix,axis=0) #[:, 0, :]result = np.mean(data, axis=0)
    reference_matrix = abs(reference_matrix)
    reference_matrix = np.log(reference_matrix).T
    return reference_matrix

data_root = r'H:\Storage\Airport\BJ'
output_root = r'D:\Private\xiaoan\autoencoder\val_1018'

date_folders = sorted(os.listdir(data_root))[401:500]  # 选取前150个日期文件夹
file_name = f"Slab1.npy"
file_counter = 401

for date_folder in date_folders:
    date_folder_path = os.path.join(data_root, date_folder)
    file_path = os.path.join(date_folder_path, file_name)
    if os.path.exists(file_path):
        data = np.load(file_path)#读取数据    
        data_2 = data[peak_index - 1250 : peak_index + 1250, :] #step1 裁时间帧
        data_3 = column125(data_2) #step2 裁测点
        data_4 = compute_csm(data, SAMPLING_RATE, NPERSEG, OVERLAP) #step3 算cpsd

        output_file = os.path.join(output_root, f'{file_counter:03}.npy')
        np.save(output_file, data_4)
        
