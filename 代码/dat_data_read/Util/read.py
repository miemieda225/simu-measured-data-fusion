#定义这个类
import numpy as np
import os
import struct
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class dasNZ:
    def __init__(self, path):
        """
        初始化类
        Args:
            path (str): 数据文件路径
        """
        self.path = path

    def read_dat(self, s_col0=0, e_col0=None, s_row0=0, e_row0=None):
        float_size = 4  # 单精度浮点数占 4 字节
        try:
            with open(self.path, 'rb') as f:
                binary_data = f.read()

            # 解析文件头部信息（前 10 个 float 数据）
            header = struct.unpack('10f', binary_data[:40])
            start_time = "{:0>4d}-{:0>2d}-{:0>2d} {:0>2d}:{:0>2d}:{:0>2d}".format(
                int(header[0]), int(header[1]), int(header[2]),
                int(header[3]), int(header[4]), int(header[5])
            )
            fs = int(header[6])  # 采样频率
            framelen = int(header[7])  # 每帧长度
            pointnum = int(header[9])  # 测点数

            if fs == 0 and framelen == 0 and pointnum == 0:
                fs = 2500
                framelen = 10000
                pointnum = 705
                print(self.path, "文件参数错误，已自动设置为默认值。")

            # 提取数据部分
            data_start_offset = 40
            data_length = framelen * pointnum * float_size
            if len(binary_data) < data_start_offset + data_length:
                raise ValueError("文件数据长度不足，可能文件已损坏或格式不正确。")

            # 使用 numpy 解析数据
            data = np.frombuffer(binary_data[data_start_offset:], dtype=np.float32)
            data = data.reshape((pointnum, framelen)).T  # 转置为 (framelen, pointnum)

            # 截取需要的数据
            data_subset = data[s_row0:e_row0, s_col0:e_col0]
            data_subset = np.nan_to_num(data_subset)  # 填充 NaN 为 0

            return data_subset, start_time

        except struct.error as e:
            raise ValueError(f"文件解析失败，可能文件格式不正确。错误信息: {e}")
        except Exception as e:
            raise RuntimeError(f"读取文件时发生未知错误: {e}")


def peak_get(data, num=3):
    signal = data[:, 100]  # 选择第100个测点的信号
    peaks, _ = find_peaks(signal)  # 使用find_peaks函数找到局部峰值

    # 获取每个峰值的幅度
    peak_values = signal[peaks]

    # 选择最大的三个峰值及其对应的索引
    top_3_peaks_indices = np.argsort(peak_values)[-3:]  # 选择最大三个峰值的索引
    top_3_peaks_indices = top_3_peaks_indices[np.argsort(peaks[top_3_peaks_indices])]  # 按照时间顺序排序

    # 获取这三个峰值的时间和信号值
    top_3_peaks = peaks[top_3_peaks_indices]
    top_3_peak_values = signal[top_3_peaks]
    return top_3_peaks

def get_full(target_dir):
    file_names = [f for f in os.listdir(target_dir) if f.endswith('.dat')]
    file_paths = [os.path.join(target_dir, file_name) for file_name in file_names]
    fin_array = []
    for i, (file_name, file_path) in enumerate(zip(file_names, file_paths)):
        dat_array, time = dasNZ(file_path).read_dat()

        fin_array.append(dat_array)
    final_data = np.concatenate(fin_array, axis=0)
    return final_data

def cut_data(data):
    peak = peak_get(data)
    bound1 = peak[0]-2501
    bound2 = peak[2]+2501
    final_array = data[bound1:bound2,85:210]
    return final_array

def get_downsample(target_dir,de_factor = 10):
    file_names = [f for f in os.listdir(target_dir) if f.endswith('.dat')]
    file_paths = [os.path.join(target_dir, file_name) for file_name in file_names]
    
    down_array = []
    for i, (file_name, file_path) in enumerate(zip(file_names, file_paths)):
        dat_array, time = dasNZ(file_path).read_dat()
        downsampled_dat = dat_array[::de_factor, :]
        down_array.append(downsampled_dat)    
    
    final_data = np.concatenate(down_array, axis=0)
    
    return final_data

def figure_print(data):
    signal = data1[:, 100]
    peaks, _ = find_peaks(signal)  # 使用find_peaks函数找到局部峰值

    # 获取每个峰值的幅度
    peak_values = signal[peaks]

    # 选择最大的三个峰值及其对应的索引
    top_3_peaks_indices = np.argsort(peak_values)[-3:]  # 选择最大三个峰值的索引
    top_3_peaks_indices = top_3_peaks_indices[np.argsort(peaks[top_3_peaks_indices])]  # 按照时间顺序排序

    # 获取这三个峰值的时间和信号值
    top_3_peaks = peaks[top_3_peaks_indices]
    top_3_peak_values = signal[top_3_peaks]
    plt.plot(signal, label='Signal')
    plt.scatter(top_3_peaks, top_3_peak_values, color='red', label='Top 3 Impact Points', zorder=5)

    # 图例和标签
    plt.title('Top 3 Impact Points on Signal')
    plt.xlabel('Time (Frames)')
    plt.ylabel('Signal Value at 100th Point')
    plt.legend()
    plt.grid(True)

    # 显示图形
    plt.show()