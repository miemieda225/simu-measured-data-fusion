#检查时间间隔
import os
from datetime import datetime

# 设置父文件夹路径
parent_folder = r'D:\研三\仿真数据\结构试槽-接缝数据\20241109\data\2024-11-09'  # 替换为你的文件夹路径

# 存储每个文件的时间戳
timestamps = []
folders = []
# 遍历test1到test17文件夹
for i in range(1, 8):
    folder_path = os.path.join(parent_folder, f'test{i}')
    
    # 获取文件夹中的所有文件
    files = [f for f in os.listdir(folder_path) if f.endswith('-out.dat')]
    
    # 提取时间戳并按时间排序
    for file in files:
        # 提取时间部分 (假设文件名格式是 2024-11-08-10-22-17-out.dat)
        time_str = file.replace('-out.dat', '')  # 重构时间字符串，去掉-out.dat
        time_str = time_str.split('-')[-3:]  # 获取最后三个部分，即时间部分
        time_str = ':'.join(time_str)
        timestamp = datetime.strptime(time_str, "%H:%M:%S")  # 转换为datetime对象
        timestamps.append(timestamp)
        folders.append(f'test{i}')

# 检查时间间隔是否为4秒
for i in range(1, len(timestamps)):
    time_diff = (timestamps[i] - timestamps[i - 1]).seconds
    if time_diff != 4:
        print(f"在文件夹 {folders[i-1]} 中文件 {timestamps[i-1]} 和 {timestamps[i]} 之间的间隔不是 4 秒，实际间隔为 {time_diff} 秒。")
