import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data: np.ndarray):
        """
        初始化数据集
        :param data: 数据 NumPy 数组，原始形状为 (样本数, 513, 125)
        """
        # 将 NumPy 数组转换为 PyTorch Tensor
        self.data_tensor = torch.from_numpy(data).float()

    def __len__(self):
        """
        返回数据集的大小（样本数）
        """
        return self.data_tensor.shape[0]

    def __getitem__(self, idx):
        """
        返回指定索引的样本，并进行维度重排和通道添加。
        :param idx: 样本索引
        """
        sample = self.data_tensor[idx]
        sample = sample.unsqueeze(0)  # 增加一个通道维度，变成 (1, 125, 513)
        return sample


def loader_cre(output_dir, dataset_type='scaled'):
    """
    从指定的目录加载训练集、验证集和测试集数据，并创建 CustomDataset 实例
    :param output_dir: 数据文件存放的目录路径
    :param dataset_type: 数据文件类型 ('scaled' 或 'minmax'，用于选择数据文件)
    :return: 训练集、验证集和测试集的 CustomDataset 实例
    """
    # 根据 dataset_type 设置文件名
    if dataset_type == 'scaled':
        train_file = os.path.join(output_dir, 'train_data_scaled.npy')
        val_file = os.path.join(output_dir, 'val_data_scaled.npy')
        test_file = os.path.join(output_dir, 'test_data_scaled.npy')
    elif dataset_type == 'minmax':
        train_file = os.path.join(output_dir, 'train_data_minmax.npy')
        val_file = os.path.join(output_dir, 'val_data_minmax.npy')
        test_file = os.path.join(output_dir, 'test_data_minmax.npy')
    else:
        raise ValueError("Unsupported dataset type. Use 'scaled' or 'minmax'.")

    # 加载数据
    print(f"Loading train data from {train_file}")
    train_data_np = np.load(train_file)
    print(f"Loading validation data from {val_file}")
    val_data_np = np.load(val_file)
    print(f"Loading test data from {test_file}")
    test_data_np = np.load(test_file)

    # 创建 CustomDataset 实例
    train_dataset = CustomDataset(train_data_np)
    val_dataset = CustomDataset(val_data_np)
    test_dataset = CustomDataset(test_data_np)

    # 返回数据集
    return train_dataset, val_dataset, test_dataset
