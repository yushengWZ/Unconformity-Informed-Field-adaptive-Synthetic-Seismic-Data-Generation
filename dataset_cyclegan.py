import torch
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
import random
import cv2


def map_values(data, src_range=(0, 0.4), dst_range=(0.2, 0.4)):
    data_mapped = data.copy()  # 复制数据以避免修改原始数组
    mask = (data_mapped > src_range[0]) & (data_mapped < src_range[1])  # 布尔掩码
    if np.any(mask):  # 确保存在满足条件的值
        min_val = np.min(data_mapped[mask])  # 源范围的最小值
        max_val = np.max(data_mapped[mask])  # 源范围的最大值
        if max_val > min_val:  # 避免除以零
            # 线性映射：(x - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
            data_mapped[mask] = (data_mapped[mask] - min_val) / (max_val - min_val) * (dst_range[1] - dst_range[0]) + \
                                dst_range[0]
        else:
            # 如果所有值相等，设为目标范围起点
            data_mapped[mask] = dst_range[0]
    return data_mapped


def percentile_normalize(data, min_percentile=1, max_percentile=99):
    min_clip = np.percentile(data, min_percentile)
    max_clip = np.percentile(data, max_percentile)
    normalized_data = (data - min_clip) / (max_clip - min_clip)
    return np.clip(normalized_data, 0, 1).astype(np.float32)


def histogram_equalization(data):
    H, W, D = data.shape
    equalized_data = np.zeros_like(data, dtype=np.float32)

    # 对每张图像应用直方图均衡化
    for d in range(D):
        # 提取单张图像
        img = data[:, :, d]

        # 计算直方图
        hist, bins = np.histogram(img.ravel(), bins=256, range=[0, img.max()])
        cdf = hist.cumsum()  # 累积分布函数 (CDF)
        cdf = cdf / cdf[-1]  # 归一化 CDF 到 [0, 1]

        # 插值实现均衡化
        equalized_img = np.interp(img.ravel(), bins[:-1], cdf * (img.max() - img.min()) + img.min())
        equalized_img = equalized_img.reshape(img.shape)

        # 归一化到 [0, 1]
        equalized_img = (equalized_img - equalized_img.min()) / (equalized_img.max() - equalized_img.min())

        # 存储结果
        equalized_data[:, :, d] = equalized_img

    return equalized_data


def get_neighbors(img, i, j):
    """
    获取像素 (i, j) 的 8 邻域值，按顺时针顺序 P2 到 P9。
    """
    H, W = img.shape
    neighbors = []
    # P2 到 P9 顺序：(i-1,j), (i-1,j+1), (i,j+1), (i+1,j+1), (i+1,j), (i+1,j-1), (i,j-1), (i-1,j-1)
    coords = [(i - 1, j), (i - 1, j + 1), (i, j + 1), (i + 1, j + 1), (i + 1, j), (i + 1, j - 1), (i, j - 1),
              (i - 1, j - 1)]
    for ni, nj in coords:
        if 0 <= ni < H and 0 <= nj < W:
            neighbors.append(img[ni, nj])
        else:
            neighbors.append(0)
    return neighbors


def thinning_iteration(img, iter):
    """
    Zhang-Suen 细化算法的单次迭代。
    iter=0: 删除满足条件的像素（奇数步）
    iter=1: 删除满足条件的像素（偶数步）
    """
    H, W = img.shape
    marker = np.zeros((H, W), dtype=np.uint8)
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if img[i, j] == 0:
                continue
            # 获取 8 邻域
            p2, p3, p4, p5, p6, p7, p8, p9 = get_neighbors(img, i, j)
            # 条件 A: 2 <= N(P1) <= 6（邻域中 1 的数量）
            N = sum([p2, p3, p4, p5, p6, p7, p8, p9])
            if not (2 <= N <= 6):
                continue
            # 条件 B: S(P1) = 1（0 到 1 的转换次数）
            S = sum([(p2 == 0 and p3 == 1), (p3 == 0 and p4 == 1), (p4 == 0 and p5 == 1),
                     (p5 == 0 and p6 == 1), (p6 == 0 and p7 == 1), (p7 == 0 and p8 == 1),
                     (p8 == 0 and p9 == 1), (p9 == 0 and p2 == 1)])
            if S != 1:
                continue
            # 条件 C 和 D（奇数步）
            if iter == 0:
                if (p2 * p4 * p6) != 0 or (p4 * p6 * p8) != 0:
                    continue
            # 条件 C 和 D（偶数步）
            else:
                if (p2 * p4 * p8) != 0 or (p2 * p6 * p8) != 0:
                    continue
            marker[i, j] = 1
    img[marker == 1] = 0
    return img


def thinning(img):
    """
    使用 Zhang-Suen 算法进行骨架化，直到没有像素被移除。
    """
    img = img.copy()
    while True:
        prev = img.copy()
        # 奇数步
        img = thinning_iteration(img, 0)
        # 偶数步
        img = thinning_iteration(img, 1)
        # 如果没有变化，退出
        if np.array_equal(img, prev):
            break
    return img


def refine_fault_to_one_pixel_numpy(source_data, fault_data):
    """
    逐张切片将 fault 数据中的断层宽度细化为 1 像素，并扣除 source_data。

    参数：
        source_data: 输入数据，形状 [n, 1, H, W]
        fault_data: 断层数据，形状 [n, 1, H, W]

    返回：
        refined_source_data: 处理后的 source_data，形状 [n, 1, H, W]
    """
    n, _, H, W = source_data.shape
    refined_source_data = source_data.copy()

    # 逐张切片处理
    for i in range(n):
        # 提取单张切片
        source_slice = source_data[i, 0, :, :]  # [128, 128]
        fault_slice = fault_data[i, 0, :, :]  # [128, 128]

        # 将 fault 数据转换为二值图像（非零值为 1）
        fault_binary = (fault_slice > 0).astype(np.uint8)

        # 骨架化处理，将宽度细化为 1 像素
        thinned_fault = thinning(fault_binary)

        # 转换为与原 fault_data 匹配的格式
        refined_fault_slice = thinned_fault.astype(np.float32)

        # 直接扣除：将细化后的断层区域置 0
        fault_mask = refined_fault_slice > 0
        source_slice[fault_mask] = 0

        # 存储处理后的切片
        refined_source_data[i, 0, :, :] = source_slice

    return refined_source_data


def rearrange_faultunconf(unconf, fault, weight):
    weight = weight
    fault[fault > 0] = 1
    output = unconf.copy()

    # 获取断层位置（假设 fault 中断层像素值为 1）
    fault_mask = fault == 1

    # 找到断层像素的索引
    fault_indices = np.where(fault_mask)

    # 遍历每一帧（3704 帧）
    for frame_idx in range(3704):
        # 获取当前帧的 fault 掩码
        frame_fault = fault[frame_idx]  # [128, 128]
        frame_unconf = unconf[frame_idx]  # [128, 128]

        # 找到当前帧中断层的位置
        fault_y, fault_x = np.where(frame_fault == 1)

        for y, x in zip(fault_y, fault_x):
            # 判断是否为断层左边像素（x 左侧无断层像素，或左侧是边界）
            if x == 0 or frame_fault[y, x - 1] != 1:
                # 左边像素：用左侧 unconf 值，乘以权重
                if x > 0:  # 确保不在边界
                    output[frame_idx, y, x] = frame_unconf[y, x - 1] * weight
                else:
                    # 如果在左边界，保持原值并乘以权重
                    output[frame_idx, y, x] = frame_unconf[y, x] * weight
            # 判断是否为断层右边像素（x 右侧无断层像素，或右侧是边界）
            elif x == 127 or frame_fault[y, x + 1] != 1:
                # 右边像素：用右侧 unconf 值，乘以权重
                if x < 127:  # 确保不在边界
                    output[frame_idx, y, x] = frame_unconf[y, x + 1] * weight
                else:
                    # 如果在右边界，保持原值并乘以权重
                    output[frame_idx, y, x] = frame_unconf[y, x] * weight
    return output


def his_dataset():
    # source_labels
    file_str = r'data_for_training\unconf_cut.npy'
    source_data = np.load(file_str).astype(np.float32)
    source_data = histogram_equalization(source_data)
    source_data = source_data.transpose([2, 0, 1])
    source_data = np.expand_dims(source_data, axis=1)
    source_data = source_data[:2000, ...]
    # target_data
    file_str = r'data_for_training\kerry_cut.npy'
    target_data = np.load(file_str).astype(np.float32).transpose([2, 0, 1])
    target_data = np.expand_dims(target_data, axis=1)
    target_data = target_data[:2000, ...]
    print(f'source{source_data.shape}\ntarget{target_data.shape}')

    for i in range(target_data.shape[0]):
        temp = target_data[i, 0, :, :]
        temp = (temp - temp.mean()) / temp.std()
        temp = (temp - temp.min()) / (temp.max() - temp.min())  # 归一化到 [0, 1]
        target_data[i, 0, :, :] = temp

    source_data = torch.from_numpy(source_data)
    target_data = torch.from_numpy(target_data)

    # data set
    data_set = Data.TensorDataset(source_data, target_data)

    return data_set


def his_dataset_SYN2KERRYori():
    # source_labels
    file_str = r'data_for_training\FFEunconformity.npy'
    source_data = np.load(file_str).astype(np.float32)
    source_data = np.expand_dims(source_data, axis=1)
    source_data = source_data[:2000, ...]

    file_str = r'data_for_training\kerry_cut.npy'
    target_data = np.load(file_str).astype(np.float32).transpose([2, 0, 1])
    target_data = np.expand_dims(target_data, axis=1)
    target_data = target_data[:2000, ...]
    print(f'source{source_data.shape}\ntarget{target_data.shape}')

    source_data = source_data[:target_data.shape[0], ...]

    for i in range(target_data.shape[0]):
        temp = target_data[i, 0, :, :]
        temp = (temp - temp.mean()) / temp.std()
        temp = (temp - temp.min()) / (temp.max() - temp.min())
        target_data[i, 0, :, :] = temp

    source_data = torch.from_numpy(source_data)
    target_data = torch.from_numpy(target_data)

    # data set
    data_set = Data.TensorDataset(source_data, target_data)

    return data_set
