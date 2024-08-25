import os
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.optim import SGD
from matplotlib import pyplot as plt
from scipy.signal import convolve

# 假设你有这些模块：KITTIDataset, PatchProvider，具体实现见项目文档
from stereo_batch_provider import KITTIDataset, PatchProvider

class StereoMatchingNetwork(nn.Module):
    def __init__(self):
        super(StereoMatchingNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.relu = nn.ReLU()

    def forward(self, X):
        X = X.permute(0, 3, 1, 2)  # 调整维度顺序: (batch_size, channels, height, width)
        features = self.conv1(X)
        features = self.relu(features)
        features = self.conv2(features)
        features = self.relu(features)
        features = self.conv3(features)
        features = self.relu(features)
        features = self.conv4(features)
        features = F.normalize(features, dim=1, p=2)
        return features.permute(0, 2, 3, 1)  # 还原维度为 (batch_size, height, width, n_features)

def add_padding(I, padding):
    """
    Adds zero padding to an RGB or grayscale image.

    Args:
        I (np.ndarray): HxW numpy array containing grayscale or RGB image
        padding (int): Number of pixels to pad on each side

    Returns:
        P (np.ndarray): Padded image
    """
    if len(I.shape) == 2:  # Grayscale image
        H, W = I.shape
        P = np.zeros((H + 2 * padding, W + 2 * padding), dtype=I.dtype)
        P[padding:-padding, padding:-padding] = I
    elif len(I.shape) == 3:  # RGB image
        H, W, C = I.shape
        P = np.zeros((H + 2 * padding, W + 2 * padding, C), dtype=I.dtype)
        P[padding:-padding, padding:-padding, :] = I
    else:
        raise ValueError("Unsupported image dimension")

    return P

def calculate_similarity_score(compute_features, Xl, Xr):
    features_left = compute_features(Xl)
    features_right = compute_features(Xr)
    score = torch.sum(features_left * features_right, dim=1)
    return score


def hinge_loss(score_pos, score_neg, label):
    loss = torch.max(0.2 + score_neg - score_pos, torch.zeros_like(score_pos))
    avg_loss = torch.mean(loss)

    # 计算相似度
    similarity = torch.stack([score_pos, score_neg], dim=1)
    labels = torch.argmax(label, dim=1)

    # 先在最后一个维度上取argmax，然后在dim=1取argmax
    predictions = torch.argmax(similarity, dim=-1)  # 在最后一个维度上取最大值
    predictions = torch.argmax(predictions, dim=1)  # 现在在第二个维度上取最大值

    # 打印以调试
    print(f"Labels shape: {labels.shape}")
    print(f"Predictions shape (after argmax): {predictions.shape}")

    # 计算准确率
    acc = torch.mean((labels == predictions).float())

    return avg_loss, acc

def training_loop(compute_features, patches, optimizer, iterations=1000, batch_size=128):
    loss_list = []
    try:
        print("Starting training loop.")
        for idx, batch in zip(range(iterations), patches.iterate_batches(batch_size)):
            Xl, Xr_pos, Xr_neg = batch
            label = torch.eye(2).cuda()[[0]*len(Xl)]

            score_pos = calculate_similarity_score(compute_features, Xl, Xr_pos)
            score_neg = calculate_similarity_score(compute_features, Xl, Xr_neg)
            loss, acc = hinge_loss(score_pos, score_neg, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

            if idx % 50 == 0:
                print(f"Loss ({idx:04d} it):{loss:.04f} \tAccuracy: {acc:.3f}")
    finally:
        patches.stop()
        print("Finished training!")


def compute_disparity_CNN(compute_features, img_l, img_r, max_disparity=50):
    """
    Computes the disparity of the stereo image pair.

    Args:
        compute_features:  pytorch module object
        img_l: tensor holding the left image
        img_r: tensor holding the right image
        max_disparity (int): maximum disparity

    Returns:
        D: tensor holding the disparity
    """
    # Ensure the input images are on the same device as the model
    img_l = img_l.cuda()
    img_r = img_r.cuda()

    # get the image features by applying the similarity metric
    Fl = compute_features(img_l[None])
    Fr = compute_features(img_r[None])

    # images of shape B x H x W x C
    B, H, W, C = Fl.shape
    # Initialize the disparity
    disparity = torch.zeros((B, H, W)).int().cuda()  # Ensure disparity is also on GPU
    # Initialize current similarity to -infimum
    current_similarity = torch.ones((B, H, W)).cuda() * -np.inf

    # Loop over all possible disparity values
    Fr_shifted = Fr
    for d in range(max_disparity + 1):
        if d > 0:
            # initialize shifted right image
            Fr_shifted = torch.zeros_like(Fr).cuda()  # Ensure shifted image is also on GPU
            # insert values which are shifted to the right by d
            Fr_shifted[:, :, d:] = Fr[:, :, :-d]

        # Calculate similarities
        sim_d = torch.sum(Fl * Fr_shifted, dim=3)
        # Check where similarity for disparity d is better than current one
        indices_pos = sim_d > current_similarity
        # Enter new similarity values
        current_similarity[indices_pos] = sim_d[indices_pos]
        # Enter new disparity values
        disparity[indices_pos] = d

    return disparity

def visualize_disparity(disparity, im_left, im_right, title='Disparity Map', max_disparity=50):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(im_left, cmap='gray')
    plt.title("Left Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(im_right, cmap='gray')
    plt.title("Right Image")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(disparity, cmap='jet', vmin=0, vmax=max_disparity)
    plt.title(title)
    plt.colorbar(label='Disparity')
    plt.axis('off')

    plt.show()

# 假设有这些路径和数据集
input_dir = './KITTI_2015_subset'
model_out_dir = './output/handcrafted_stereo/model'
os.makedirs(model_out_dir, exist_ok=True)

# 加载训练数据集
dataset = KITTIDataset(
    os.path.join(input_dir, "data_scene_flow/training/"),
    os.path.join(input_dir, "data_scene_flow/training/disp_noc_0"),
)
patches = PatchProvider(dataset, patch_size=(9, 9))

compute_features = StereoMatchingNetwork()
compute_features.train()
compute_features.cuda()

optimizer = SGD(compute_features.parameters(), lr=3e-4, momentum=0.9)

# 训练循环
training_loop(compute_features, patches, optimizer, iterations=1000, batch_size=128)

# 加载测试数据集
dataset = KITTIDataset(os.path.join(input_dir, "data_scene_flow/testing/"))
compute_features.eval()

for i in range(len(dataset)):
    img_left, img_right = dataset[i]
    img_left_padded = add_padding(img_left, padding=4)
    img_right_padded = add_padding(img_right, padding=4)
    img_left_padded, img_right_padded = torch.Tensor(img_left_padded), torch.Tensor(img_right_padded)

    disparity_map = compute_disparity_CNN(compute_features, img_left_padded, img_right_padded, max_disparity=50)

    title = f'Disparity map for image {i:04d}'
    # 将 disparity_map 移动到 CPU 并转换为 NumPy 数组
    disparity_map_cpu = disparity_map.squeeze().cpu().numpy()

    # 然后传递给 visualize_disparity 函数
    visualize_disparity(
        disparity_map_cpu,  # 这是已经从 GPU 转换到 CPU 并转为 NumPy 数组的视差图
        img_left.squeeze(),  # 如果 img_left 已经是 NumPy 数组，直接使用 squeeze()
        img_right.squeeze(),  # 如果 img_right 已经是 NumPy 数组，直接使用 squeeze()
        title,
        max_disparity=50
    )