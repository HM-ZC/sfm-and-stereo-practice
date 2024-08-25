import numpy as np
import matplotlib.pyplot as plt
import cv2


def add_padding(I, padding):
    """
    Adds zero padding to an RGB or grayscale image.

    Args:
        I (np.ndarray): HxWx? numpy array containing RGB or grayscale image

    Returns:
        P (np.ndarray): (H+2*padding)x(W+2*padding)x? numpy array containing zero padded image
    """
    if len(I.shape) == 2:
        H, W = I.shape
        padded = np.zeros((H + 2 * padding, W + 2 * padding), dtype=np.float32)
        padded[padding:-padding, padding:-padding] = I
    else:
        H, W, C = I.shape
        padded = np.zeros((H + 2 * padding, W + 2 * padding, C), dtype=I.dtype)
        padded[padding:-padding, padding:-padding] = I

    return padded


def sad(image_left, image_right, window_size=3, max_disparity=50):
    """
    Compute the sum of absolute differences between image_left and image_right.

    Args:
        image_left (np.ndarray): HxW numpy array containing grayscale right image
        image_right (np.ndarray): HxW numpy array containing grayscale left image
        window_size: window size (default 3)
        max_disparity: maximal disparity to reduce search range (default 50)

    Returns:
        D (np.ndarray): HxW numpy array containing the disparity for each pixel
    """

    D = np.zeros_like(image_left)

    # add zero padding
    padding = window_size // 2
    image_left = add_padding(image_left, padding).astype(np.float32)
    image_right = add_padding(image_right, padding).astype(np.float32)

    height = image_left.shape[0]
    width = image_left.shape[1]

    # 遍历图像的每个像素
    for y in range(padding, height - padding):
        for x in range(padding, width - padding):
            min_sad = float('inf')
            best_disparity = 0
            # 在视差范围内查找最小SAD
            for d in range(max_disparity):
                if x - d >= padding:
                    # 计算窗口内SAD
                    sad_value = np.sum(np.abs(image_left[y - padding:y + padding + 1, x - padding:x + padding + 1] -
                                              image_right[y - padding:y + padding + 1,
                                              x - padding - d:x + padding + 1 - d]))
                    if sad_value < min_sad:
                        min_sad = sad_value
                        best_disparity = d
            D[y - padding, x - padding] = best_disparity
    return D


def visualize_disparity(disparity, im_left, im_right, title='Disparity Map', max_disparity=50):
    """
    Generates a visualization for the disparity map.

    Args:
        disparity (np.array): disparity map
        title: plot title
        out_file: output file path
        max_disparity: maximum disparity
    """
    plt.figure(figsize=(15, 5))

    # 显示左图
    plt.subplot(1, 3, 1)
    plt.imshow(im_left, cmap='gray')
    plt.title("Left Image")
    plt.axis('off')

    # 显示右图
    plt.subplot(1, 3, 2)
    plt.imshow(im_right, cmap='gray')
    plt.title("Right Image")
    plt.axis('off')

    # 显示视差图
    plt.subplot(1, 3, 3)
    plt.imshow(disparity, cmap='jet', vmin=0, vmax=max_disparity)
    plt.title(title)
    plt.colorbar(label='Disparity')
    plt.axis('off')

    plt.show()

def compare_window_sizes(im_left, im_right, window_sizes, max_disparity=50):
    """
    尝试不同的窗口大小，并对比生成的视差图。

    Args:
        im_left: 左视图图像
        im_right: 右视图图像
        window_sizes: 要尝试的窗口大小列表
        max_disparity: 最大视差值（默认50）
    """
    for window_size in window_sizes:
        # 计算视差图
        disparity_map = sad(im_left, im_right, window_size, max_disparity)
        # 可视化视差图
        title = f'Disparity Map (Window Size: {window_size})'
        visualize_disparity(disparity_map, im_left, im_right, title=title, max_disparity=max_disparity)


# 示例使用
if __name__ == "__main__":
    # 加载一对立体图像（假设你有一对立体图像left.png和right.png）
    im_left = cv2.imread(r'C:\Users\14168\1\Python\pythonProject\homework\homework-03-sfm-and-stereo-HM-ZC-main\code\stereo\KITTI_2015_subset\data_scene_flow\training\image_2\000001_10.png', cv2.IMREAD_GRAYSCALE)
    im_right = cv2.imread(r'C:\Users\14168\1\Python\pythonProject\homework\homework-03-sfm-and-stereo-HM-ZC-main\code\stereo\KITTI_2015_subset\data_scene_flow\training\image_3\000001_10.png', cv2.IMREAD_GRAYSCALE)

    # 定义要尝试的窗口大小列表
    window_sizes = [3, 7, 15]

    # 对比不同窗口大小生成的视差图
    compare_window_sizes(im_left, im_right, window_sizes, max_disparity=50)