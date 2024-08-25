import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像并转换为RGB
img1 = cv2.cvtColor(cv2.imread(r'C:\Users\14168\1\Python\pythonProject\homework\homework-03-sfm-and-stereo-HM-ZC-main\code\stereo\KITTI_2015_subset\data_scene_flow\training\image_2\000002_10.png'), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread(r'C:\Users\14168\1\Python\pythonProject\homework\homework-03-sfm-and-stereo-HM-ZC-main\code\stereo\KITTI_2015_subset\data_scene_flow\training\image_3\000002_10.png'), cv2.COLOR_BGR2RGB)

# 假设我们有相机的内参矩阵
K1 = np.array([[469.8769, 0, 334.8598],
               [0, 469.8360, 240.2752],
               [0, 0, 1.0]])

K2 = np.array([[469.8769, 0, 334.8598],
               [0, 469.8360, 240.2752],
               [0, 0, 1.0]])


def get_keypoints(img1, img2):
    descriptor = cv2.SIFT_create(nfeatures=10000)

    keypoints1, features1 = descriptor.detectAndCompute(img1, None)
    keypoints2, features2 = descriptor.detectAndCompute(img2, None)

    bf = cv2.BFMatcher_create(cv2.NORM_L2)
    matches = bf.knnMatch(features1, features2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    keypoints1 = np.float32([keypoints1[good_match.queryIdx].pt for good_match in good]).reshape(-1, 2)
    keypoints2 = np.float32([keypoints2[good_match.trainIdx].pt for good_match in good]).reshape(-1, 2)

    keypoints1 = np.concatenate([keypoints1, np.ones((keypoints1.shape[0], 1))], axis=-1)
    keypoints2 = np.concatenate([keypoints2, np.ones((keypoints2.shape[0], 1))], axis=-1)

    return keypoints1, keypoints2


keypoints1, keypoints2 = get_keypoints(img1, img2)

def compute_fundamental_matrix(keypoints1, keypoints2):
    A = np.zeros((keypoints1.shape[0], 9))
    for i in range(keypoints1.shape[0]):
        x1, y1 = keypoints1[i][:2]
        x2, y2 = keypoints2[i][:2]
        A[i] = [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]

    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt
    return F

F = compute_fundamental_matrix(keypoints1, keypoints2)

def compute_essential_matrix(F, K1, K2):
    E = K2.T @ F @ K1
    return E

E = compute_essential_matrix(F, K1, K2)

R1, R2, t = cv2.decomposeEssentialMat(E)

def triangulate_point(keypoints1, keypoints2, K1, K2, R, t):
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K2 @ np.hstack((R, t))

    points_3d = []
    for p1, p2 in zip(keypoints1, keypoints2):
        A = np.zeros((4, 4))
        A[0] = p1[0] * P1[2] - P1[0]
        A[1] = p1[1] * P1[2] - P1[1]
        A[2] = p2[0] * P2[2] - P2[0]
        A[3] = p2[1] * P2[2] - P2[1]

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[-1]
        points_3d.append(X)

    return np.array(points_3d)

# 选择R和t的正确解
R, t = R1, t  # 假设这是正确的解
points_3d = triangulate_point(keypoints1, keypoints2, K1, K2, R, t)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='r', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()