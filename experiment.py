# 导入必要的库
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import radon, iradon, resize
from skimage.data import shepp_logan_phantom

# 生成一phantom图像
P = shepp_logan_phantom()
P_resized = resize(P, (128, 128))  # Resize the phantom image to 128x128

# 显示phantom图像
print(f'显示phantom图像')
plt.figure()
plt.imshow(P_resized, cmap='gray')
plt.title('Resized Phantom Image')
plt.show()

# 使用radon变换，考察图像的正弦曲线图
print(f'使用radon变换，考察图像的正弦曲线图')

theta = np.linspace(0., 180., max(P_resized.shape), endpoint=False)
R_512 = radon(P_resized, theta, circle=True)
plt.figure()
plt.imshow(R_512, cmap='hot', extent=(0, 180, R_512.shape[0], 0), aspect='auto')
plt.xlabel('Parallel Rotation Angle - $\\theta$ (degrees)')
plt.ylabel('Parallel Sensor Position - $x\'$ (pixels)')
plt.title('Sinogram of the Resized Phantom Image')
plt.colorbar()
plt.show()

# 生成一个简单的图像并考察正弦曲线的数量及分布情况
print(f'生成一个简单的图像并考察正弦曲线的数量及分布情况')

f = np.zeros((256, 256))
i = [87, 103, 225]
j = [95, 124, 189]
f[i, j] = 1
plt.figure()
plt.imshow(f, cmap='gray')
plt.title('Simple Image')
plt.show()

# 对简单图像进行radon变换
theta = np.linspace(0, 180, max(f.shape), endpoint=False)
R_256 = radon(f, theta, circle=True)
plt.figure()
plt.imshow(R_256, cmap='hot', extent=(0, 180, R_256.shape[0], 0), aspect='auto')
plt.xlabel('Parallel Rotation Angle - $\\theta$ (degrees)')
plt.ylabel('Parallel Sensor Position - $x\'$ (pixels)')
plt.title('Sinogram of Simple Image')
plt.colorbar()
plt.show()

# 分析：




# Radon 变换投影重建
theta1 = np.linspace(0, 170, 18, endpoint=False)
R1 = radon(P_resized, theta1, circle=True)
num_angles_R1 = len(theta1)
print(f'Number of angles for R1: {num_angles_R1}')

theta2 = np.linspace(0, 175, 36, endpoint=False)
R2 = radon(P_resized, theta2, circle=True)
num_angles_R2 = len(theta2)
print(f'Number of angles for R2: {num_angles_R2}')

theta3 = np.linspace(0, 178, 90, endpoint=False)
R3 = radon(P_resized, theta3, circle=True)
num_angles_R3 = len(theta3)
print(f'Number of angles for R3: {num_angles_R3}')

# 展示不同角度下的正弦曲线图
print(f'展示18、36、90三种不同角度离散形式下的正弦曲线图')
plt.figure(figsize=(15, 5))

# R1 Sinogram
plt.subplot(131)
plt.imshow(R1, cmap='hot', extent=(0, 170, R1.shape[0], 0), aspect='auto')
plt.xlabel('Parallel Rotation Angle - $\\theta$ (degrees)')
plt.ylabel('Parallel Sensor Position - $x\'$ (pixels)')
plt.title('R1 Sinogram')
plt.colorbar()

# R2 Sinogram
plt.subplot(132)
plt.imshow(R2, cmap='hot', extent=(0, 175, R2.shape[0], 0), aspect='auto')
plt.xlabel('Parallel Rotation Angle - $\\theta$ (degrees)')
plt.ylabel('Parallel Sensor Position - $x\'$ (pixels)')
plt.title('R2 Sinogram')
plt.colorbar()

# R3 Sinogram
plt.subplot(133)
plt.imshow(R3, cmap='hot', extent=(0, 178, R3.shape[0], 0), aspect='auto')
plt.xlabel('Parallel Rotation Angle - $\\theta$ (degrees)')
plt.ylabel('Parallel Sensor Position - $x\'$ (pixels)')
plt.title('R3 Sinogram')
plt.colorbar()

plt.tight_layout()
plt.show()

# 分析：



print(f'展示18、36、90三种不同角度离散形式下的投影重建效果')

# 笔束反投影重建：
output_size = max(P.shape)
dtheta1 = theta1[1] - theta1[0]
I1 = iradon(R1, theta1, output_size=output_size, filter_name='ramp')
plt.figure()
plt.subplot(131)
plt.imshow(I1, cmap='gray')
plt.title('I1')

dtheta2 = theta2[1] - theta2[0]
I2 = iradon(R2, theta2, output_size=output_size, filter_name='ramp')
plt.subplot(132)
plt.imshow(I2, cmap='gray')
plt.title('I2')

dtheta3 = theta3[1] - theta3[0]
I3 = iradon(R3, theta3, output_size=output_size, filter_name='ramp')
plt.subplot(133)
plt.imshow(I3, cmap='gray')
plt.title('I3')

plt.tight_layout()
plt.show()

# 分析：



R = radon(P_resized, theta, circle=True)
I1 = iradon(R, theta, output_size=128, filter_name=None)
I2 = iradon(R, theta, output_size=128, filter_name='ramp')
plt.figure(figsize=(10, 3))
plt.subplot(131)
plt.imshow(P_resized, cmap='gray')
plt.title('Original')

plt.subplot(132)
plt.imshow(I1, cmap='gray')
plt.title('Unfiltered Backprojection')

plt.subplot(133)
plt.imshow(I2, cmap='gray')
plt.title('Filtered Backprojection')
plt.tight_layout()
plt.show()

# 继续滤波反投影重建模拟
# 使用不同的滤波器进行反投影重建
plt.figure()
plt.subplot(221)
plt.imshow(iradon(R, theta, output_size=128, filter_name='ramp'), cmap='gray')
plt.title('Ram-Lak filter')

plt.subplot(222)
plt.imshow(iradon(R, theta, output_size=128, filter_name='shepp-logan'), cmap='gray')
plt.title('Shepp-Logan filter')

plt.subplot(223)
plt.imshow(iradon(R, theta, output_size=128, filter_name='cosine'), cmap='gray')
plt.title('Cosine filter')

plt.subplot(224)
plt.imshow(iradon(R, theta, output_size=128, filter_name='hamming'), cmap='gray')
plt.title('Hamming filter')

plt.tight_layout()
plt.show()

# 分析：



# 噪声对图像的影响：
print(f'展示噪声对图像的影响')

# 给phantom图像添加椒盐噪声
P_noisy = P + np.random.uniform(-0.02, 0.02, P.shape)
P_noisy = np.clip(P_noisy, 0, 1)
plt.figure()
plt.imshow(P, cmap='gray')
plt.title('Noisy phantom image')
plt.show()

print(f'展示不同的角度进行反投影重建中，噪声对图像的影响')

# 使用不同的角度进行反投影重建
angle = np.linspace(0, 179, 180, endpoint=False)
R = radon(P, angle, circle=True)
I1 = iradon(R, angle, output_size=128, filter_name='ramp')
plt.figure()
plt.subplot(131)
plt.imshow(I1, cmap='gray')
plt.title('180 angles')

angle = np.linspace(0, 179, 90, endpoint=False)
R = radon(P, angle, circle=True)
I2 = iradon(R, angle, output_size=128, filter_name='ramp')
plt.subplot(132)
plt.imshow(I2, cmap='gray')
plt.title('90 angles')

angle = np.linspace(0, 179, 45, endpoint=False)
R = radon(P, angle, circle=True)
I3 = iradon(R, angle, output_size=128, filter_name='ramp')
plt.subplot(133)
plt.imshow(I3, cmap='gray')
plt.title('45 angles')

plt.tight_layout()
plt.show()

# 分析：
