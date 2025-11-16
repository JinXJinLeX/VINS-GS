import numpy as np
from plyfile import PlyData
from scipy.special import sph_harm
import matplotlib.pyplot as plt

# === 参数 ===
ply_path = "point_cloud.ply"     # 修改成你的路径
num_points = 20                # 可视化前 20 个高斯
scale_factor = 3.0             # 调整球谐变形强度
l_max = 3                      # 你的是3阶
num_coeffs = (l_max + 1) ** 2  # 16个

# === 读取 PLY ===
plydata = PlyData.read(ply_path)
v = plydata["vertex"]

# 读取 sh_conf_0~15
shconf_names = [p.name for p in v.properties if p.name.startswith("sh_conf_")]
shconf_names = sorted(shconf_names, key=lambda x: int(x.split('_')[-1]))

sh_conf = np.stack([np.asarray(v[name]) for name in shconf_names], axis=-1)
print(f"Loaded sh_conf: shape={sh_conf.shape}, mean={sh_conf.mean():.4f}, std={sh_conf.std():.4f}")

# === 生成球面网格 ===
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 200)
theta, phi = np.meshgrid(theta, phi)

# 构造所有 16 个球谐基（实部）
Y_basis = []
for l in range(l_max + 1):
    for m in range(-l, l + 1):
        Y_basis.append(sph_harm(m, l, phi, theta).real)
Y_basis = np.stack(Y_basis, axis=-1)  # [phi, theta, 16]

# === 可视化前 num_points 个点 ===
fig = plt.figure(figsize=(15, 12))
for idx in range(num_points):
    coeffs = sh_conf[idx]  # (16,)

    # 合成标量场
    f_val = np.tensordot(Y_basis, coeffs, axes=([2], [0]))

    # 半径扰动 (1 + α*f)
    r = 1 + scale_factor * f_val
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # 归一化颜色映射
    norm = (f_val - f_val.min()) / (f_val.max() - f_val.min())
    colors = plt.cm.coolwarm(norm)

    ax = fig.add_subplot(4, 5, idx + 1, projection='3d')
    ax.plot_surface(x, y, z, facecolors=colors, linewidth=0, antialiased=False)
    ax.set_title(f"SH Conf #{idx}", fontsize=10)
    ax.axis('off')

plt.suptitle("Visualization of sh_conf (l=0~3)", fontsize=16)
plt.tight_layout()
plt.show()

