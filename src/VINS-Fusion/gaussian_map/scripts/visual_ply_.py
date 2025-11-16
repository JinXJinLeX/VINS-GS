import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.special import sph_harm

# === 参数 ===
ply_path = "point_cloud.ply"  # 修改路径
num_points = 1000               # 显示前多少个高斯
scale_factor = 5.0            # 控制球谐形变幅度
base_radius = 0.1             # 每个高斯球的基础半径
l_max = 3                      # 3阶SH → 共16个基函数
resolution = 20               # 球面网格分辨率
axis_length = 0.5             # 坐标轴长度

# === 读取 PLY ===
plydata = PlyData.read(ply_path)
v = plydata["vertex"]

xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1)

shconf_names = sorted([p.name for p in v.properties if p.name.startswith("sh_conf_")],
                      key=lambda x: int(x.split("_")[-1]))
sh_conf = np.stack([np.asarray(v[name]) for name in shconf_names], axis=-1)

print(f"Loaded sh_conf: shape={sh_conf.shape}, mean={sh_conf.mean():.3f}, std={sh_conf.std():.3f}, "
      f"min={sh_conf.min():.3f}, max={sh_conf.max():.3f}")

# === 自动归一化，防止爆炸 ===
sh_conf = np.clip(sh_conf, np.percentile(sh_conf, 1), np.percentile(sh_conf, 99))
sh_conf = (sh_conf - sh_conf.mean(axis=1, keepdims=True)) / (sh_conf.std(axis=1, keepdims=True) + 1e-8)

# === 球面采样 ===
theta = np.linspace(0, np.pi, resolution)
phi = np.linspace(0, 2 * np.pi, resolution*2)
theta, phi = np.meshgrid(theta, phi)

# 构造16个球谐基（实部）
Y_basis = []
for l in range(l_max + 1):
    for m in range(-l, l + 1):
        Y_basis.append(sph_harm(m, l, phi, theta).real)
Y_basis = np.stack(Y_basis, axis=-1)  # (phi, theta, 16)

# === Open3D 可视化 ===
meshes = []

sel_idx = np.arange(min(num_points, xyz.shape[0]))

# 简单的蓝-红渐变 colormap
def value_to_color(val):
    c = (val + 1) / 2
    return np.array([c, 0, 1-c])

for idx in sel_idx:
    coeffs = sh_conf[idx]
    f_val = np.tensordot(Y_basis, coeffs, axes=([2], [0]))
    f_val = f_val / np.max(np.abs(f_val))  # 归一化 [-1,1]

    r = base_radius * (1 + scale_factor * f_val)

    # 球坐标 -> 笛卡尔
    x = r * np.sin(theta) * np.cos(phi) + xyz[idx, 0]
    y = r * np.sin(theta) * np.sin(phi) + xyz[idx, 1]
    z = r * np.cos(theta) + xyz[idx, 2]

    vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)

    # 构造三角面
    triangles = []
    n_phi, n_theta = theta.shape
    for i in range(n_phi-1):
        for j in range(n_theta-1):
            idx0 = i * n_theta + j
            idx1 = i * n_theta + (j+1)
            idx2 = (i+1) * n_theta + j
            idx3 = (i+1) * n_theta + (j+1)
            triangles.append([idx0, idx2, idx1])
            triangles.append([idx1, idx2, idx3])
    triangles = np.array(triangles)

    # 创建 Open3D 网格
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # 顶点颜色
    f_flat = f_val.flatten()
    colors = np.array([value_to_color(v) for v in f_flat])
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    mesh.compute_vertex_normals()
    meshes.append(mesh)

# === 添加坐标轴 ===
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_length, origin=[0, 0, 0])
meshes.append(axis)

# === 显示 ===
o3d.visualization.draw_geometries(meshes)

