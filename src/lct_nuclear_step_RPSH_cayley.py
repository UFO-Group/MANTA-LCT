import torch
from torch import nn
import numpy as np
import math
from scipy.interpolate import interp2d, interpn

scale_L=1.8897259885789233

device = torch.device('cuda')
nstate = 2
Natoms= 61


def get_geom(QM_in_path,natoms):
    
    with open(QM_in_path, 'r') as f:
        lines = f.readlines()

    initcond_xyz = ''.join(lines[:natoms + 2])
    geom = read(io.StringIO(initcond_xyz), format='xyz')
    return geom

def topo_list(filename):
    bonds = []
    angles = []
    dihedrals = []

    line_re = re.compile(r'^(?:\d+\.\s*)?([BAD])\((.*?)\)', re.IGNORECASE)

    with open(filename, 'r') as f:
        for raw in f:
            line = raw.strip()
            m = line_re.match(line)
            if not m:
                continue

            typ = m.group(1).upper()      # 'B' / 'A' / 'D'
            inside = m.group(2)           # 比如 "C   1,C   0"
            parts = [p.strip() for p in inside.split(',')]
            # 从每一段里提取数字
            nums = [int(re.search(r'(\d+)', p).group(1)) for p in parts]

            if typ == 'B' and len(nums) == 2:
                bonds.append(tuple(nums))
            elif typ == 'A' and len(nums) == 3:
                angles.append(tuple(nums))
            elif typ == 'D' and len(nums) == 4:
                dihedrals.append(tuple(nums))

    return bonds, angles, dihedrals

def compute_internals(coords, bonds, angles, dihedrals):

    p = np.asarray(coords, dtype=float)
    values = []

    # 1. 键长
    for i, j in bonds:
        r = p[j] - p[i]
        values.append(np.linalg.norm(r))

    # 2. 键角
    for i, j, k in angles:
        v1 = p[i] - p[j]
        v2 = p[k] - p[j]
        cosφ = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cosφ = np.clip(cosφ, -1.0, 1.0)
        values.append(np.arccos(cosφ))

    # 3. 二面角（单函数内完成）
    for i, j, k, l in dihedrals:
        # 三段键向量
        b1 = p[j] - p[i]
        b2 = p[k] - p[j]
        b3 = p[l] - p[k]
        # 单位化 b2
        b2_u = b2 / np.linalg.norm(b2)
        # 两个面法向量
        n1 = np.cross(b1, b2_u)
        n2 = np.cross(b2_u, b3)
        # 退化检查
        if np.linalg.norm(n1) == 0 or np.linalg.norm(n2) == 0:
            values.append(np.nan)
            continue
        # 计算 x, y
        x = np.dot(n1, n2)
        y = np.dot(b2_u, np.cross(n1, n2))
        # atan2 返回 (-π, π]
        values.append(np.arctan2(y, x))

    return values


def _cross_matrix_torch(v: torch.Tensor) -> torch.Tensor:
    return torch.tensor([
        [ 0.0,   -v[2],  v[1]],
        [ v[2],   0.0,  -v[0]],
        [-v[1],  v[0],   0.0]
    ], dtype=v.dtype, device=v.device)



##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Batch computing for dXdQ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def compute_dihedral_and_grads(coords):
    """
    coords: (B, Nφ, 4, 3)
        coords[..., 0, :] = r_i
        coords[..., 1, :] = r_j
        coords[..., 2, :] = r_k
        coords[..., 3, :] = r_l

    返回:
        phi: (B, Nφ)
        grads: (B, Nφ, 4, 3)
    """

    B, Nphi, _, _ = coords.shape

    # ----------------------------------------
    # 1. 为自动微分准备
    # ----------------------------------------
    c = coords.clone().detach().requires_grad_(True)   # (B,Nφ,4,3)

    r_i = c[..., 0, :]      # (B,Nφ,3)
    r_j = c[..., 1, :]
    r_k = c[..., 2, :]
    r_l = c[..., 3, :]

    # ----------------------------------------
    # 2. 计算二面角 φ = atan2(y, x)
    # ----------------------------------------
    b1 = r_j - r_i                  # (B,Nφ,3)
    b2 = r_k - r_j
    b3 = r_l - r_k

    # 单位化 b2
    b2_norm = torch.norm(b2, dim=-1, keepdim=True) + 1e-12
    b2u = b2 / b2_norm              # (B,Nφ,3)

    n1 = torch.cross(b1, b2u, dim=-1)  # (B,Nφ,3)
    n2 = torch.cross(b2u, b3, dim=-1)  # (B,Nφ,3)

    x = torch.sum(n1 * n2, dim=-1)     # (B,Nφ)
    y = torch.sum(b2u * torch.cross(n1, n2, dim=-1), dim=-1)  # (B,Nφ)

    phi = torch.atan2(y, x)            # (B,Nφ), 弧度

    # ----------------------------------------
    # 3. 自动微分：对 (B,Nφ,4,3) 求梯度
    # ----------------------------------------
    grad_c = torch.autograd.grad(
        phi, c,
        grad_outputs=torch.ones_like(phi),
        create_graph=False,
        retain_graph=False
    )[0]   # (B,Nφ,4,3)

    # ----------------------------------------
    # 4. 返回 φ 和梯度
    # ----------------------------------------
    return phi, grad_c



def compute_angle_and_grads(coords):
    """
    输入:
        coords: (B, Nθ, 3, 3)
            coords[..., 0, :] = r_i
            coords[..., 1, :] = r_j
            coords[..., 2, :] = r_k

    输出:
        theta: (B, Nθ)          # 弧度
        dtheta_dR: (B, Nθ, 3, 3)
            最后两个维度对应 (r_i, r_j, r_k)
            dθ/dr_i, dθ/dr_j, dθ/dr_k
    """

    eps = 1e-12

    # ------------------------------------------
    # 1. 提取原子坐标
    # ------------------------------------------
    r_i = coords[..., 0, :]          # (B,Nθ,3)
    r_j = coords[..., 1, :]
    r_k = coords[..., 2, :]

    # 向量
    u = r_i - r_j                    # (B,Nθ,3)
    v = r_k - r_j

    # 长度
    ru = torch.norm(u, dim=-1, keepdim=True)  # (B,Nθ,1)
    rv = torch.norm(v, dim=-1, keepdim=True)

    # 单位向量
    u_hat = u / (ru + eps)
    v_hat = v / (rv + eps)

    # ------------------------------------------
    # 2. cosθ 与 θ (弧度)
    # ------------------------------------------
    cos_theta = torch.sum(u_hat * v_hat, dim=-1, keepdim=True)  # (B,Nθ,1)
    cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    theta = torch.acos(cos_theta).squeeze(-1)     # (B,Nθ)

    # sinθ
    sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta + eps)    # (B,Nθ,1)

    # ------------------------------------------
    # 3. d(cosθ)/du, d(cosθ)/dv (解析)
    # ------------------------------------------
    # dcos/du
    dcos_du = (
        v_hat / (ru + eps)
        - cos_theta * u_hat / (ru + eps)
    )    # (B,Nθ,3)

    # dcos/dv
    dcos_dv = (
        u_hat / (rv + eps)
        - cos_theta * v_hat / (rv + eps)
    )    # (B,Nθ,3)

    # ------------------------------------------
    # 4. dθ/d(cosθ) = -1/sinθ
    # ------------------------------------------
    dtheta_dcos = -1.0 / (sin_theta + eps)   # (B,Nθ,1)

    # ------------------------------------------
    # 5. 链式法则 dθ/du, dθ/dv
    # ------------------------------------------
    dtheta_du = dtheta_dcos * dcos_du   # (B,Nθ,3)
    dtheta_dv = dtheta_dcos * dcos_dv

    # ------------------------------------------
    # 6. 还原到 r_i, r_j, r_k
    #
    # u = r_i - r_j
    # v = r_k - r_j
    #
    # dθ/dr_i = dθ/du
    # dθ/dr_k = dθ/dv
    # dθ/dr_j = -(dθ/du + dθ/dv)
    # ------------------------------------------
    dtheta_dri = dtheta_du                  # (B,Nθ,3)
    dtheta_drk = dtheta_dv
    dtheta_drj = -(dtheta_du + dtheta_dv)

    # ------------------------------------------
    # 7. 打包 (B,Nθ,3,3)
    # ------------------------------------------
    dtheta_dR = torch.stack(
        [dtheta_dri, dtheta_drj, dtheta_drk], dim=2
    )  # (B,Nθ,3,3)

    return theta, dtheta_dR



def compute_bond_and_grads(coords):
    """
    coords: (B, Nb, 2, 3)
        coords[...,0,:] = r_i
        coords[...,1,:] = r_j

    return:
        r:        (B, Nb)
        dr_dR:    (B, Nb, 2, 3)
    """

    r_i = coords[..., 0, :]    # (B,Nb,3)
    r_j = coords[..., 1, :]

    # v = r_i - r_j
    v = r_i - r_j              # (B,Nb,3)

    # r = |v|
    r = torch.norm(v, dim=-1, keepdim=True)   # (B,Nb,1)

    eps = 1e-12
    inv_r = 1.0 / (r + eps)

    # dr/dv = v / |v|
    dr_dv = v * inv_r          # (B,Nb,3)

    # mapping to atom coordinates
    # dv/dr_i = +I
    # dv/dr_j = -I
    dr_dri = dr_dv             # (B,Nb,3)
    dr_drj = -dr_dv            # (B,Nb,3)

    # pack
    dr_dR = torch.stack([dr_dri, dr_drj], dim=2)  # (B,Nb,2,3)

    return r.squeeze(-1), dr_dR

def get_dQdX_batch(X):
    
    r, dr_dR = compute_bond_and_grads(X[:, list_bond])
    theta, dtheta_dR = compute_angle_and_grads(X[:, list_ang])
    phi, dphi_dR = compute_dihedral_and_grads(X[:, list_dih])
    
    Q = torch.concat((r, theta, phi),axis=1)
    #X.shape
    dr_dR_ = torch.zeros([len(Q), Nbod, N, 3]).type(torch.float32).to(device)
    for iatom in range(2):
        dr_dR_[:, np.arange(Nbod), list_bond[:,iatom]] = dr_dR[:,:,iatom]
    
    dtheta_dR_ = torch.zeros([len(Q), Nang, N, 3]).type(torch.float32).to(device)
    for iatom in range(3):
        dtheta_dR_[:, np.arange(Nang), list_ang[:,iatom]] = dtheta_dR[:,:,iatom]
    
    dphi_dR_ = torch.zeros([len(Q), Ndih, N, 3]).type(torch.float32).to(device)
    for iatom in range(4):
        dphi_dR_[:, np.arange(Ndih), list_dih[:,iatom]] = dphi_dR[:,:,iatom]

    dQdX = torch.concat((dr_dR_, dtheta_dR_, dphi_dR_), axis=1)

    return Q, dQdX


def build_dxdq_batch(B):
    """
    B: (B, nI, 3N)
       Wilson B-matrix (batch version)

    返回:
    dxdq: (B, nI, 3N)
    """
    # -------------------------
    # 1. G = B @ B.T   (batch)
    # -------------------------
    # B: (B,nI,3N)
    G = B @ B.transpose(-1, -2)           # (B,nI,nI)

    # -------------------------
    # 2. batch eigen decomposition
    # -------------------------
    # torch.linalg.eigh 支持 batch
    w, V = torch.linalg.eigh(G)           # w: (B,nI), V: (B,nI,nI)

    # -------------------------
    # 3. 计算阈值 tol （每个 batch 一个 tol）
    # -------------------------
    eps = torch.finfo(w.dtype).eps        # 浮点精度
    # max(G.shape) = max(nI,nI) = nI，因此直接用 w.shape[-1]
    nI = w.shape[-1]
    # tol 形状: (B,1) 方便广播
    tol = (nI * torch.max(w, dim=-1).values * eps).unsqueeze(-1)   # (B,1)

    # -------------------------
    # 4. 广义逆特征值（batch）
    # -------------------------
    # w > tol → invert, else → 0
    w_inv = torch.where(w > tol, 1.0 / w, torch.zeros_like(w))     # (B,nI)

    # -------------------------
    # 5. 重建 G⁺ = V diag(w_inv) V^T
    # -------------------------
    # 先做 V * w_inv，自动广播到列：
    # V:    (B,nI,nI)
    # w_inv:(B,nI) → (B,1,nI)
    G_inv = (V * w_inv.unsqueeze(-2)) @ V.transpose(-1, -2)        # (B,nI,nI)

    # -------------------------
    # 6. dxdq = G_inv @ B  （batch）
    # -------------------------
    dxdq = G_inv @ B               # (B,nI,3N)

    return dxdq




#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# 元素符号 → 原子序数
PERIODIC_TABLE = {
    "H": 1,  "He": 2,
    "Li": 3, "Be": 4, "B": 5,  "C": 6,  "N": 7,  "O": 8,  "F": 9,  "Ne": 10,
    "Na": 11,"Mg": 12,"Al": 13,"Si": 14,"P": 15, "S": 16,"Cl": 17,"Ar": 18,
    "K": 19, "Ca": 20,
    # 如需更多元素你再告诉我我帮你加
}

def read_xyz(filename, dtype=torch.float32, device="cpu"):
    """
    读取 xyz 文件并生成 Z 和 R
    返回:
        Z: (N,) 原子序数 (float)
        R: (N, 3) 坐标 (float)
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # 第一行是原子数
    N = int(lines[0].strip())

    Z_list = []
    R_list = []

    for line in lines[2:2+N]:  # 从第三行开始是数据
        parts = line.split()
        
        if len(parts) < 4:
            continue
        
        symbol = parts[0]
        x, y, z = map(float, parts[1:4])

        # 原子符号转原子序数
        if symbol not in PERIODIC_TABLE:
            raise ValueError(f"元素符号 {symbol} 不在 periodic table 中，请补充定义。")

        Z_list.append(PERIODIC_TABLE[symbol])
        R_list.append([x, y, z])

    Z = torch.tensor(Z_list, dtype=dtype, device=device)
    R = torch.tensor(R_list, dtype=dtype, device=device)

    return Z, R


def topo_list(filename):
    bonds = []
    angles = []
    dihedrals = []

    line_re = re.compile(r'^(?:\d+\.\s*)?([BAD])\((.*?)\)', re.IGNORECASE)

    with open(filename, 'r') as f:
        for raw in f:
            line = raw.strip()
            m = line_re.match(line)
            if not m:
                continue

            typ = m.group(1).upper()      # 'B' / 'A' / 'D'
            inside = m.group(2)           # 比如 "C   1,C   0"
            parts = [p.strip() for p in inside.split(',')]
            # 从每一段里提取数字
            nums = [int(re.search(r'(\d+)', p).group(1)) for p in parts]

            if typ == 'B' and len(nums) == 2:
                bonds.append(tuple(nums))
            elif typ == 'A' and len(nums) == 3:
                angles.append(tuple(nums))
            elif typ == 'D' and len(nums) == 4:
                dihedrals.append(tuple(nums))

    return bonds, angles, dihedrals


#import torch

def compute_dihedral_and_grads(coords):
    """
    coords: (B, Nφ, 4, 3)
        r1 = coords[...,0,:]
        r2 = coords[...,1,:]
        r3 = coords[...,2,:]
        r4 = coords[...,3,:]
    return:
        cosφ: (B, Nφ)
        sinφ: (B, Nφ)
        dcos_dR: (B, Nφ, 4, 3)
        dsin_dR: (B, Nφ, 4, 3)
    """

    r1 = coords[..., 0, :]  # (B,Nφ,3)
    r2 = coords[..., 1, :]
    r3 = coords[..., 2, :]
    r4 = coords[..., 3, :]

    # bond vectors
    b1 = r2 - r1
    b2 = r3 - r2
    b3 = r4 - r3

    # normal vectors
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)

    # norms
    a = torch.norm(n1, dim=-1, keepdim=True)  # |n1|
    b = torch.norm(n2, dim=-1, keepdim=True)  # |n2|
    c = torch.norm(b2, dim=-1, keepdim=True)  # |b2|

    eps = 1e-12

    # ========== φ components ==========
    dot_n1_n2 = torch.sum(n1 * n2, dim=-1, keepdim=True)  # (B,Nφ,1)

    cos_phi = dot_n1_n2 / (a * b + eps)              # (B,Nφ,1)
    cross_term = torch.sum(b1 * n2, dim=-1, keepdim=True)  # (B,Nφ,1)
    sin_phi = cross_term * c / (a * b + eps)

    cos_phi = cos_phi.squeeze(-1)
    sin_phi = sin_phi.squeeze(-1)

    # ========== dcos/dA, dcos/dB ==========
    # A = n1, B = n2
    dcos_dA = n2 / (a * b + eps) - dot_n1_n2 / ((a * a) * b + eps) * n1
    dcos_dB = n1 / (a * b + eps) - dot_n1_n2 / (a * (b * b) + eps) * n2

    # ========== dsin/dA, dsin/dB ==========
    dsin_dA = -(sin_phi[..., None]) * n1 / (a * a + eps)

    dsin_dB = (c / (a * b + eps)) * b1 - (sin_phi[..., None]) * n2 / (b * b + eps)

    # ========== dA/db1, dA/db2 ==========
    # A = b1 × b2
    # ∂A/∂b1 = [·] × b2
    # ∂A/∂b2 = b1 × [·]
    dA_db1 = lambda X: torch.cross(X, b2, dim=-1)
    dA_db2 = lambda X: torch.cross(b1, X, dim=-1)

    # ========== dB/db2, dB/db3 ==========
    # B = b2 × b3
    dB_db2 = lambda X: torch.cross(X, b3, dim=-1)
    dB_db3 = lambda X: torch.cross(b2, X, dim=-1)

    # ========== dcos/dr1 ==========
    dcos_dr1 = -dA_db1(dcos_dA)

    # ========== dcos/dr2 ==========
    dcos_dr2 = (
        dA_db1(dcos_dA)           # +I contribution from db1/dr2
        - dA_db2(dcos_dA)         # -I contribution from db2/dr2
        + dB_db2(dcos_dB)
    )

    # ========== dcos/dr3 ==========
    dcos_dr3 = (
        dA_db2(dcos_dA)          # from db2/dr3
        + dB_db2(dcos_dB)        # from db2/dr3
        - dB_db3(dcos_dB)        # from db3/dr3
    )

    # ========== dcos/dr4 ==========
    dcos_dr4 = dB_db3(dcos_dB)

    # ==================================================
    # sinφ 的梯度同理
    # ==================================================

    dsin_dr1 = -dA_db1(dsin_dA)

    dsin_dr2 = (
        dA_db1(dsin_dA)
        - dA_db2(dsin_dA)
        + dB_db2(dsin_dB)
    )

    dsin_dr3 = (
        dA_db2(dsin_dA)
        + dB_db2(dsin_dB)
        - dB_db3(dsin_dB)
    )

    dsin_dr4 = dB_db3(dsin_dB)

    # pack gradients
    dcos_dR = torch.stack([dcos_dr1, dcos_dr2, dcos_dr3, dcos_dr4], dim=2)
    dsin_dR = torch.stack([dsin_dr1, dsin_dr2, dsin_dr3, dsin_dr4], dim=2)

    return cos_phi, sin_phi, dcos_dR, dsin_dR

#import torch

def compute_angle_and_grads(coords):
    """
    coords: (B, Nθ, 3, 3)
        r_i = coords[...,0,:]
        r_j = coords[...,1,:]
        r_k = coords[...,2,:]

    return:
        cosθ: (B, Nθ)
        dcos_dR: (B, Nθ, 3, 3)
    """

    r_i = coords[..., 0, :]   # (B,Nθ,3)
    r_j = coords[..., 1, :]
    r_k = coords[..., 2, :]

    # vectors
    v1 = r_i - r_j            # (B,Nθ,3)
    v2 = r_k - r_j

    # lengths
    n1 = torch.norm(v1, dim=-1, keepdim=True)   # (B,Nθ,1)
    n2 = torch.norm(v2, dim=-1, keepdim=True)

    eps = 1e-12

    # cosθ
    dot12 = torch.sum(v1 * v2, dim=-1, keepdim=True)  # (B,Nθ,1)
    cos_theta = (dot12 / (n1 * n2 + eps)).squeeze(-1)

    # ===========================
    # 解析梯度 d(cosθ)/d(v1), d(cosθ)/d(v2)
    # ===========================

    # dcos/dv1
    dcos_dv1 = (
        v2 / (n1 * n2 + eps)
        - (dot12 / (n1**3 * n2 + eps)) * v1
    )  # (B,Nθ,3)

    # dcos/dv2
    dcos_dv2 = (
        v1 / (n1 * n2 + eps)
        - (dot12 / (n1 * n2**3 + eps)) * v2
    )  # (B,Nθ,3)

    # ===========================
    # 现在映射回 r_i, r_j, r_k 的梯度
    #
    # v1 = r_i − r_j
    # v2 = r_k − r_j
    #
    # dv1/dr_i = +I
    # dv1/dr_j = -I
    # dv2/dr_k = +I
    # dv2/dr_j = -I
    # ===========================

    # dr_i
    dcos_dri = dcos_dv1

    # dr_k
    dcos_drk = dcos_dv2

    # dr_j
    dcos_drj = -(dcos_dv1 + dcos_dv2)

    # pack to (B,Nθ,3,3)
    dcos_dR = torch.stack([dcos_dri, dcos_drj, dcos_drk], dim=2)

    return cos_theta, dcos_dR

#import torch

def compute_bond_and_grads(coords):
    """
    coords: (B, Nb, 2, 3)
        coords[...,0,:] = r_i
        coords[...,1,:] = r_j

    return:
        r:        (B, Nb)
        dr_dR:    (B, Nb, 2, 3)
    """

    r_i = coords[..., 0, :]    # (B,Nb,3)
    r_j = coords[..., 1, :]

    # v = r_i - r_j
    v = r_i - r_j              # (B,Nb,3)

    # r = |v|
    r = torch.norm(v, dim=-1, keepdim=True)   # (B,Nb,1)

    eps = 1e-12
    inv_r = 1.0 / (r + eps)

    # dr/dv = v / |v|
    dr_dv = v * inv_r          # (B,Nb,3)

    # mapping to atom coordinates
    # dv/dr_i = +I
    # dv/dr_j = -I
    dr_dri = dr_dv             # (B,Nb,3)
    dr_drj = -dr_dv            # (B,Nb,3)

    # pack
    dr_dR = torch.stack([dr_dri, dr_drj], dim=2)  # (B,Nb,2,3)

    return r.squeeze(-1), dr_dR

#import torch

def compute_inverse_distance_and_grads(coords):
    """
    coords: (B, Npair, 2, 3)
        coords[...,0,:] = R_i
        coords[...,1,:] = R_j

    return:
        D:      (B, Npair)
        dD_dR:  (B, Npair, 2, 3)
                 gradient wrt (R_i, R_j)
    """

    Ri = coords[..., 0, :]  # (B,Npair,3)
    Rj = coords[..., 1, :]  # (B,Npair,3)

    # distance vector
    v = Ri - Rj             # (B,Npair,3)

    # distance magnitude r = |v|
    r = torch.norm(v, dim=-1, keepdim=True)  # (B,Npair,1)
    eps = 1e-12

    # inverse distance
    D = 1.0 / (r + eps)     # (B,Npair,1)

    # analytic gradient dD/dv = -v / r^3
    v_over_r3 = v / ((r + eps) ** 3)      # (B,Npair,3)
    dD_dv = -v_over_r3                    # (B,Npair,3)

    # map to atom coordinates
    # dv/dRi = +I
    # dv/dRj = -I
    dD_dRi = dD_dv                        # (B,Npair,3)
    dD_dRj = -dD_dv                       # (B,Npair,3)

    # stack
    dD_dR = torch.stack([dD_dRi, dD_dRj], dim=2)  # (B,Npair,2,3)

    return D.squeeze(-1), dD_dR

def get_nonbonded_pairs(Natoms, list_bond, list_ang):
    """
    输入:
        Natoms: 总原子数
        list_bond: [(i, j), ...]     # 1-2 pair
        list_ang:  [(i, j, k), ...]  # angle i-j-k -> i-k is 1-3
    
    输出:
        pairs: 所有排除了1-2和1-3作用后的 pair（i<j）
    """

    # ---- 1. 1–2 排除 ----
    excl_12 = set(tuple(sorted((i, j))) for (i, j) in list_bond)

    # ---- 2. 1–3 排除：从 angle(i,j,k) 得到 (i,k) ----
    excl_13 = set()
    for (i, j, k) in list_ang:
        excl_13.add(tuple(sorted((i, k))))

    # ---- 总排除集合 ----
    excluded = excl_12 | excl_13

    # ---- 3. 全部 i<j 的 pair ----
    pairs = []
    for i in range(Natoms):
        for j in range(i+1, Natoms):
            if (i, j) not in excluded:
                pairs.append((i, j))

    return pairs


def get_RdXdR(R):
    
    r, dr_dR = compute_bond_and_grads(R[:, list_bond])
    cos_theta, dcostheta_dR = compute_angle_and_grads(R[:, list_ang])
    D, dD_dR = compute_inverse_distance_and_grads(R[:, list_pair])
    cos_phi, sin_phi, dcosphi_dR, dsinphi_dR = compute_dihedral_and_grads(R[:, list_dih])
    
    X = torch.concat((r, cos_theta, cos_phi, sin_phi, D),axis=1)
    #X.shape
    dr_dR_ = torch.zeros([len(X), Nbod, N, 3]).type(torch.float32).to(device)
    for iatom in range(2):
        dr_dR_[:, np.arange(Nbod), list_bond[:,iatom]] = dr_dR[:,:,iatom]
    
    dcostheta_dR_ = torch.zeros([len(X), Nang, N, 3]).type(torch.float32).to(device)
    for iatom in range(3):
        dcostheta_dR_[:, np.arange(Nang), list_ang[:,iatom]] = dcostheta_dR[:,:,iatom]
    
    dcosphi_dR_ = torch.zeros([len(X), Ndih, N, 3]).type(torch.float32).to(device)
    for iatom in range(4):
        dcosphi_dR_[:, np.arange(Ndih), list_dih[:,iatom]] = dcosphi_dR[:,:,iatom]
    
    dsinphi_dR_ = torch.zeros([len(X), Ndih, N, 3]).type(torch.float32).to(device)
    for iatom in range(4):
        dsinphi_dR_[:, np.arange(Ndih), list_dih[:,iatom]] = dsinphi_dR[:,:,iatom]
    
    dD_dR_ = torch.zeros([len(X), NinvD, N, 3]).type(torch.float32).to(device)
    for iatom in range(2):
        dD_dR_[:, np.arange(NinvD), list_pair[:,iatom]] = dD_dR[:,:,iatom]

    dX_dR = torch.concat((dr_dR_, dcostheta_dR_, dcosphi_dR_, dsinphi_dR_, dD_dR_), axis=1)

    return X, dX_dR


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def calc_NAC_viaHess(energys_,forces_,hess_): ## for two states :(i,j) ---> NAC_ij
    
    ## energys_ : torch ( Ntraj x Nstate)
    ## forces_ : torch (Ntraj x Nstate x Natoms x 3)
    ## hess_ : torch  (Ntraj x Nstate x 3Natom x 3Natom)
    
    Nsamp = energys_.shape[0]
    Natoms = forces_.shape[2]
    
    istate = 0
    jstate = 1
    
    Hi = hess_[:,istate]
    Hj = hess_[:,jstate]
    
    dE = energys_[:,istate]-energys_[:,jstate]
    
    forces_ = -forces_
    GiGi = forces_[:,istate].reshape(Nsamp,Natoms*3,1)*forces_[:,istate].reshape(Nsamp,1,Natoms*3)
    GiGj = forces_[:,istate].reshape(Nsamp,Natoms*3,1)*forces_[:,jstate].reshape(Nsamp,1,Natoms*3)
    GjGi = forces_[:,jstate].reshape(Nsamp,Natoms*3,1)*forces_[:,istate].reshape(Nsamp,1,Natoms*3)
    GjGj = forces_[:,jstate].reshape(Nsamp,Natoms*3,1)*forces_[:,jstate].reshape(Nsamp,1,Natoms*3)
    
    G_diff = 0.5*(forces_[:,istate] -forces_[:, jstate])
    G_diff2= G_diff.reshape(Nsamp,Natoms*3,1)*G_diff.reshape(Nsamp,1,Natoms*3)
    
    #dH_2_ij = 0.5*(dE.unsqueeze(1).unsqueeze(2)*(Hi-Hj) + GiGi + GjGj - 2*GiGj)
    #magnitude = dH_2_ij/2-G_diff2
    
    magnitude = 0.5*0.5*(dE.unsqueeze(1).unsqueeze(2)*(Hi-Hj))
    
    u,s,vh = np.linalg.svd(magnitude.numpy())
    ev, ew = torch.tensor(vh[:,0]), torch.tensor(s[:,0])
    
    #u,s,vh = torch.linalg.svd(magnitude)
    #ev, ew = vh[:,0], s[:,0]
    
    hopping_direction = ev.reshape(Nsamp,Natoms,3)  ## for NAC[i,j]
    hopping_magnitude = ew.reshape(Nsamp,1,1)
    
    magnitude = torch.sqrt(hopping_magnitude)/torch.abs(dE.unsqueeze(1).unsqueeze(2)) #/2
    
    threshold_dE = 0.04 #0.04
    
    Mask_nac = (torch.abs(dE.unsqueeze(1).unsqueeze(2)) < threshold_dE).type(torch.float32)
    
    nacij = hopping_direction*magnitude*Mask_nac  ## for NAC_ij
    
    nstate = 2
    NACdr = torch.zeros([Nsamp,nstate,nstate,Natoms,3])
    NACdr[:,0:1,1:2] = nacij.unsqueeze(1).unsqueeze(1)
    NACdr[:,1:2,0:1] = -nacij.unsqueeze(1).unsqueeze(1)
    NACdr[:,0:1,0:1] = torch.zeros_like(nacij.unsqueeze(1).unsqueeze(1))
    NACdr[:,1:2,1:2] = torch.zeros_like(nacij.unsqueeze(1).unsqueeze(1))
    
    return NACdr


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    

def frame_vector(X, m):  ## X is struct from traj (units: bohr)
    
    Xc = ( torch.sum(X*m, 1)/ torch.sum(m[:,:,0]) ).unsqueeze(1)
    Xalig = X - Xc

    ## two group for up half (2 N-atoms) and down half (4 C-atoms)
    list_upmol = np.array([16,17])
    list_downmol = np.array([2,5,10,13])
    
    ## two group for forward half (2 C-atoms) and backward half (z C-atoms)
    list_forwmol = np.array([10,13])
    list_backmol = np.array([2,5])

    ## calc the geometrical center for each segment

    upmol_c = torch.mean(Xalig[:,list_upmol],1).unsqueeze(1)
    downmol_c = torch.mean(Xalig[:,list_downmol],1).unsqueeze(1)
    
    forwmol_c = torch.mean(Xalig[:,list_forwmol],1).unsqueeze(1)
    backmol_c = torch.mean(Xalig[:,list_backmol],1).unsqueeze(1)

    ## calc the frame
    
    vec_z = (upmol_c - downmol_c)/torch.sqrt(torch.sum(torch.square(upmol_c - downmol_c), -1)).unsqueeze(-1)
    vec_y = (forwmol_c - backmol_c)/torch.sqrt(torch.sum(torch.square(forwmol_c - backmol_c), -1)).unsqueeze(-1)
    vec_x = torch.cross(vec_z,vec_y)
    
    vec_x_norm = vec_x/torch.sqrt(torch.sum(torch.square(vec_x),-1)).unsqueeze(-1)
    vec_z_norm = vec_z
    vec_y_norm = torch.cross(vec_x_norm, vec_z_norm)

    return vec_x_norm, vec_y_norm, vec_z_norm, Xc
    
    

def num_gradients(f, Q):
    Q_alt = Q[np.array([1,0])]
    h = 1e-4
    grads = np.append((f(Q_alt[0]+h, Q_alt[1]) - f(Q_alt[0]-h, Q_alt[1]))/(2*h),
                      (f(Q_alt[0], Q_alt[1]+h) - f(Q_alt[0], Q_alt[1]-h))/(2*h))
    grads_alt = grads[np.array([1,0])]
    return grads_alt
    
def calcdihCNC(x_numpy):

    vec01 = x_numpy[:,16,:] - x_numpy[:,0,:]
    vec01 = vec01/np.expand_dims(np.sqrt(np.sum(np.square(vec01),-1)),-1)
    
    vec12 = x_numpy[:,17,:] - x_numpy[:,16,:]
    vec12 = vec12/np.expand_dims(np.sqrt(np.sum(np.square(vec12),-1)),-1)
    
    vec23 = x_numpy[:,1,:] - x_numpy[:,17,:]
    vec23 = vec23/np.expand_dims(np.sqrt(np.sum(np.square(vec23),-1)),-1)
    
    n1 = np.cross(vec01,vec12,-1)/np.expand_dims( np.sqrt(np.sum(np.square(np.cross(vec01,vec12,-1)),-1)), -1 )
    n2 = np.cross(vec12,vec23,-1)/np.expand_dims( np.sqrt(np.sum(np.square(np.cross(vec12,vec23,-1)),-1)), -1 )
    
    cos_ = np.sum(n1*n2,-1)
    
    sin_ = np.sum(np.cross(n1,n2,-1)*vec12,-1)
    
    theta = calctheta(cos_, sin_)
    
    ## remove the edg of dihedral at 68.5 deg.
    list_exedg = np.where(np.abs(theta) > 67.5*np.pi/180)[0]
    modify_theta = np.sign(theta[list_exedg])*(67.5*np.pi/180)
    theta[list_exedg] = modify_theta
    
    return theta  


def calctheta(cos_dih, sin_dih): ### give cos_or_sin_name // return the true angle value (rad.)
    diheddata = np.zeros_like(cos_dih)
    
    
    La = np.where(cos_dih > np.sqrt(2)/2)
    Lb = np.where(sin_dih > np.sqrt(2)/2)
    Lc = np.where(cos_dih < -np.sqrt(2)/2)
    Ld = np.where(sin_dih < -np.sqrt(2)/2)
    
    
    diheddata[La[0]] =         np.arcsin(sin_dih[La[0]]) #*180/np.pi
    diheddata[Lb[0]] =         np.arccos(cos_dih[Lb[0]]) #*180/np.pi
    diheddata[Lc[0]] = (np.pi-np.arcsin(sin_dih[Lc[0]])) #*180/np.pi
    diheddata[Ld[0]] =        -np.arccos(cos_dih[Ld[0]]) #*180/np.pi
    
    return diheddata


def num_gradients(zvalue, Q):
    
    h = 1e-4
    
    dz = interpn( (bond,bond,dihed), zvalue, Q+np.array([[h,0.,0.]]),  method = 'linear')-interpn( (bond,bond,dihed), zvalue, Q-np.array([[h,0.,0.]]),  method = 'linear')
    dzdq1 = dz/(2*h)
    
    dz = interpn( (bond,bond,dihed), zvalue, Q+np.array([[0.,h,0.]]),  method = 'linear')-interpn( (bond,bond,dihed), zvalue, Q-np.array([[0.,h,0.]]),  method = 'linear')
    dzdq2 = dz/(2*h)
    
    grads_num = np.concatenate((np.expand_dims(dzdq1,-1),
                                np.expand_dims(dzdq2,-1)),axis=-1)
    
    return grads_num


def Jacob_DXDQ(xyzcoord_Bohr, Q):
    Nsamp = Q.shape[0]

    dXdQ = np.zeros([Nsamp,Natoms,3,2])
    for iatom in range(Natoms):
        for ixyz in range(3):
        
            dXdQ[:,iatom, ixyz, :] = num_gradients(xyzcoord_Bohr[:,:,:,iatom, ixyz], Q)  ## shape: [Ntraj x 2]
            
            
    dXdQ = torch.tensor(dXdQ.reshape(Nsamp, Natoms*3, 2)).type(torch.float32)
    
    return dXdQ
    

def Get_RingMatrix(Nbead):
    ring_mat = np.zeros([Nbead, Nbead])
    ring_mat[np.arange(Nbead), np.arange(Nbead)] = np.ones([Nbead])*2.
    
    ring_mat[np.arange(Nbead-1), np.arange(Nbead-1)+1] = np.ones(Nbead-1)*(-1.)
    ring_mat[np.arange(Nbead-1)+1, np.arange(Nbead-1)] = np.ones(Nbead-1)*(-1.)
    ring_mat[0,Nbead-1] = -1.
    ring_mat[Nbead-1,0] = -1.

    return ring_mat


def Cmatrix(Nbead):
    C_mat = np.zeros([Nbead, Nbead])
    for j in range(1,Nbead+1):
        for k in range(Nbead):
            
            if k==0:
                Cjk = np.sqrt(1/Nbead)
            elif k < Nbead/2:
                Cjk = np.sqrt(2/Nbead) * np.cos(2*np.pi*j*k/Nbead)
            
            elif k == Nbead/2:
                Cjk = np.sqrt(1/Nbead)*(-1)**j
            else:
                Cjk = np.sqrt(2/Nbead) * np.sin(2*np.pi*j*k/Nbead)
            
            C_mat[j-1, k] = Cjk
            
    
    return C_mat
        
        
def Cmatrix_eigvals(Nbead):
    return ( 2*np.sin(np.arange(Nbead)*np.pi/Nbead) )**2


def beads_nmode_convert(Nring, Nbead, x, v, Matrix):
    ## input shape is  [ Nring, Nbead, Natoms, 3] or
    ##                 [ Nring, Nbead, 3×Natoms ] or 
    ##                 [ Nring×Nbead, Natoms, 3 ]
    
    ## output shape is [ Nring, Nbead, 3*Natoms ]
    
    ##    Matrix is C matrix (C_mat) % shape = ( Nbead × Nbead )
    ##    beads reps (x) to normal modes (xn):  xn = Mat^T × x  
    ##             Or
    ##  normal modes (xn) to beads reps (x):   x = Mat × xn
    
    x_ = torch.matmul( torch.tile(torch.tensor(Matrix.T).unsqueeze(0).type(torch.float32), [Nring,1,1 ]), 
                            x.reshape(Nring, Nbead, Natoms*3) )
    
    v_ = torch.matmul( torch.tile(torch.tensor(Matrix.T).unsqueeze(0).type(torch.float32), [Nring,1,1 ]), 
                            v.reshape(Nring, Nbead, Natoms*3) )
    
    return x_, v_

def CayleyAT_transform(Nring, Nbead, C_mat_eigval,ωn,Δt):
    
    
    A_mat = np.zeros([Nbead, 2, 2])
    A_mat[:,1,0] = -C_mat_eigval*(ωn**2)
    A_mat[:,0,1] = np.ones([Nbead])
    
    cayA = np.matmul(np.linalg.inv(np.eye(2) - Δt*A_mat/2), np.eye(2) + Δt*A_mat/2 )
    
    cayA_T = torch.tile(torch.tensor(np.transpose(cayA, [0,2,1])).type(torch.float32).unsqueeze(0), [Nring,1,1,1])
    
    return cayA_T


def free_RPstep(Nring, Nbead, x, v, Matrix, cayA_T):
    
    ## S1:
    ## beads to nomal modes
    x_v_n = beads_nmode_convert(Nring, Nbead, x, v, Matrix)
    
    ## S2:
    ## propagate for Δt
    x_v_n_ = torch.matmul( torch.concat((x_v_n[0].unsqueeze(-1),x_v_n[1].unsqueeze(-1)),axis=-1) , cayA_T )
    
    ## S3:
    ## normal modes to beads
    x_, v_ = beads_nmode_convert(Nring, Nbead, x_v_n_[:,:,:,0], x_v_n_[:,:,:,1], Matrix.T)
    
    return x_.reshape(Nring*Nbead, Natoms, 3), v_.reshape(Nring*Nbead, Natoms, 3)

    

def schnet_input(positions_x, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_, m, device):
    
    ## Nsamp : number of samples
    Nsamp = positions_x.shape[0]
    
    ### remove the mass center of positions_X @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    xnp = positions_x.cpu().detach().numpy()  ## nograds positions (ndarry)
    massc = torch.sum( torch.tensor(xnp).type(torch.float32)*m.cpu(),1).unsqueeze(1)/torch.sum(m.cpu()[:,:,0])
    positions_xc = positions_x - massc.to(device)
    ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    input_snet = {}
    input_snet['_positions'] = positions_xc.to(device)
    input_snet['_atomic_numbers'] = atomic_numbers_[:Nsamp].to(device)
    input_snet['_cell_offset'] = cell_offset_[:Nsamp].to(device)
    input_snet['_cell'] = cell_[:Nsamp].to(device)
    input_snet['_neighbors'] = neighbors_[:Nsamp].to(device)
    input_snet['_neighbor_mask'] = neighbor_mask_[:Nsamp].to(device)
    input_snet['_atom_mask'] = atom_mask_[:Nsamp].to(device)
    
    return input_snet


def Hamiltonian_output(positions_x, model, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_, m, device):
    E_ref = -2086.620067 
    
    input_snet = schnet_input(positions_x, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_, m, device)

    output_snet = model(input_snet)
    Energys_ = output_snet['energy'].cpu().detach()
    Forces_ = output_snet['gradients'].cpu().detach()*(-1)
    
    H_t = (Energys_.unsqueeze(2)-E_ref)*torch.tile(torch.eye(nstate).unsqueeze(0),[Ntraj,1,1])

    F_t = Forces_  
    
    return H_t, F_t #, NAC_t


#@@@@  if for all rings (Nring) --> positions_x (aver on beads): [Nring x Na x3] 
def Hamiltonian_output_nac(positions_x, model, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_, m, device):
    
    Nsamp = positions_x.shape[0]
    
    E_ref = -2086.620067 #-344.1 #-230.7012
    
    input_snet = schnet_input(positions_x,
                              atomic_numbers_,
                              cell_offset_,
                              cell_,
                              neighbors_,
                              neighbor_mask_,
                              atom_mask_,
                              m, device)

    output_snet = model(input_snet)

    Hess_ = output_snet['hessian'].cpu().detach()
    Energys_ = output_snet['energy'].cpu().detach()
    Forces_ = output_snet['gradients'].cpu().detach()*(-1)
    
    H_t = (Energys_.unsqueeze(2)-E_ref)*torch.tile(torch.eye(nstate).unsqueeze(0),[Nsamp,1,1])
    NAC_t = calc_NAC_viaHess(Energys_,Forces_,Hess_)
    F_t = Forces_
    
    return H_t.detach(), F_t.detach(), NAC_t.detach(), Hess_.detach()



#@@@@  if for all rings (Nring) --> positions_x (aver on beads): [Nring x Na x3] 
def Hamiltonian_output_nac(positions_x, Nsplit, model, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_, m, device, isNAMD):
    
    Nsamp = positions_x.shape[0]
    
    E_ref = -2086.620067 
    
    H_t = torch.zeros([Nsamp, nstate, nstate])
    F_t = torch.zeros([Nsamp, nstate, Natoms, 3])
    NAC_t = torch.zeros([Nsamp, nstate, nstate, Natoms, 3])
    Hess_t = torch.zeros([Nsamp, nstate, 3*Natoms, 3*Natoms])
    
    for batch_list in np.array_split(np.arange(Nsamp), Nsplit ):
        input_snet = schnet_input(positions_x[batch_list],
                              atomic_numbers_,
                              cell_offset_,
                              cell_,
                              neighbors_,
                              neighbor_mask_,
                              atom_mask_,
                              m, device)
        output_snet = model(input_snet)
        Energys_ = output_snet['energy'].cpu().detach()
        Forces_ = output_snet['gradients'].cpu().detach()*(-1)
        
        H_ = (Energys_.unsqueeze(2)-E_ref)*torch.tile(torch.eye(nstate).unsqueeze(0),[len(batch_list),1,1])
        if isNAMD==True:
            Hess_ = output_snet['hessian'].cpu().detach()
            Hess_t[batch_list] = Hess_
            NAC_ = calc_NAC_viaHess(Energys_,Forces_,Hess_)
            NAC_t[batch_list] = NAC_

        #### cat element into matrix#####
        H_t[batch_list] = H_
        F_t[batch_list] = Forces_
        
    
    return H_t.detach(), F_t.to(device).detach() , NAC_t.to(device).detach(), Hess_t.to(device).detach()



def vverlet_X(x_t, v_t, F_t, m, Δt):
    a_t = F_t/m
    x_t_Δt = x_t + v_t*Δt + 0.5*a_t*Δt**2
    
    return x_t_Δt

def vverlet_V(v_t, F_t, F_t_Δt, m, Δt):
    a_t = F_t/m
    a_t_Δt = F_t_Δt/m
    v_t_Δt = v_t + 0.5*(a_t + a_t_Δt)*Δt
    
    return v_t_Δt


def veloc_halfstep(v, F, m, Δt):
    
    return v + (F/m)*(Δt/2)


def Dipole_Moment(position_x, model_dp, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_, m, device):
    Nsamp = position_x.shape[0]
  
    position_x = position_x.detach()
    position_x.requires_grad = True
    sch_in = schnet_input(position_x, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_, m, device)
   
    Dipoles_ = model_dp(sch_in)['dipoles']
    
    DM = torch.zeros([Nsamp,nstate,nstate,1,3])
    DM[:,0,0,:] = Dipoles_[:,0:1]
    DM[:,0,1,:] = Dipoles_[:,1:2]
    DM[:,1,0,:] = Dipoles_[:,1:2]
    DM[:,1,1,:] = Dipoles_[:,2:3]
    
    DDipoles_ = torch.zeros([Nsamp,3,3,Natoms,3])
    
    for i in range(3):
        for j in range(3): 
            DDipoles_[:,i,j] = torch.autograd.grad(Dipoles_[:,i,j], position_x, 
                                                   grad_outputs=torch.ones([Nsamp,]).to(device),
                                                   retain_graph=True)[0]
    
    dμdx = torch.zeros([Nsamp,nstate,nstate,3,Natoms,3])
    
    dμdx[:,0,0] = DDipoles_[:,0]
    dμdx[:,0,1] = DDipoles_[:,1]
    dμdx[:,1,0] = DDipoles_[:,1]
    dμdx[:,1,1] = DDipoles_[:,2]
    
    position_x.requires_grad = False
    
    return DM, dμdx


def Dipole_Moment(positions, Nsplit, model_dp, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_, m, device):
    Nsamp = positions.shape[0]
    ### calc at cpu device
    
    DM = torch.zeros([Nsamp,nstate,nstate,1,3])
    dμdx = torch.zeros([Nsamp,nstate,nstate,3,Natoms,3])
    
    for batch_list in np.array_split(np.arange(Nsamp), Nsplit ):
        position_x = positions[batch_list].detach()
        position_x.requires_grad = True
        sch_in = schnet_input(position_x, atomic_numbers_, cell_offset_, cell_, neighbors_, neighbor_mask_, atom_mask_, m, device)
        #sch_in_cpu = {k:v.to('cpu') for k,v in sch_in.items()}
        Dipoles_ = model_dp(sch_in)['dipoles']
        
        #DM = torch.zeros([len(batch_list),nstate,nstate,1,3])
        DM[batch_list,0,0,:] = Dipoles_[:,0:1].cpu()
        DM[batch_list,0,1,:] = Dipoles_[:,1:2].cpu()
        DM[batch_list,1,0,:] = Dipoles_[:,1:2].cpu()
        DM[batch_list,1,1,:] = Dipoles_[:,2:3].cpu()
        
        DDipoles_ = torch.zeros([len(batch_list),3,3,Natoms,3])
        for i in range(3):
            for j in range(3):
                DDipoles_[:,i,j] = torch.autograd.grad(Dipoles_[:,i,j], position_x, 
                                                   grad_outputs=torch.ones([len(batch_list),]).to(device),
                                                   retain_graph=True)[0]
                
    
        #dμdx = torch.zeros([Nsamp,nstate,nstate,3,Natoms,3])
        
        dμdx[batch_list,0,0] = DDipoles_[:,0].cpu()
        dμdx[batch_list,0,1] = DDipoles_[:,1].cpu()
        dμdx[batch_list,1,0] = DDipoles_[:,1].cpu()
        dμdx[batch_list,1,1] = DDipoles_[:,2].cpu()
        
        
        position_x.requires_grad = False
        
    positions.requires_grad = False
    
    return DM.detach().to(device), dμdx.detach().to(device)



def actfunct(r):  ## r unit: bohr
    a = 1.
    b = 2.1*1.8897259885789233
    return -0.5*torch.tanh(r*a-b*a)+0.5

def sctfunct(r):
    
    n=9
    n_=16
    r0=2.1*1.8897259885789233
    
    return (1-(r/r0)**n)/(1-(r/r0)**n_)


def fmulti_funct(rx,ry):
    scale_L = 1.8897259885789233
    a = 50/scale_L
    r0 = 2.8*scale_L
    
    Z = torch.tanh(-a*(torch.sqrt( rx**2 + ry**2 ) - r0))
    
    return torch.maximum(Z,torch.zeros_like(Z))

def fmulti_funct_circle(rx,ry):
    scale_L = 1.8897259885789233
    a = 50/scale_L
    r0 = 0.95*scale_L 
    
    rcenter = 1.3*scale_L
    
    Z = torch.tanh(-a*(torch.sqrt( (rx-rcenter)**2 + (ry-rcenter)**2 ) - r0))
    
    return torch.maximum(Z,torch.zeros_like(Z))

def H_MCH(H, μ, E_laser):
    
    return H.to(device) - torch.sum(torch.sum(μ.to(device)*E_laser.to(device),-1),-1)


def H_Diagonal(Hmch):
    
    eigvals_ = torch.linalg.eigh(Hmch)[0]
    eigvecs_ = torch.linalg.eigh(Hmch)[1]
    
    return eigvals_, eigvecs_

def zlowdin(M):
    
    M = M.to(torch.complex64)
    
    M_hermi = torch.transpose(torch.conj(M),1,2)
    S = torch.matmul(M_hermi,M)
    s, u = torch.linalg.eig(S)
    s_inv_half = (1/torch.sqrt(s).unsqueeze(1))*np.eye(nstate,nstate)
    
    s_inv_half = (s_inv_half.detach()).type(torch.complex64)
    
    u_hermi = torch.transpose(torch.conj(u),1,2)
    
    
    M_ort = torch.matmul(torch.matmul(torch.matmul(u,s_inv_half), u_hermi), M)
    M_lowdin = M_ort/torch.sqrt(torch.sum(torch.abs(M_ort),1)).unsqueeze(1)
    
    return M_lowdin.to(torch.float32)


def Phase_correct_U(eigvecs_t, eigvecs_t_Δt, eigvals_t_Δt):

    diag_degen_diff=1e-9
    
    S = torch.matmul(torch.transpose(torch.conj(eigvecs_t_Δt),1,2), eigvecs_t)
    
    E_val = eigvals_t_Δt.unsqueeze(1)
    
    for i in range(nstate):
        for j in  range(nstate):
            mark_list = np.int32((torch.abs(S.cpu())[:,i:i+1,j:j+1])**2 >0.5)*np.int64(torch.abs(E_val.cpu()[:,:,i:i+1]-E_val.cpu()[:,:,j:j+1]) > diag_degen_diff)
            
            if i==j:
                continue
            
            else:

                S_gather = torch.zeros(S.cpu().size())
                S_gather[:,:,0] = S[:,:,1].cpu().detach()
                S_gather[:,:,1] = S[:,:,0].cpu().detach()
                S = S_gather*mark_list + S.cpu().detach()*(1-mark_list)
                S = S.type(torch.float32)
                
                E_val_gather = torch.zeros(E_val.cpu().size())
                E_val_gather[:,:,0] = E_val[:,:,1].cpu().detach()
                E_val_gather[:,:,1] = E_val[:,:,0].cpu().detach()
                
                E_val = E_val_gather*mark_list + E_val.cpu().detach()*(1-mark_list)
                E_val = E_val.type(torch.float32)
            
    Mask = np.int32(torch.abs(E_val.cpu()[:,:,0:1]-E_val.cpu()[:,:,0:1])<diag_degen_diff)*np.array([[1,0],[0,0]]) + \
    np.int32(torch.abs(E_val.cpu()[:,:,0:1]-E_val.cpu()[:,:,1:2])<diag_degen_diff)*np.array([[0,1],[0,0]]) +\
    np.int32(torch.abs(E_val.cpu()[:,:,1:2]-E_val.cpu()[:,:,0:1])<diag_degen_diff)*np.array([[0,0],[1,0]]) +\
    np.int32(torch.abs(E_val.cpu()[:,:,1:2]-E_val.cpu()[:,:,1:2])<diag_degen_diff)*np.array([[0,0],[0,1]])
    
    P_project = (S*Mask).type(torch.float32)
    P_project_oth = zlowdin(P_project).to(device)

    eigvecs_t_Δt_correct = torch.matmul(eigvecs_t_Δt, P_project_oth)
    
    return eigvecs_t_Δt_correct, P_project_oth



def Interpolation_U(H_old, U_old, H_now):
                 #Hmch_t, eigvecs_t, Hmch_t_Δt

    U_old_dummy = U_old
    
    Nsubstep = 20
    for istep in range(1,Nsubstep+1):
        H = H_old + (H_now-H_old)*istep/Nsubstep ## interpolat overlap H
        
        eigval_H, eigvec_H = H_Diagonal(H)

        U, P_proj = Phase_correct_U(U_old_dummy, eigvec_H,eigval_H)
                
        U_old_dummy = U
        
    return U, eigval_H, P_proj 


def F_Diagonal(F, dμdx, E_laser, eigvecs_correct):

    Nsamp = F.shape[0]
    F_diag = torch.zeros([Nsamp,nstate,nstate, Natoms,3])
    F_diag[:,0,0,:] = F[:,0,:]
    F_diag[:,1,1,:] = F[:,1,:]
    F_diag = F_diag.to(device)
    
    F_diag = F_diag + torch.sum(dμdx*E_laser,-3)


    ev_expand = torch.tile(eigvecs_correct.unsqueeze(3).unsqueeze(3),(1,1,1,Natoms,3))

    F_diag = torch.matmul(torch.transpose(ev_expand.permute(0,3,4,1,2),3,4),
                          torch.matmul(F_diag.permute(0,3,4,1,2), ev_expand.permute(0,3,4,1,2)))

    F_diag = F_diag.permute(0,3,4,1,2)[:,[0,1],[0,1],:,:]
    
    return F_diag


def propagator(H_isub, K_isub, μ_isub, laser_isub, Δt_small):

    #Hmch 
    Hmch_isub = H_isub.to(device) - torch.sum(torch.sum(μ_isub.to(device)*laser_isub.to(device), -1), -1)
    #laser(t_isub, OMega, fwhm, t_c)
    
    expH = torch.linalg.matrix_exp(-(Hmch_isub*1j + K_isub)*Δt_small)
    
    return expH


def Rtotal(H_t, H_t_Δt, NAC_t, NAC_t_Δt, v_t, v_t_Δt, μ_t, μ_t_Δt, eigvecs_t, eigvecs_t_Δt, laser_t, laser_t_Δt, Δt, t):
    
    Nsamp = eigvecs_t.shape[0]
    eigvecs_t = eigvecs_t.to(torch.complex64)
    eigvecs_t_Δt = eigvecs_t_Δt.to(torch.complex64)
    
    ## NAC
    K_t = torch.sum(torch.sum(NAC_t*(v_t.unsqueeze(1).unsqueeze(1)),3),3)
    K_t_Δt = torch.sum(torch.sum(NAC_t_Δt*(v_t_Δt.unsqueeze(1).unsqueeze(1)),3),3)
    
    Nsubstep = 25
    Δt_small = Δt/Nsubstep
    
    R_total = torch.tile(torch.eye(nstate).unsqueeze(0).type(torch.complex64),[Nsamp,1,1]).to(device)
    
    for isub in range(1,Nsubstep+1):
        
        H_isub = H_t + (H_t_Δt - H_t)*isub/Nsubstep
        K_isub = K_t + (K_t_Δt - K_t)*isub/Nsubstep
        laser_isub = laser_t + (laser_t_Δt - laser_t)*isub/Nsubstep
        
        ## Dipole moment
        μ_isub = μ_t + (μ_t_Δt - μ_t)*isub/Nsubstep
        t_isub = t + Δt*isub/Nsubstep
        
        expH = propagator(H_isub, K_isub, μ_isub, laser_isub, Δt_small)
        R_total = torch.matmul(expH, R_total)
    
    R_total = torch.matmul( torch.matmul(torch.transpose(torch.conj(eigvecs_t_Δt) ,1,2),R_total), eigvecs_t)
    
    return R_total


def update_coefficient(R_total,coefficient):
    
    return torch.matmul(R_total,coefficient)


def Prob_hop(coefficient_t, coefficient_t_Δt, R_total):
    Nsamp = coefficient_t.shape[0]
    
    Prob_true = torch.zeros([Nsamp,nstate,nstate])
    
    x = torch.transpose(torch.conj(coefficient_t)*coefficient_t, 1,2)
    w = torch.transpose(torch.conj(coefficient_t_Δt)*coefficient_t_Δt, 1,2)
    
    R_diag = torch.sum(R_total*torch.eye(nstate).to(device).unsqueeze(0),2).unsqueeze(1)
    
    j = torch.transpose(coefficient_t_Δt,1,2)*torch.conj(R_diag)*torch.transpose(torch.conj(coefficient_t),1,2)
    z = torch.real( torch.transpose(torch.conj(coefficient_t),1,2)*torch.conj(R_total)*coefficient_t_Δt )
    
    y = torch.real(x - j)
    Prob = (1-torch.real(w/x))*torch.real(z/y)
    
    Prob = torch.where(torch.isnan(Prob), torch.full_like(Prob, 0), Prob)
    Prob = torch.where(torch.isinf(Prob), torch.full_like(Prob, 0), Prob)
    #print(Prob)
    
    Prob_true[:,1,0] = Prob[:,1,0]
    Prob_true[:,0,1] = Prob[:,0,1]
    
    return Prob_true


def update_activatestate(activatestate_index, seqtraj_index, Prob, H, v, m, random_num):

    laser_hop_width = 1/27.2114
    OMega = 15/627.51
    compara_p_r = torch.tensor(Prob.detach().numpy() > random_num).to(torch.int32)
    hop_index = compara_p_r[seqtraj_index, list(1-np.int32(activatestate_index)), activatestate_index ]
    
    ## determine laser or nonadiabatic hopping
    deter_laser = np.int32(np.abs(torch.abs((H[:,1,1] - H[:,0,0])).detach().numpy() - OMega) < laser_hop_width)
    hop_index_laserinduced = hop_index.detach().numpy()*deter_laser
    hop_index_nonadiabatic = hop_index.detach().numpy() - hop_index_laserinduced
    
    ## check the frustrated hopping
    
    activatestate_index_new = list(np.int32(activatestate_index) + ((-1)**np.int32(activatestate_index))*hop_index_nonadiabatic )
    
    
    GAP_hop = H[seqtraj_index,activatestate_index_new,activatestate_index_new] - H[seqtraj_index,activatestate_index,activatestate_index]
    GAP_hop = GAP_hop.detach().numpy()
    
    
    E_kin = torch.sum(torch.sum(0.5*m*v**2,1),1).cpu().detach().numpy()
    
    deter_normalhop = np.int32(E_kin > GAP_hop)
    
    hop_index_normalhop = hop_index_nonadiabatic*deter_normalhop
    hop_index_frusthop = hop_index_nonadiabatic - hop_index_normalhop
    
    
    f_factor = np.sqrt(np.abs((E_kin - GAP_hop)/E_kin))*hop_index_normalhop + (1-hop_index_normalhop)
    f_factor = torch.tensor(f_factor).type(torch.float32).unsqueeze(1).unsqueeze(1)
    
    hop_index_True = hop_index_normalhop + hop_index_laserinduced
    
    
    activatestate_index_next = list(np.int32(activatestate_index) + ((-1)**np.int32(activatestate_index))*hop_index_True)
    
    return activatestate_index_next, f_factor


def EDC_decoherence(coefficients, activatestate_index, seqtraj_index, H, eigvals, v, m, Δt):

    c_α = coefficients[seqtraj_index,list(1-np.int32(activatestate_index)),:]
    c_β = coefficients[seqtraj_index,activatestate_index,:]
    
    E_kin = torch.sum(torch.sum(0.5*m*v**2,1),1)
    
    gap = torch.abs(eigvals[:,1] - eigvals[:,0])
#    gap = torch.abs(H[:,1,1] - H[:,0,0])
    
    scale_α = torch.exp(-0.5*Δt*gap*(1/(1 + 0.1/E_kin)))
    c_α_p = scale_α.unsqueeze(1)*c_α
    
    scale_β = torch.sqrt(1 - torch.abs(c_α_p)**2)/torch.abs(c_β)
    c_β_p = scale_β*c_β
    
    coefficients[seqtraj_index,list(1-np.int32(activatestate_index)),:] = c_α_p
    coefficients[seqtraj_index,activatestate_index,:] = c_β_p
    
    return coefficients


def get_initconds(paths_initcoord, paths_initveloc):
    initial_coord = np.zeros([1,Natoms,3])
    initial_veloc = np.zeros([1,Natoms,3])    
    
    for i in range(Ntraj):
        COORD = np.loadtxt(f'{paths_initcoord}/X_{i}.txt')
        coord_i = COORD.reshape(1,Natoms,3)
        initial_coord = np.concatenate((initial_coord,coord_i),axis=0)
    
        VELOC = np.loadtxt(f'{paths_initveloc}/V_{i}.txt')
        veloc_i = VELOC.reshape(1,Natoms,3)
        initial_veloc = np.concatenate((initial_veloc,veloc_i),axis=0)

    initial_coord = initial_coord[1:]
    initial_veloc = initial_veloc[1:]
    
    return initial_coord, initial_veloc