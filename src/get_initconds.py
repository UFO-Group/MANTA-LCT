#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import numpy as np

# ===================== 可改参数区 =====================
TRAJ_GLOB = "TRAJ_*"          # 轨迹目录匹配（可改成 /path/to/run/TRAJ_*）
XYZ_NAME = "output.xyz"       # 结构文件名
VELOC_NAME = "veloc"          # 速度文件名
OUT_INITCONDS = "initconds"   # 总输出文件名（写在当前目录）
FLOAT_FMT = "{: .8f}"         # 坐标/速度输出格式
# =====================================================

# 第二列：原子序数（你例子里 C=6.0, H=1.0）
ATOMIC_NUMBER = {
    "H": 1.0, "He": 2.0,
    "Li": 3.0, "Be": 4.0, "B": 5.0, "C": 6.0, "N": 7.0, "O": 8.0, "F": 9.0, "Ne": 10.0,
    "Na": 11.0, "Mg": 12.0, "Al": 13.0, "Si": 14.0, "P": 15.0, "S": 16.0, "Cl": 17.0, "Ar": 18.0,
    "K": 19.0, "Ca": 20.0,
    "Sc": 21.0, "Ti": 22.0, "V": 23.0, "Cr": 24.0, "Mn": 25.0, "Fe": 26.0, "Co": 27.0, "Ni": 28.0,
    "Cu": 29.0, "Zn": 30.0,
    "Br": 35.0, "I": 53.0,
}

def extract_traj_index(traj_dir: str) -> int:
    """
    从目录名提取 Index 数字：
      TRAJ_005 -> 5
      TRAJ_12  -> 12
    若提取失败抛异常。
    """
    base = os.path.basename(os.path.normpath(traj_dir))
    m = re.search(r"(\d+)", base)
    if not m:
        raise ValueError(f"目录名中未找到数字: {base}")
    return int(m.group(1))


def read_last_xyz_frame(xyz_path: str):
    """
    读取多帧 XYZ 的最后一帧：
      Natoms
      comment (忽略)
      Natoms 行：symbol x y z (可带额外列，忽略)
    返回：natoms, symbols(list[str]), pos(np.ndarray natoms x 3)
    """
    last_natoms = None
    last_symbols = None
    last_pos = None

    with open(xyz_path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            try:
                natoms = int(line.split()[0])
            except ValueError:
                raise ValueError(f"[ERROR] {xyz_path} Natoms 解析失败：{line}")

            comment = f.readline()
            if not comment:
                raise ValueError(f"[ERROR] {xyz_path} Natoms 后缺 comment（文件可能截断）")

            symbols = []
            pos = np.zeros((natoms, 3), dtype=float)

            for i in range(natoms):
                atom_line = f.readline()
                if not atom_line:
                    raise ValueError(f"[ERROR] {xyz_path} 原子行不足（帧不完整）")
                parts = atom_line.split()
                if len(parts) < 4:
                    raise ValueError(f"[ERROR] {xyz_path} 原子行列数不足：{atom_line.strip()}")

                symbols.append(parts[0])
                pos[i, 0] = float(parts[1])
                pos[i, 1] = float(parts[2])
                pos[i, 2] = float(parts[3])

            last_natoms = natoms
            last_symbols = symbols
            last_pos = pos

    if last_natoms is None:
        raise ValueError(f"[ERROR] {xyz_path} 未读取到任何帧。")

    return last_natoms, last_symbols, last_pos


def read_last_veloc_frame(veloc_path: str, natoms: int) -> np.ndarray:
    """
    读取 veloc 文件最后一个 Index 块：
        Index k
        (natoms 行) vx vy vz
    返回：vel (natoms x 3)
    """
    last_block_pos = None

    with open(veloc_path, "r") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip().startswith("Index"):
                last_block_pos = pos

    if last_block_pos is None:
        raise ValueError(f"[ERROR] {veloc_path} 中未找到任何 'Index' 块。")

    velocities = []
    with open(veloc_path, "r") as f:
        f.seek(last_block_pos)
        header = f.readline().strip()  # Index xxx
        if not header.startswith("Index"):
            raise RuntimeError(f"[ERROR] {veloc_path} seek 定位失败，未读到 Index 行。")

        while len(velocities) < natoms:
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            vx, vy, vz = map(float, parts[:3])
            velocities.append([vx, vy, vz])

    if len(velocities) != natoms:
        raise ValueError(
            f"[ERROR] {veloc_path} 最后 Index 块读取到 {len(velocities)} 行速度，期望 natoms={natoms}。"
        )

    return np.array(velocities, dtype=float)


def write_one_block(f, index: int, symbols, pos: np.ndarray, vel: np.ndarray):
    """
    向已打开文件 f 追加写一个 initconds block（8列，不写质量列）：
      Index    k
      Atoms
      sym Z x y z vx vy vz
    """
    natoms = len(symbols)
    if pos.shape != (natoms, 3):
        raise ValueError("[ERROR] pos shape 不匹配")
    if vel.shape != (natoms, 3):
        raise ValueError("[ERROR] vel shape 不匹配")

    f.write(f"Index    {index}\n")
    f.write("Atoms\n")
    for i in range(natoms):
        sym = symbols[i]
        if sym not in ATOMIC_NUMBER:
            raise KeyError(f"[ERROR] 未在 ATOMIC_NUMBER 中找到元素 '{sym}'，请补充映射。")

        Z = ATOMIC_NUMBER[sym]
        x, y, z = pos[i]
        vx, vy, vz = vel[i]

        # 8列：sym Z x y z vx vy vz
        f.write(
            f"{sym:<2s} "
            f"{Z:>4.1f} "
            f"{FLOAT_FMT.format(x)} "
            f"{FLOAT_FMT.format(y)} "
            f"{FLOAT_FMT.format(z)} "
            f"{FLOAT_FMT.format(vx)} "
            f"{FLOAT_FMT.format(vy)} "
            f"{FLOAT_FMT.format(vz)}\n"
        )


def main():
    traj_dirs = sorted([d for d in glob.glob(TRAJ_GLOB) if os.path.isdir(d)])
    if not traj_dirs:
        raise SystemExit(f"[ERROR] 未找到目录匹配：{TRAJ_GLOB}")

    ok, bad = 0, 0

    # 为了确保 Index 顺序一致：按提取到的数字排序
    dirs_with_idx = []
    for d in traj_dirs:
        try:
            idx = extract_traj_index(d)
            dirs_with_idx.append((idx, d))
        except Exception as e:
            print(f"[SKIP] {d}: {e}")
            bad += 1

    dirs_with_idx.sort(key=lambda x: x[0])

    with open(OUT_INITCONDS, "w") as out:
        for idx, d in dirs_with_idx:
            xyz_path = os.path.join(d, XYZ_NAME)
            vel_path = os.path.join(d, VELOC_NAME)

            if not os.path.exists(xyz_path):
                print(f"[SKIP] {d}: 缺少 {XYZ_NAME}")
                bad += 1
                continue
            if not os.path.exists(vel_path):
                print(f"[SKIP] {d}: 缺少 {VELOC_NAME}")
                bad += 1
                continue

            try:
                natoms, symbols, pos = read_last_xyz_frame(xyz_path)
                vel = read_last_veloc_frame(vel_path, natoms)
                write_one_block(out, idx, symbols, pos, vel)
                print(f"[OK] {d} -> block Index={idx} (natoms={natoms})")
                ok += 1
            except Exception as e:
                print(f"[FAIL] {d}: {e}")
                bad += 1

    print(f"\nDone. Wrote '{OUT_INITCONDS}'. OK blocks={ok}, FAIL/SKIP={bad}")


if __name__ == "__main__":
    main()
