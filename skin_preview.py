# -*- coding: utf-8 -*-
"""
Minecraft 经典 64x64 皮肤 3D 预览器(双层: 第一层身体 + 第二层帽子/外套/袖子/裤腿)
用法:
    python skin_preview.py [皮肤png路径]
不带参数则默认读取本目录 skins 文件夹里的第一张 png。

交互(与 LittleSkin / HMCL 预览一致):
    左键拖动  -> 旋转模型
    鼠标滚轮  -> 缩放
    右键拖动  -> 平移视角
    按 W      -> 切换线框/实体显示
    按 R      -> 恢复默认视角
    按 Esc    -> 退出
"""

import sys
import os
import math
import gc
import time

import pygame
import numpy as np
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective


# ---------------------------------------------------------------------------
# 经典 64x64 皮肤 UV 映射表(双层)。
# 每个身体部位: size=(宽x,高y,深z), pos=模型中心坐标。
#   uv["base"]    = 第一层(身体本体) 各面在皮肤图上的像素区域
#   uv["overlay"] = 第二层(帽子/外套/袖子/裤腿) 各面区域
# 区域 rect = (left, top, width, height)，坐标原点为皮肤图左上角。
# overlay 坐标由同尺寸 box 平移推导:
#   hat     整体 +32x ; 躯干/右臂/右腿 整体 +16y ; 左臂 +16x ; 左腿 -16x
# ---------------------------------------------------------------------------
def _r(x, y, w, h):
    return (x, y, w, h)


PARTS = {
    # 头: 8x8x8
    "head": dict(
        size=(8, 8, 8),
        pos=(0, 28, 0),
        uv=dict(
            base=dict(
                top=_r(8, 0, 8, 8),
                bottom=_r(16, 0, 8, 8),
                right=_r(0, 8, 8, 8),
                front=_r(8, 8, 8, 8),
                left=_r(16, 8, 8, 8),
                back=_r(24, 8, 8, 8),
            ),
            overlay=dict(  # hat 层, 整体 +32x
                top=_r(40, 0, 8, 8),
                bottom=_r(48, 0, 8, 8),
                right=_r(32, 8, 8, 8),
                front=_r(40, 8, 8, 8),
                left=_r(48, 8, 8, 8),
                back=_r(56, 8, 8, 8),
            ),
        ),
    ),
    # 身体: 8x12x4
    "body": dict(
        size=(8, 12, 4),
        pos=(0, 18, 0),
        uv=dict(
            base=dict(
                top=_r(20, 16, 8, 4),
                bottom=_r(28, 16, 8, 4),
                right=_r(16, 20, 4, 12),
                front=_r(20, 20, 8, 12),
                left=_r(28, 20, 4, 12),
                back=_r(32, 20, 8, 12),
            ),
            overlay=dict(  # 外套, 整体 +16y
                top=_r(20, 32, 8, 4),
                bottom=_r(28, 32, 8, 4),
                right=_r(16, 36, 4, 12),
                front=_r(20, 36, 8, 12),
                left=_r(28, 36, 4, 12),
                back=_r(32, 36, 8, 12),
            ),
        ),
    ),
    # 右臂(角色右侧): slim(Alex)纤细手臂, 宽3
    # 原版展开公式(单原点 + 盒子尺寸, 见 setUVs): 原点 u=40,v=16, (w,h,d)=(3,12,4)
    #   top    = (u+d,   v,        w, d) = (44,16,3,4)
    #   bottom = (u+w+d, v,        w, d) = (47,16,3,4)
    #   right  = (u,     v+d,      d, h) = (40,20,4,12)
    #   front  = (u+d,   v+d,      w, h) = (44,20,3,12)
    #   left   = (u+w+d, v+d,      d, h) = (47,20,4,12)
    #   back   = (u+w+2d,v+d,      w, h) = (51,20,3,12)
    # ★ 宽度从 4 变 3 时, bottom/left/back 都要跟着左移一格(48→47, 52→51),
    #   不是"矩形不动只读 3 列"。之前写成 48/48/52 就是这里的偏差来源。
    "right_arm": dict(
        size=(3, 12, 4),
        pos=(-5.5, 18, 0),
        uv=dict(
            base=dict(
                top=_r(44, 16, 3, 4),
                bottom=_r(47, 16, 3, 4),
                right=_r(40, 20, 4, 12),
                front=_r(44, 20, 3, 12),
                left=_r(47, 20, 4, 12),
                back=_r(51, 20, 3, 12),
            ),
            overlay=dict(  # 袖子, 原点 u=40,v=32 (base 整体 +16y)
                top=_r(44, 32, 3, 4),
                bottom=_r(47, 32, 3, 4),
                right=_r(40, 36, 4, 12),
                front=_r(44, 36, 3, 12),
                left=_r(47, 36, 4, 12),
                back=_r(51, 36, 3, 12),
            ),
        ),
    ),
    # 左臂(角色左侧): slim(Alex)纤细手臂, 宽3
    # 原点 u=32,v=48, (w,h,d)=(3,12,4):
    #   top    = (36,48,3,4)   bottom = (39,48,3,4)
    #   right  = (32,52,4,12)  front  = (36,52,3,12)
    #   left   = (39,52,4,12)  back   = (43,52,3,12)
    "left_arm": dict(
        size=(3, 12, 4),
        pos=(5.5, 18, 0),
        uv=dict(
            base=dict(
                top=_r(36, 48, 3, 4),
                bottom=_r(39, 48, 3, 4),
                right=_r(32, 52, 4, 12),
                front=_r(36, 52, 3, 12),
                left=_r(39, 52, 4, 12),
                back=_r(43, 52, 3, 12),
            ),
            overlay=dict(  # 袖子, 原点 u=48,v=48
                top=_r(52, 48, 3, 4),
                bottom=_r(55, 48, 3, 4),
                right=_r(48, 52, 4, 12),
                front=_r(52, 52, 3, 12),
                left=_r(55, 52, 4, 12),
                back=_r(59, 52, 3, 12),
            ),
        ),
    ),
    # 右腿(角色右侧): 4x12x4
    "right_leg": dict(
        size=(4, 12, 4),
        pos=(-2, 6, 0),
        uv=dict(
            base=dict(
                top=_r(4, 16, 4, 4),
                bottom=_r(8, 16, 4, 4),
                right=_r(0, 20, 4, 12),
                front=_r(4, 20, 4, 12),
                left=_r(8, 20, 4, 12),
                back=_r(12, 20, 4, 12),
            ),
            overlay=dict(  # 裤腿, 整体 +16y
                top=_r(4, 32, 4, 4),
                bottom=_r(8, 32, 4, 4),
                right=_r(0, 36, 4, 12),
                front=_r(4, 36, 4, 12),
                left=_r(8, 36, 4, 12),
                back=_r(12, 36, 4, 12),
            ),
        ),
    ),
    # 左腿(角色左侧): 4x12x4
    "left_leg": dict(
        size=(4, 12, 4),
        pos=(2, 6, 0),
        uv=dict(
            base=dict(
                top=_r(20, 48, 4, 4),
                bottom=_r(24, 48, 4, 4),
                right=_r(16, 52, 4, 12),
                front=_r(20, 52, 4, 12),
                left=_r(24, 52, 4, 12),
                back=_r(28, 52, 4, 12),
            ),
            overlay=dict(  # 裤腿, 整体 -16x
                top=_r(4, 48, 4, 4),
                bottom=_r(8, 48, 4, 4),
                right=_r(0, 52, 4, 12),
                front=_r(4, 52, 4, 12),
                left=_r(8, 52, 4, 12),
                back=_r(12, 52, 4, 12),
            ),
        ),
    ),
}

SKIN_SIZE = 64  # 经典皮肤尺寸


ARM_STYLES = ("slim", "classic")  # 纤细(默认) / 宽形


# ---------------------------------------------------------------------------
# 映射总表: 3D 面名 ↔ 贴图上的 UV 块 ↔ 该面上 u 的增长方向
#
# 这里的"面名"是 3D 方向名, 不是模板图上的字面标签 —— 这是之前渲染错位的根:
# 旧代码把 UV 表里叫 right 的块直接贴给 3D 的 right(+x) 面, 而方块模板的
# right 块其实是给 -x 面用的。于是两臂的侧面/背面的横向取样一起反了,
# 背视里红线黄线就呈现"对角"。下面的关系已被三方资料共同印证:
#   · Minecraft wiki 皮肤页: 手臂/腿的 "right" 是这些部位的"外面", "left" 是"里面"
#   · skin3d / skinview3d 的 setUVs: 原点 u 那块 → -x(左, 本地 west), u+w+d 那块 → +x(右, 本地 east)
#   · mcskins.org 皮肤编辑器内嵌模型(alex64/steve64)逐面 UV, 数值与 PARTS 完全吻合
#
#  3D 面名   法线        贴图块(公式)      u 增长方向(从外面看)   v 方向
#  ---------------------------------------------------------------------------
#  left     (-1, 0, 0)  right = (u, ...)   从 z=-d/2 往 +z        上→下
#  front    ( 0, 0, 1)  front = (u+d, ...) 从 x=-w/2 往 +x        上→下
#  right    (+1, 0, 0)  left  = (u+w+d,...)从 z=+d/2 往 -z        上→下
#  back     ( 0, 0,-1)  back  = (u+w+2d,...)从 x=+w/2 往 -x       上→下 ← 镜像
#  top      ( 0, 1, 0)  top                      --               从 z=-d/2 往 +z
#  bottom   ( 0,-1, 0)  bottom                   --               从 z=-d/2 往 +z
#
# 注意 back 这一行是唯一需要镜像的: 一圈四个侧面首尾相接时, 必然有一面要反着
# 贴(u0 落在 +x 端而不是 -x 端), 原版这一面就是背面。这条由用户实测确认
# (游戏内/HMCL 背视花纹方向与本程序原来的"同向"不同, 对调后才一致)。
#
# 要手动改坐标: 改 PARTS 里各部位的 uv[base/overlay][面名]; 想整体换某个面的
# 贴图块, 改 FACE_UV_KEY; 想让某个面左右对调, 改 FACE_UV_FLIP。
# 改完跑 check_uv_table(True); 面朝向相关的改动请配合 pv_uvpipe.py 复核
# (它逐像素比对"屏幕上画的颜色"和"_pick_texel 命中的纹素", 镜像错位会立刻暴露)。
FACE_UV_KEY = {
    "right": "left",  # +x 面 → 公式里的 left 块(u+w+d, 皮肤块集合的"里面"块)
    "left": "right",  # -x 面 → 公式里的 right 块(原点 u, 皮肤块集合的"外面"块)
    "front": "front",
    "back": "back",
    "top": "top",
    "bottom": "bottom",
}

# 哪几面在横向上把 u 反过来(True = u0/u1 对调)。
# 渲染(build_geometry_parts)、鼠标拾取(_pick_texel)、像素高亮(_face_cell_quad)、
# 渲染、鼠标拾取、高亮 都读这张表, 所以改这里一处就能整体保持一致。
FACE_UV_FLIP = {
    "back": True,  # 背面: u0 在 +x 端、u1 在 -x 端(与 front 相反)
}


def face_u_flip(face):
    """该 3D 面是否需要横向对调 UV(单一数据源)。"""
    return bool(FACE_UV_FLIP.get(face, False))


# ---------------------------------------------------------------------------
# 原版 UV 展开公式(与 vanilla ModelPart / skinview3d 的 setUVs 一致):
#   每个部位只给一个"贴图原点"(u, v) 和盒子尺寸 (w, h, d), 六个面由公式算出:
#     top    = (u + d,       v,       w, d)
#     bottom = (u + w + d,   v,       w, d)
#     right  = (u,           v + d,   d, h)
#     front  = (u + d,       v + d,   w, h)
#     left   = (u + w + d,   v + d,   d, h)
#     back   = (u + w + 2*d, v + d,   w, h)
#   所以盒子某一维变化时, 后面那几个面会跟着平移 —— 手臂 4→3 宽, bottom/left/back
#   都会左移 1 格。PARTS 里手写的数值必须满足这个公式, 否则就是位置偏差
#   (check_uv_table() 用来自检)。
# ---------------------------------------------------------------------------
UV_ORIGIN = {
    # 部位: (base 原点, overlay 原点)
    "head": ((0, 0), (32, 0)),
    "body": ((16, 16), (16, 32)),
    "right_arm": ((40, 16), (40, 32)),
    "left_arm": ((32, 48), (48, 48)),
    "right_leg": ((0, 16), (0, 32)),
    "left_leg": ((16, 48), (0, 48)),
}


def expected_uv_rect(origin, face, size):
    """按原版公式算出某面的 UV 矩形, 用于和手写的 PARTS 数值对照。"""
    u, v = origin
    w, h, d = size
    return {
        "top": (u + d, v, w, d),
        "bottom": (u + w + d, v, w, d),
        "right": (u, v + d, d, h),
        "front": (u + d, v + d, w, h),
        "left": (u + w + d, v + d, d, h),
        "back": (u + w + 2 * d, v + d, w, h),
    }[face]


def check_uv_table(verbose=True):
    """自检: 手写的 PARTS 是否与展开公式完全一致。返回不一致条目列表。
    手改 PARTS 的坐标后跑一次, 能立刻看出哪一面偏了。"""
    bad = []
    for style in ARM_STYLES:
        parts = build_parts(style)
        for name, p in parts.items():
            size = tuple(p["size"])
            for layer, key in (("base", 0), ("overlay", 1)):
                origin = UV_ORIGIN[name][key]
                for face, rect in p["uv"][layer].items():
                    exp = expected_uv_rect(origin, face, size)
                    if tuple(rect) != exp:
                        bad.append((style, name, layer, face, tuple(rect), exp))
                        if verbose:
                            print(
                                f"[不一致] {style:<7}{name}/{layer}/{face:<6} "
                                f"写={tuple(rect)} 公式={exp}"
                            )
    if verbose:
        print(
            f"UV 表自检: {'全部符合公式' if not bad else str(len(bad)) + ' 处不一致'}"
        )
    return bad


def build_parts(arm_style="slim"):
    """按手臂宽度类型返回 PARTS。经典(Steve)手臂宽4px, slim(Alex)宽3px。"""
    import copy

    parts = copy.deepcopy(PARTS)
    if arm_style == "classic":
        for side, arm in (("right", "right_arm"), ("left", "left_arm")):
            parts[arm]["size"] = (4, 12, 4)
            parts[arm]["pos"] = (-6.0 if side == "right" else 6.0, 18, 0)
        # 宽形(w=4)时, 这些面按公式重新落位: 原点 (40,16)/(40,32)
        #   bottom = u+w+d = 48 ; left = u+w+d = 48 ; back = u+w+2d = 52
        parts["right_arm"]["uv"]["base"].update(
            top=_r(44, 16, 4, 4),
            bottom=_r(48, 16, 4, 4),
            front=_r(44, 20, 4, 12),
            left=_r(48, 20, 4, 12),
            back=_r(52, 20, 4, 12),
        )
        parts["right_arm"]["uv"]["overlay"].update(
            top=_r(44, 32, 4, 4),
            bottom=_r(48, 32, 4, 4),
            front=_r(44, 36, 4, 12),
            left=_r(48, 36, 4, 12),
            back=_r(52, 36, 4, 12),
        )
        parts["left_arm"]["uv"]["base"].update(
            top=_r(36, 48, 4, 4),
            bottom=_r(40, 48, 4, 4),
            front=_r(36, 52, 4, 12),
            left=_r(40, 52, 4, 12),
            back=_r(44, 52, 4, 12),
        )
        parts["left_arm"]["uv"]["overlay"].update(
            top=_r(52, 48, 4, 4),
            bottom=_r(56, 48, 4, 4),
            front=_r(52, 52, 4, 12),
            left=_r(56, 52, 4, 12),
            back=_r(60, 52, 4, 12),
        )
    return parts


def build_geometry_parts(layer, alpha=None, arm_style="slim"):
    """按部位构建某一层(base 或 overlay)的几何, 返回 {部位名: (verts, texs, norms)}。
    每部位独立的顶点数组, 便于动画时对各部位做关节旋转后再合并渲染。

    面名的含义(按 3D 法线, 不是皮肤模板上的 "right/left" 标签):
        right = +x 面   left = -x 面   front = +z(脸/胸口)  back = -z(后背)
        top = +y 面     bottom = -y 面
    皮肤模板图上的 "right/left" 指的是"外面/里面"(见 Minecraft Wiki 皮肤页),
    与本文件的 right/left 不是一回事: 对右臂来说外面是 -x, 对左臂外面是 +x。
    本函数对两条手臂用同一套贴图块分配(未做 mirror), 因此两侧手臂的
    "+x 块 / -x 块"是按同一规则取的, 不是镜像关系。
    """
    parts = build_parts(arm_style)
    thick = 0.5 if layer == "overlay" else 0.0
    result = {}
    for name, p in parts.items():
        vs, ts, ns = [], [], []
        w, h, d = p["size"]
        w, h, d = w + thick, h + thick, d + thick
        cx, cy, cz = p["pos"]
        uv = p["uv"][layer]

        def add_face(region, nx, ny, nz, flip_u=False):
            rx, ry, rw, rh = region
            if (
                alpha is not None
                and 0 <= rx
                and 0 <= ry
                and rx + rw <= alpha.shape[1]
                and ry + rh <= alpha.shape[0]
                and not alpha[ry : ry + rh, rx : rx + rw].any()
            ):
                return
            u0, v0 = rx / SKIN_SIZE, ry / SKIN_SIZE
            u1, v1 = (rx + rw) / SKIN_SIZE, (ry + rh) / SKIN_SIZE
            if flip_u:  # 该面横向镜像(见 FACE_UV_FLIP): 交换 u0/u1 即可
                u0, u1 = u1, u0
            if nx != 0:  # 左右面: 横向=z
                px_ = nx * w / 2
                if nx > 0:
                    # +x 面从外面看(+z 在左), u0 要落在 z=+d/2
                    quad = [
                        (px_, -h / 2, d / 2, u0, v1),
                        (px_, -h / 2, -d / 2, u1, v1),
                        (px_, h / 2, -d / 2, u1, v0),
                        (px_, h / 2, d / 2, u0, v0),
                    ]
                else:
                    # -x 面从外面看(-z 在左), u0 要落在 z=-d/2
                    quad = [
                        (px_, -h / 2, -d / 2, u0, v1),
                        (px_, -h / 2, d / 2, u1, v1),
                        (px_, h / 2, d / 2, u1, v0),
                        (px_, h / 2, -d / 2, u0, v0),
                    ]
            elif nz != 0:  # 前后面
                pz_ = nz * d / 2
                quad = [
                    (-w / 2, -h / 2, pz_, u0, v1),
                    (w / 2, -h / 2, pz_, u1, v1),
                    (w / 2, h / 2, pz_, u1, v0),
                    (-w / 2, h / 2, pz_, u0, v0),
                ]
            else:  # 顶底面
                py_ = ny * h / 2
                quad = [
                    (-w / 2, py_, -d / 2, u0, v1),
                    (w / 2, py_, -d / 2, u1, v1),
                    (w / 2, py_, d / 2, u1, v0),
                    (-w / 2, py_, d / 2, u0, v0),
                ]
            for idx in (0, 1, 2, 0, 2, 3):
                x, y, z, u, v = quad[idx]
                vs.append((x + cx, y + cy, z + cz))
                ts.append((u, v))
                ns.append((nx, ny, nz))

        for fname, nx, ny, nz in (
            ("right", 1, 0, 0),
            ("left", -1, 0, 0),
            ("front", 0, 0, 1),
            ("back", 0, 0, -1),
            ("top", 0, 1, 0),
            ("bottom", 0, -1, 0),
        ):
            add_face(uv[FACE_UV_KEY[fname]], nx, ny, nz, face_u_flip(fname))
        result[name] = (
            np.array(vs, np.float32).reshape(-1, 3),
            np.array(ts, np.float32).reshape(-1, 2),
            np.array(ns, np.float32).reshape(-1, 3),
        )
    return result


def merge_parts(parts):
    """把按部位的几何合并为单一 (verts, texs, norms) 数组(渲染用)。

    每个部位先 reshape 成 (-1,3)/(-1,2) 再按行 concatenate。不能直接 np.vstack:
    某部位整层全透明(例如只有第一层有像素的皮肤, overlay 六面全跳过)时, 该部位
    的数组是 1 维空数组 array([], float32); vstack 会把 6 个空数组堆成形状 (6,0)
    的"假非空"数组 —— len(verts)=6 骗过了空几何守卫, 渲染时 glNormal3fv(norms[0])
    收到 0 字节数组, PyOpenGL 抛
    "Expected 12 byte array, got 0 byte array" 直接崩掉(exe 里表现为
    "Failed to execute script")。"""
    all_v, all_t, all_n = [], [], []
    for v, t, n in parts.values():
        all_v.append(np.asarray(v, np.float32).reshape(-1, 3))
        all_t.append(np.asarray(t, np.float32).reshape(-1, 2))
        all_n.append(np.asarray(n, np.float32).reshape(-1, 3))
    if not all_v:
        return (
            np.empty((0, 3), np.float32),
            np.empty((0, 2), np.float32),
            np.empty((0, 3), np.float32),
        )
    return (
        np.concatenate(all_v, axis=0),
        np.concatenate(all_t, axis=0),
        np.concatenate(all_n, axis=0),
    )


def rotate_verts(vs, joint, axis, ang_deg):
    """把顶点数组 vs 绕 joint 点、绕单位轴 axis 旋转 ang_deg 度(Rodrigues)。
    用于关节动画(如手臂绕肩、腿绕髋)。vs 为 (N,3) numpy 数组。"""
    ang = math.radians(ang_deg)
    ax, ay, az = axis
    n = math.hypot(ax, ay, az) or 1.0
    ax, ay, az = ax / n, ay / n, az / n
    c = math.cos(ang)
    s = math.sin(ang)
    K = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]], np.float64)
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    j = np.array(joint)
    return (vs - j) @ R.T + j


def build_geometry(layer, alpha=None, arm_style="slim"):
    """构建某一层(base 或 overlay)的三角形几何(合并所有部位)。
    返回 (vertices, texcoords, normals)。overlay 立方体比 base 略大一圈。"""
    return merge_parts(build_geometry_parts(layer, alpha, arm_style))


def read_skin_alpha(path):
    """读取皮肤 alpha 通道并归一化到 64x64 经典布局, 返回 np 布尔数组。
    高分辨率皮肤(128x128 等)按 64 布局放大, 用 NEAREST 缩回保留每格像素。
    用 with 打开: 坏图在 convert/resize 抛异常时也能立刻关掉文件句柄,
    否则 Windows 下这张皮肤会因为"正在使用"而改不了名(无法隔离成 ERROR_<原名>)。"""
    with Image.open(path) as im:
        img = im.convert("RGBA")
        a = np.array(img)[:, :, 3]
        if a.shape[0] != 64 or a.shape[1] != 64:
            with img.resize((64, 64), Image.NEAREST) as small:
                a = np.array(small)[:, :, 3]
    img.close()
    return a > 0


ERROR_PREFIX = "ERROR_"  # 出错皮肤的重命名前缀; 扫描时跳过这类文件


def error_path(path):
    """给出问题皮肤的隔离路径: 原文件名前加 ERROR_ 前缀(同目录、扩展名不变)。
    例: Dream.png -> ERROR_Dream.png

    不用 ERROR_1/ERROR_2 编号: 编号要靠"扫目录数已用最大值"或状态文件里的计数
    来递增, 既依赖外部状态、又可能被篡改/清理, 一旦计数错了就会撞名 —— 撞名后
    改名失败, 坏皮肤留在原位, 下次依然会被点开再崩。原文件名本来就各不相同,
    加前缀就天然唯一, 不需要任何外部计数。

    若 ERROR_ 前缀的文件已存在(同名坏图被重新下载过一次), 继续往前加前缀
    (ERROR_ERROR_xxx.png), 直到拿到一个当前不存在的名字。"""
    folder = os.path.dirname(os.path.abspath(path))
    name = os.path.basename(path)
    candidate = os.path.join(folder, ERROR_PREFIX + name)
    # 极端情况下的兜底: 一直加前缀直到不重名(每轮都检查磁盘, 不依赖任何计数)
    while os.path.exists(candidate):
        candidate = os.path.join(folder, ERROR_PREFIX + os.path.basename(candidate))
    return candidate


def quarantine_skin(path):
    """把"打开就崩"的皮肤文件改名隔离: 文件名前加 ERROR_(同目录、保留扩展名)。
    改名后 _scan_skins 会跳过它, 不会被再点开; 改不动(被占用/无权限)时返回 None。
    只改文件名, 不动图片内容 —— 出问题的原图完整保留, 只是被隔离。

    重命名会因"文件正被占用"(WinError 32)失败: 坏图抛出异常后, 异常的 traceback
    仍引用着读取函数的栈帧, 里面的 PIL 图像/文件句柄没有释放。所以每次失败都先
    gc.collect() 回收一次再重试(同时重新取名字, 避免和刚出现的文件撞名)。"""
    if not path or not os.path.exists(path):
        return None
    dst = error_path(path)
    last_err = None
    for _ in range(6):
        try:
            os.rename(path, dst)
            print(
                f"皮肤出错, 已隔离改名: {os.path.basename(path)} -> {os.path.basename(dst)}"
            )
            return dst
        except OSError as e:
            last_err = e
            gc.collect()
            time.sleep(0.05)
            dst = error_path(path)
    print(f"隔离皮肤失败(文件被占用?): {path} -> {last_err}")
    return None


def detect_arm_style(arr):
    """按原版判定规则识别手臂类型: 采样 x=54..55, y=20..31 这一块(=24 个像素),
    不透明数 < 6(25%) 判为纤细(Alex, 3px), 否则宽形(Steve, 4px)。

    为什么是 x=54..55: 宽形手臂的背面块占 52..55(4px), 纤细只有 51..53(3px),
    所以 54/55 这两列在纤细皮肤里是"死区"(永远是透明的)。mcskins.org 的
    编辑器也正是取 (54,20)(54,22)(54,24)(54,26)(55,20)(55,24) 这几点, 任一不透明
    即判宽形。取 x=46..47 是错的 —— 修正后的布局里 x=46 落在纤细的正面块(44..46)
    内, 于是永远判成宽形。
    128×128 皮肤按比例换算该区域。判不出来时按整图兜底。"""
    sh, sw = arr.shape[0], arr.shape[1]
    sx, sy = sw / 64.0, sh / 64.0
    x0, x1 = int(54 * sx), max(int(56 * sx), int(54 * sx) + 1)
    y0, y1 = int(20 * sy), max(int(32 * sy), int(20 * sy) + 1)
    reg = arr[y0:y1, x0:x1, 3]
    total = reg.size
    if total <= 0:
        return "slim"
    opaque = int((reg > 0).sum())
    # 原版阈值: 24 个采样像素里少于 6 个不透明(=25%)判为纤细
    return "slim" if opaque < 0.25 * total else "classic"


def load_skin_arrays(path):
    """读取皮肤, 返回 (显示用 RGBA 数组, 原始 RGBA 数组)。
    显示用数组: alpha>0 一律二值化为 255(半透明像素也显示, 避免 alpha test
    丢弃造成镂空); base 层原本透明的像素保持透明(与原版/HMCL 一致, 会留洞),
    不再做任何补洞填充。
    原始数组: 保持文件原样, 编辑保存时只把被改过的像素写回它。"""
    # 用 with 打开: 截断/损坏的图在 convert 抛异常时也能立刻关掉文件句柄。
    # 否则句柄会被异常 traceback 钉住, 这张皮肤在 Windows 下会因"正在使用"
    # 而改不了名, 隔离成 ERROR_<原名> 就失败了(实测 WinError 32)。
    with Image.open(path) as im:
        img = im.convert("RGBA")
    raw = np.array(img)
    img.close()
    arr = raw.copy()
    if (
        arr.shape[2] == 4
        and int((arr[:, :, 3] > 0).sum())
        and int((arr[:, :, 3] < 255).sum())
    ):
        arr[arr[:, :, 3] > 0, 3] = 255
    return arr, raw


def upload_texture(arr):
    """把 RGBA 数组上传为 OpenGL 纹理(NEAREST 像素风), 返回纹理 id。"""
    th, tw = arr.shape[0], arr.shape[1]
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    # 不要带 flip=-1(垂直翻转会导致 UV 错位); 按原样上传。
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        tw,
        th,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        np.ascontiguousarray(arr).tobytes(),
    )
    return texture


def load_skin(path):
    """读取皮肤并上传为 OpenGL 纹理, 返回纹理 id。"""
    arr, _raw = load_skin_arrays(path)
    return upload_texture(arr)


class Viewer:
    UNDO_MAX = 10  # 撤销栈深度

    def __init__(self, skin_path):
        pygame.init()
        self.W, self.H = 900, 700
        self.SW = 180  # 侧边栏固定宽度
        self.x0 = self.W - self.SW  # 侧边栏左边界(3D 预览区右侧)
        self._flags = pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE
        self.screen = pygame.display.set_mode((self.W, self.H), self._flags)
        self._font = None
        # 支持无初始皮肤或传入文件夹: 找不到文件也照常启动(可点"导入文件夹"),
        # 传入文件夹时自动扫描并加载其中第一张皮肤。
        self.folder = None
        self.skin_list = []
        self._failed_skins = set()  # 已判定"打开就崩"的皮肤路径(绝不写进状态文件)
        self.skin_index = 0
        if skin_path and os.path.exists(skin_path):
            if os.path.isdir(skin_path):
                self.folder = skin_path
                self.skin_list = self._scan_skins(skin_path)
                self.current_skin = self.skin_list[0] if self.skin_list else None
                self.skin_index = 0
            else:
                self.current_skin = skin_path
                self.folder = os.path.dirname(os.path.abspath(skin_path))
                self.skin_list = self._scan_skins(self.folder)
                try:
                    self.skin_index = self.skin_list.index(skin_path)
                except ValueError:
                    self.skin_index = 0
        else:
            self.current_skin = None
        # 右侧边栏按钮(坐标相对侧边栏左边界 x0 偏移, 使用时加 x0)
        self.buttons = [
            ("导入文件夹", (10, 58, 160, 38)),
            ("上一张", (10, 104, 76, 38)),
            ("下一张", (86, 104, 76, 38)),
            ("刷新", (10, 150, 76, 38)),
            ("重命名", (86, 150, 76, 38)),
            ("走路", (10, 196, 76, 38)),
            ("挥手", (86, 196, 76, 38)),
            # 手臂渲染模式: 两种模式手动切换, 默认纤细
            ("宽形", (10, 242, 76, 38)),
            ("纤细", (86, 242, 76, 38)),
            ("编辑皮肤", (10, 286, 160, 38)),
        ]
        # 3D 像素编辑状态(点「编辑皮肤」后进入, 弹出调色板窗口)
        self.edit_mode = False
        self.palette = None
        self.paint_color = (220, 70, 70)  # 当前画笔颜色
        self.brush = 1  # 笔刷大小(1/2/3 px)
        self.eyedropper = False  # 取色模式: 左键改为吸取模型上的颜色
        self.erase_mode = False  # 删除像素(橡皮): 左键把像素 alpha 置 0
        self.rotate_mode = False  # 旋转/平移模式: 左键拖动转视角(不绘制)
        self.hover_texel = None  # 鼠标悬停命中的纹理像素(高亮用)
        self.hover2d = None  # 调色板二维图里悬停的像素 (x, y)
        self.sel2d = None  # 调色板二维图里点选的像素 (x, y), 3D 同步高亮
        self.texture_version = 0  # 纹理内容版本号(二维图据此刷新)
        self.undo_stack = []  # 撤销栈: 每一笔 = {(x,y): (改动前显示值, 改动前原始值)}
        self._stroke = None  # 当前正在画的一笔(按下到抬起之间)
        self.skin_arr_orig = None  # 进入编辑时的原始数组快照(判断改动集合)
        self._painting = False
        self.edited = set()  # 已改动的像素坐标(保存时写回原图)
        # 搜索框: 输入关键字实时过滤当前文件夹皮肤
        self.search_active = False
        self.search_text = ""
        self.search_rect = (10, 22, 160, 30)
        self.search_results = []  # 过滤后的匹配文件列表
        self.search_scroll = 0  # 列表滚动偏移
        self._click_item = None  # 记录单击的列表项(用于双击检测)
        self._click_time = 0.0
        pygame.display.set_caption(
            os.path.basename(self.current_skin)
            if self.current_skin
            else "Minecraft 皮肤预览 - 点击右侧「导入文件夹」选择皮肤"
        )
        self._setup_gl()

        # 手臂渲染模式: "slim"=纤细(默认) / "classic"=宽形; 侧边栏手动切换并记忆
        self._arm_style = self._restore_arm_style()
        if self.current_skin:
            try:
                self.skin_arr_display, self.skin_arr_raw = load_skin_arrays(
                    self.current_skin
                )
                self.texture = upload_texture(self.skin_arr_display)
                self._skin_alpha = read_skin_alpha(self.current_skin)  # 面级过滤用
                if not getattr(self, "_arm_style_manual", False):
                    # 没有手动指定过宽形/纤细时, 跟随皮肤文件自动判定
                    self._arm_style = detect_arm_style(self.skin_arr_display)
            except Exception as e:
                # 一启动就崩的皮肤: 立刻隔离改名 + 把状态文件里的皮肤清掉,
                # 否则下次启动还会再打开它, 整个程序都进不去(用户报告的问题)。
                msg = f"{type(e).__name__}: {e}"
                del e
                gc.collect()
                self._handle_skin_error(msg, "启动加载")
        else:
            self.texture = None
            self.skin_arr_display = None
            self.skin_arr_raw = None
            self._skin_alpha = None
        # 双层几何: base(第一层) + overlay(第二层, 外扩), 透明面按皮肤 alpha 跳过
        # 按部位保存, 动画时对各部位做关节旋转后再合并渲染
        self._rebuild_geometry()
        # 动画状态(走路/挥手)
        self.anim = None  # None/"walk"/"wave"
        self.anim_start = 0.0
        self.anim_dur = {"walk": 3.0, "wave": 2.2}

        # 相机状态
        self.yaw = -40.0
        self.pitch = -10.0
        self.dist = 68.0
        self.pan = [0.0, 0.0]
        self.wireframe = False
        self.lit = True

        self._dragging = None
        self._last = None
        self.clock = pygame.time.Clock()
        if self.current_skin:
            self._save_state()  # 记录初始加载的文件夹+皮肤, 下次启动恢复
            self._update_icon()  # 窗口图标 = 当前皮肤头部
        self._update_search()  # 初始化搜索结果列表

    @staticmethod
    def _restore_arm_style():
        """读取上次保存的手臂渲染模式, 默认纤细(slim)。"""
        try:
            import json

            with open(Viewer._state_path(), encoding="utf-8") as f:
                st = json.load(f)
            v = st.get("arm_style")
            if v in ARM_STYLES:
                return v
        except Exception:
            pass
        return "slim"

    def _rebuild_geometry(self):
        """按当前手臂模式重建 base/overlay 几何(切换宽形/纤细时调用)。"""
        empty = np.zeros((64, 64), bool)
        al = self._skin_alpha if self._skin_alpha is not None else empty
        self._parts_base = build_geometry_parts("base", al, self._arm_style)
        self._parts_overlay = build_geometry_parts("overlay", al, self._arm_style)
        self.base_geo = merge_parts(self._parts_base)
        self.overlay_geo = merge_parts(self._parts_overlay)

    def _set_arm_style(self, style):
        """切换手臂渲染模式: classic=宽形(4px), slim=纤细(3px)。"""
        if style not in ARM_STYLES or style == self._arm_style:
            return
        self._arm_style = style
        self._arm_style_manual = True  # 用户手动指定后, 换皮肤不再自动改写
        self._rebuild_geometry()
        self._save_state()  # 记住选择, 下次启动仍用该模式

    # ---- 3D 像素编辑 -----------------------------------------------------
    def _enter_edit(self):
        """进入 3D 像素编辑: 强制截断动画、回到静态姿态, 弹出调色板窗口。"""
        if self.edit_mode or not self.current_skin:
            return
        # 强制截断所有动画, 模型定格在当前姿态, 便于精确点选像素
        self.anim = None
        self.base_geo = self._merge_geo(self._parts_base)
        self.overlay_geo = self._merge_geo(self._parts_overlay)
        if self.skin_arr_display is None:
            self.skin_arr_display, self.skin_arr_raw = load_skin_arrays(
                self.current_skin
            )
        self.edit_mode = True
        self.edited = set()
        self.undo_stack = []
        self._stroke = None
        self.eyedropper = False
        self.erase_mode = False
        self.rotate_mode = False
        self.skin_arr_orig = (
            self.skin_arr_raw.copy() if self.skin_arr_raw is not None else None
        )
        self.hover_texel = None
        self.hover2d = None
        self.sel2d = None
        self._painting = False
        self._close_palette()
        try:
            self.palette = PaletteWindow(self)
        except Exception as e:
            print("调色板窗口打开失败:", e)
            self.palette = None
        print("进入编辑模式:", self.current_skin)

    def _close_palette(self):
        if self.palette is not None:
            try:
                self.palette.close()
            except Exception:
                pass
            self.palette = None

    def _exit_edit(self):
        """退出编辑且不保存(丢弃本次改动)。"""
        if not self.edit_mode:
            return
        self._close_palette()
        self.edit_mode = False
        self._painting = False
        self.hover_texel = None
        self.hover2d = None
        self.sel2d = None
        self.edited.clear()
        self.undo_stack = []
        self._stroke = None
        self.erase_mode = False
        self.rotate_mode = False
        self.skin_arr_orig = None
        print("退出编辑(未保存)")

    def _save_edit(self):
        """把改动写回皮肤文件(只覆盖被编辑过的像素), 关闭编辑器并自动刷新。"""
        if not self.edit_mode:
            return
        path = self.current_skin
        try:
            # 原始数组在编辑时已逐像素同步(显示数组里被二值化的像素不会写进文件),
            # 因此直接保存原始数组 = 文件原样 + 用户改动。
            arr = (
                self.skin_arr_raw.copy()
                if self.skin_arr_raw is not None
                else self.skin_arr_display.copy()
            )
            ext = os.path.splitext(path)[1].lower()
            fmt = "JPEG" if ext in (".jpg", ".jpeg") else "PNG"
            tmp = path + ".tmp"
            Image.fromarray(arr).save(tmp, format=fmt)
            os.replace(tmp, path)  # 原子替换, 避免保存中断损坏原文件
            print("已保存编辑:", path, "改动像素:", len(self.edited))
        except Exception as e:
            print("保存失败:", e)
            return
        self._close_palette()
        self.edit_mode = False
        self._painting = False
        self.hover_texel = None
        self.hover2d = None
        self.sel2d = None
        self.anim = None
        # 保存并退出后自动刷新: 重新加载编辑后的皮肤(纹理+几何)
        self._reload_skin(path)

    def _clear_gl_errors(self):
        """清掉残留的 GL 错误标志, 避免后续无关调用被误报(如 glDeleteTextures)。"""
        try:
            for _ in range(16):
                if glGetError() == GL_NO_ERROR:
                    break
        except Exception:
            pass

    def _set_color(self, color):
        """设置画笔颜色并同步调色板窗口。"""
        self.paint_color = (
            int(color[0]) & 255,
            int(color[1]) & 255,
            int(color[2]) & 255,
        )
        if self.palette is not None:
            self.palette.set_color(self.paint_color)

    def _upload_edit_texture(self):
        """把当前显示数组重新上传到现有纹理(即时看到笔迹)。"""
        if self.texture is None or self.skin_arr_display is None:
            return
        arr = self.skin_arr_display
        th, tw = arr.shape[0], arr.shape[1]
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            tw,
            th,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            np.ascontiguousarray(arr).tobytes(),
        )

    def _pick_texel(self, mx, my):
        """把窗口坐标反投影为射线, 与模型各部件的面求交, 返回命中的纹理像素。
        返回 dict(t, x, y, part, layer, normal, point); 未命中返回 None。"""
        if not self.current_skin or self.skin_arr_display is None:
            return None
        if not (0 <= mx < self.x0 and 0 <= my < self.H):
            return None
        arr = self.skin_arr_display
        sh, sw = arr.shape[0], arr.shape[1]
        sx, sy = sw / 64.0, sh / 64.0
        # 屏幕 → NDC → 相机空间射线
        nx = 2.0 * mx / float(self.x0) - 1.0
        ny = 1.0 - 2.0 * my / float(self.H)
        tan_half = math.tan(math.radians(20.0))  # gluPerspective 40° → 半角 20°
        aspect = self.x0 / float(self.H)
        d_cam = np.array([nx * tan_half * aspect, ny * tan_half, -1.0], float)
        d_cam /= np.linalg.norm(d_cam)
        # 相机: 世界→相机 = R*p + t, R = Rx(pitch) * Ry(yaw)
        cp, sp = math.cos(math.radians(self.pitch)), math.sin(math.radians(self.pitch))
        cyw, syw = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], float)
        Ry = np.array([[cyw, 0, syw], [0, 1, 0], [-syw, 0, cyw]], float)
        R = Rx @ Ry
        t_vec = np.array([self.pan[0], self.pan[1] - 16.0, -self.dist], float)
        d = R.T @ d_cam  # 世界空间射线方向
        o = -R.T @ t_vec  # 世界空间相机位置
        face_keys = {
            (1, 0, 0): "right",
            (-1, 0, 0): "left",
            (0, 0, 1): "front",
            (0, 0, -1): "back",
            (0, 1, 0): "top",
            (0, -1, 0): "bottom",
        }
        parts = build_parts(self._arm_style)
        best = None
        for name, p in parts.items():
            for layer in ("base", "overlay"):
                thick = 0.5 if layer == "overlay" else 0.0
                w, h, dd = (
                    p["size"][0] + thick,
                    p["size"][1] + thick,
                    p["size"][2] + thick,
                )
                cx, cy_, cz = p["pos"]
                for normal, key in face_keys.items():
                    rx, ry_, rw, rh = p["uv"][layer][FACE_UV_KEY[key]]
                    X0 = max(0, int(rx * sx))
                    Y0 = max(0, int(ry_ * sy))
                    W = min(sw - X0, max(1, int(rw * sx)))
                    H = min(sh - Y0, max(1, int(rh * sy)))
                    if W <= 0 or H <= 0:
                        continue
                    # 该面上有没有不透明像素(全透明则这个面根本不存在)
                    if not (arr[Y0 : Y0 + H, X0 : X0 + W, 3] > 0).any():
                        continue
                    if normal[0] != 0:
                        plane = cx + normal[0] * w / 2.0
                        axis = 0
                    elif normal[1] != 0:
                        plane = cy_ + normal[1] * h / 2.0
                        axis = 1
                    else:
                        plane = cz + normal[2] * dd / 2.0
                        axis = 2
                    den = d[axis]
                    if abs(den) < 1e-6:
                        continue
                    # 只允许命中"朝向相机"的面: 法线与视线方向夹角 > 90°。
                    # 否则鼠标在轮廓边缘会选中背面/侧面那些屏幕上看不见的面,
                    # 结果就是把像素画到了另一侧的"对角"面上。
                    if normal[axis] * den > 0.0:
                        continue
                    tt = (plane - o[axis]) / den
                    if tt <= 0.05 or (best is not None and tt > best["t"]):
                        continue
                    pt = o + tt * d
                    if normal[0] != 0:
                        if normal[0] > 0:  # +x 面: u=0 在 z=+d/2
                            u = ((cz + dd / 2.0) - pt[2]) / dd
                        else:
                            u = (pt[2] - (cz - dd / 2.0)) / dd
                        v = ((cy_ + h / 2.0) - pt[1]) / h
                    elif normal[1] != 0:
                        u = (pt[0] - (cx - w / 2.0)) / w
                        v = ((cz + dd / 2.0) - pt[2]) / dd
                    else:
                        u = (pt[0] - (cx - w / 2.0)) / w
                        v = ((cy_ + h / 2.0) - pt[1]) / h
                    if face_u_flip(key):  # 该面横向镜像(背面): 屏幕 u 与贴图 u 相反
                        u = 1.0 - u
                    if not (-0.001 <= u <= 1.001 and -0.001 <= v <= 1.001):
                        continue
                    u = min(max(u, 0.0), 0.9999)
                    v = min(max(v, 0.0), 0.9999)
                    tx = min(W - 1, int(u * W))
                    ty = min(H - 1, int(v * H))
                    X, Y = X0 + tx, Y0 + ty
                    if arr[Y, X, 3] == 0:  # 落点像素透明 → 不可编辑
                        continue
                    best = dict(
                        t=tt, x=X, y=Y, part=name, layer=layer, normal=normal, point=pt
                    )
        return best

    def _face_cell_quad(self, hit):
        """返回命中像素在该面上的世界坐标四边形(略微沿法线外移, 避免闪烁)。"""
        p = build_parts(self._arm_style)[hit["part"]]
        thick = 0.5 if hit["layer"] == "overlay" else 0.0
        w, h, dd = (
            p["size"][0] + thick,
            p["size"][1] + thick,
            p["size"][2] + thick,
        )
        cx, cy_, cz = p["pos"]
        face_keys = {
            (1, 0, 0): "right",
            (-1, 0, 0): "left",
            (0, 0, 1): "front",
            (0, 0, -1): "back",
            (0, 1, 0): "top",
            (0, -1, 0): "bottom",
        }
        rx, ry_, rw, rh = p["uv"][hit["layer"]][
            FACE_UV_KEY[face_keys[tuple(hit["normal"])]]
        ]
        sh, sw = self.skin_arr_display.shape[0], self.skin_arr_display.shape[1]
        # 像素在面 region 内的连续比例
        u0 = (hit["x"] * 64.0 / sw - rx) / rw
        u1 = ((hit["x"] + 1) * 64.0 / sw - rx) / rw
        v0 = (hit["y"] * 64.0 / sh - ry_) / rh
        v1 = ((hit["y"] + 1) * 64.0 / sh - ry_) / rh
        u0, u1 = max(u0, 0.0), min(u1, 1.0)
        v0, v1 = max(v0, 0.0), min(v1, 1.0)
        # 该面横向镜像时, 贴图里的 u 与面上的 u 互为 1-u(和 _pick_texel 对称)
        if face_u_flip(face_keys[tuple(hit["normal"])]):
            u0, u1 = 1.0 - u1, 1.0 - u0
        n = hit["normal"]
        off = 0.04
        if n[0] != 0:  # 左右面: 横向=z, 纵向=y
            px_ = cx + n[0] * (w / 2.0 + off)
            if n[0] > 0:  # +x 面: u 从 z=+d/2 往 -z 增长
                z0 = cz + dd / 2.0 - u0 * dd
                z1 = cz + dd / 2.0 - u1 * dd
            else:  # -x 面: u 从 z=-d/2 往 +z 增长
                z0 = cz - dd / 2.0 + u0 * dd
                z1 = cz - dd / 2.0 + u1 * dd
            y_hi = cy_ + h / 2.0 - v0 * h
            y_lo = cy_ + h / 2.0 - v1 * h
            return [(px_, y_hi, z0), (px_, y_hi, z1), (px_, y_lo, z1), (px_, y_lo, z0)]
        if n[2] != 0:  # 前后面: 横向=x, 纵向=y
            pz_ = cz + n[2] * (dd / 2.0 + off)
            x0 = cx - w / 2.0 + u0 * w
            x1 = cx - w / 2.0 + u1 * w
            y_hi = cy_ + h / 2.0 - v0 * h
            y_lo = cy_ + h / 2.0 - v1 * h
            return [(x0, y_hi, pz_), (x1, y_hi, pz_), (x1, y_lo, pz_), (x0, y_lo, pz_)]
        # 顶底面: 横向=x, 纵向=z(v=0 在 +z 端)
        py_ = cy_ + n[1] * (h / 2.0 + off)
        x0 = cx - w / 2.0 + u0 * w
        x1 = cx - w / 2.0 + u1 * w
        z0 = cz + dd / 2.0 - v0 * dd
        z1 = cz + dd / 2.0 - v1 * dd
        return [(x0, py_, z0), (x1, py_, z0), (x1, py_, z1), (x0, py_, z1)]

    def _texel_cells(self, x, y, include_empty=False):
        """给定纹理像素 (x,y), 找出用到它的所有模型面(用于二维图 → 3D 高亮)。
        默认只返回真实存在的面(该面上至少有一个不透明像素);
        include_empty=True 时连全透明面也返回, 便于高亮二维图上的空白像素。"""
        if self.skin_arr_display is None:
            return []
        arr = self.skin_arr_display
        sh, sw = arr.shape[0], arr.shape[1]
        sx, sy = sw / 64.0, sh / 64.0
        face_keys = {
            (1, 0, 0): "right",
            (-1, 0, 0): "left",
            (0, 0, 1): "front",
            (0, 0, -1): "back",
            (0, 1, 0): "top",
            (0, -1, 0): "bottom",
        }
        out = []
        for name, p in build_parts(self._arm_style).items():
            for layer in ("base", "overlay"):
                for normal, key in face_keys.items():
                    rx, ry_, rw, rh = p["uv"][layer][FACE_UV_KEY[key]]
                    X0, Y0 = int(rx * sx), int(ry_ * sy)
                    W, H = int(rw * sx), int(rh * sy)
                    if W <= 0 or H <= 0:
                        continue
                    if not (X0 <= x < X0 + W and Y0 <= y < Y0 + H):
                        continue
                    if not include_empty:
                        if not (arr[Y0 : Y0 + H, X0 : X0 + W, 3] > 0).any():
                            continue
                    out.append(
                        dict(
                            x=x,
                            y=y,
                            part=name,
                            layer=layer,
                            normal=normal,
                            t=0.0,
                            point=None,
                        )
                    )
        return out

    def _draw_cell(
        self,
        hit,
        fill=(1.0, 0.86, 0.15, 0.45),
        line=(1.0, 0.95, 0.4, 1.0),
        through=False,
    ):
        """高亮一个纹理像素: 半透明填充 + 亮色描边。
        through=True 时忽略遮挡(二维图选中的像素即使被挡住也显示)。"""
        if hit is None:
            return
        quad = self._face_cell_quad(hit)
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        if through:
            glDisable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glColor4f(*fill)
        glBegin(GL_QUADS)
        for v in quad:
            glVertex3f(*v)
        glEnd()
        glColor4f(*line)
        glBegin(GL_LINE_LOOP)
        for v in quad:
            glVertex3f(*v)
        glEnd()
        glDepthFunc(GL_LESS)
        if through:
            glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        # 复位当前色: 否则残留的高亮色会经 GL_COLOR_MATERIAL 把下一帧模型整体染色
        glColor4f(1.0, 1.0, 1.0, 1.0)

    def _draw_hover(self, hit):
        """高亮鼠标悬停的可编辑像素(半透明黄块 + 亮黄描边)。"""
        self._draw_cell(hit)

    def _paint_at(self, mx, my):
        """在鼠标位置对应的纹理像素上绘制(或取色)。"""
        if not self.edit_mode:
            return
        hit = self._pick_texel(mx, my)
        if hit is None:
            return
        self._paint_texel(hit["x"], hit["y"])

    def _paint_texel(self, x, y):
        """对指定纹理像素绘制(取色模式则吸取颜色; 橡皮模式则删除)。"""
        arr = self.skin_arr_display
        if arr is None:
            return
        if self.eyedropper:
            px = arr[y, x]
            if int(px[3]) > 0:  # 空白像素没有颜色可取
                self._set_color((int(px[0]), int(px[1]), int(px[2])))
            self.eyedropper = False
            if self.palette is not None:
                self.palette.set_eyedropper(False)
            return
        self._fill_pixel(x, y)

    def _record_pixel(self, x, y):
        """记录像素改动前的值(用于撤销); 同一笔内同一像素只记一次。"""
        if self._stroke is None:
            return
        key = (x, y)
        if key in self._stroke:
            return
        disp = tuple(int(c) for c in self.skin_arr_display[y, x])
        raw = (
            tuple(int(c) for c in self.skin_arr_raw[y, x])
            if self.skin_arr_raw is not None
            else disp
        )
        self._stroke[key] = (disp, raw)

    def begin_stroke(self):
        """开始一笔(鼠标按下 / 二维图点击), 期间所有改动可整体撤销。"""
        if self._stroke is None:
            self._stroke = {}

    def end_stroke(self):
        """结束一笔: 有实际改动才入撤销栈(栈深 10)。"""
        if not self._stroke:
            self._stroke = None
            return
        self.undo_stack.append(self._stroke)
        while len(self.undo_stack) > self.UNDO_MAX:
            self.undo_stack.pop(0)
        self._stroke = None

    def undo(self):
        """撤销上一笔: 恢复该笔涉及像素的显示值与原始值。"""
        if not self.undo_stack:
            return 0
        entry = self.undo_stack.pop()
        for (x, y), (disp, raw) in entry.items():
            if self.skin_arr_display is not None:
                self.skin_arr_display[y, x] = disp
            if self.skin_arr_raw is not None:
                self.skin_arr_raw[y, x] = raw
            # 改动集合按“与进入编辑时的原始图像比”重新判定
            if self.skin_arr_orig is not None:
                if (self.skin_arr_raw[y, x] != self.skin_arr_orig[y, x]).any():
                    self.edited.add((x, y))
                else:
                    self.edited.discard((x, y))
        self.texture_version += 1
        self._upload_edit_texture()
        return len(entry)

    def _fill_pixel(self, x, y):
        """以当前颜色填充 x,y 处，按笔刷大小扩展为方阵; 橡皮模式则删除像素。
        同时写入显示数组与原始数组(二维图展示的就是原始数组, 也即文件内容+改动)。"""
        arr = self.skin_arr_display
        if arr is None:
            return
        h, w = arr.shape[0], arr.shape[1]
        n = max(1, int(self.brush))
        half = (n - 1) // 2
        r, g, b = self.paint_color
        erase = bool(self.erase_mode)
        if self._stroke is None:
            self.begin_stroke()
        changed = False
        for yy in range(y - half, y - half + n):
            for xx in range(x - half, x - half + n):
                if 0 <= yy < h and 0 <= xx < w:
                    self._record_pixel(xx, yy)
                    if erase:  # 删除: alpha=0 即视为无像素
                        arr[yy, xx, :3] = 0
                        arr[yy, xx, 3] = 0
                        if self.skin_arr_raw is not None:
                            self.skin_arr_raw[yy, xx, :3] = 0
                            self.skin_arr_raw[yy, xx, 3] = 0
                    else:
                        arr[yy, xx, :3] = (r, g, b)
                        arr[yy, xx, 3] = 255
                        if self.skin_arr_raw is not None:
                            self.skin_arr_raw[yy, xx, :3] = (r, g, b)
                            self.skin_arr_raw[yy, xx, 3] = 255
                    self.edited.add((xx, yy))
                    changed = True
        if changed:
            self.texture_version += 1
            self._upload_edit_texture()

    def _scan_skins(self, folder):
        """扫描文件夹内的皮肤图片(png/jpg/jpeg)。
        ERROR_ 开头的文件是"打开就崩"被隔离的皮肤, 一律跳过, 免得再点崩一次。"""
        if not folder or not os.path.isdir(folder):
            return []
        return [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
            and not f.upper().startswith(ERROR_PREFIX)
        ]

    def _update_search(self):
        """按搜索关键字过滤当前文件夹的皮肤列表(大小写不敏感)。"""
        kw = self.search_text.strip().lower()
        if kw:
            self.search_results = [
                f for f in self.skin_list if kw in os.path.basename(f).lower()
            ]
        else:
            self.search_results = list(self.skin_list)
        if self.search_scroll >= len(self.search_results):
            self.search_scroll = max(0, len(self.search_results) - 1)

    def _handle_skin_error(self, exc, where="", bad_path=None):
        """渲染/加载中捕获到异常时调用: 把出问题的皮肤隔离改名为 ERROR_<原名>。

        用户要求的核心防护: 出 bug 的皮肤绝不能写进 skin_preview_state.json,
        否则下次启动会恢复到它、一开就崩, 整个程序都打不开。所以:
          - 崩的是"正在显示的"皮肤 -> 清空它并立刻重写状态(皮肤名写 None),
            下次启动改为从文件夹里的正常皮肤开始(扫描时会跳过 ERROR_*);
          - 只是"切换到某张坏皮肤"失败 -> 当前好皮肤照旧显示, 状态里仍记着它,
            坏的那张只从列表里摘掉, 不会成为当前皮肤。

        bad_path 用于"加载新皮肤时抛异常"的场景: 此时 self.current_skin 还是
        上一张好皮肤, 必须由调用方把真正出错的文件路径传进来, 否则会隔离错对象。"""
        bad = bad_path or getattr(self, "current_skin", None)
        info = f"{exc!r}" if isinstance(exc, BaseException) else str(exc)
        print(f"[{where}] 皮肤异常: {bad} -> {info}")
        exc = None  # 断开对异常对象的引用: traceback 里可能牵着未关闭的文件句柄
        gc.collect()
        if not bad:
            return
        try:
            self._failed_skins.add(bad)  # 记进黑名单, _save_state 永不写它
        except Exception:
            pass
        new_path = quarantine_skin(bad)
        # 无论改名成功与否都把它从列表里摘掉, 保证不会被再次加载
        self.skin_list = [p for p in getattr(self, "skin_list", []) if p != bad]
        self.search_results = [
            p for p in getattr(self, "search_results", []) if p != bad
        ]
        crashing_current = bad == getattr(self, "current_skin", None)
        if crashing_current:
            # 崩掉的就是正在显示的皮肤: 模型/纹理全清, 回落到"无皮肤"提示,
            # 下一帧 _render 会自动加载列表里的下一张
            self.current_skin = None
            self.skin_arr_display = None
            self.skin_arr_raw = None
            self._skin_alpha = None
            self.texture = None
            self._parts_base = {}
            self._parts_overlay = {}
            self.base_geo = merge_parts({})
            self.overlay_geo = merge_parts({})
            self.hover_texel = None
            self.hover2d = None
            self.sel2d = None
            self.anim = None
            self.skin_index = max(0, min(self.skin_index, len(self.skin_list) - 1))
        self._update_search()
        self._save_state()  # 坏皮肤已摘掉/已被写成 None, 下次启动不会恢复它
        if crashing_current:
            try:
                kept = os.path.basename(new_path) if new_path else os.path.basename(bad)
                pygame.display.set_caption(
                    f"Minecraft 皮肤预览 - 上皮肤出错(已隔离为 {kept})"
                )
            except Exception:
                pass

    def _reload_skin(self, path):
        """重新加载指定皮肤图片(删除旧纹理, 创建新纹理)。
        加载失败时保留旧纹理并打印原因, 避免界面空白且无提示。"""
        if not path or not os.path.exists(path):
            print("皮肤不存在:", path)
            return
        try:
            self._clear_gl_errors()
            if getattr(self, "texture", None) is not None:
                try:
                    glDeleteTextures([self.texture])
                except Exception:
                    pass  # 旧纹理删除失败不影响重新加载
            self.skin_arr_display, self.skin_arr_raw = load_skin_arrays(path)
            self.texture = upload_texture(self.skin_arr_display)
            self._skin_alpha = read_skin_alpha(path)  # 面级过滤依赖具体皮肤
            if not getattr(self, "_arm_style_manual", False):
                style = detect_arm_style(self.skin_arr_display)
                if style != self._arm_style:
                    self._arm_style = style
                    self._rebuild_geometry()
                    self._save_state()
            self.texture_version += 1
            # 换皮肤时保持当前手臂模式(宽形/纤细由用户手动指定), 只重建几何
            self._rebuild_geometry()
            self.anim = None  # 换皮肤时停止动画
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"加载皮肤失败: {path} -> {msg}")
            del e  # 先解除异常引用(它牵着坏图的文件句柄, 不放手就改不了名)
            gc.collect()
            # 这张新皮肤是坏的: 隔离改名 ERROR_<原名>, 不写进状态; 崩的不是它自己时
            # (current_skin 仍是上一张好皮肤) 由 bad_path 指明真正的坏文件
            self._handle_skin_error(msg, "加载", bad_path=path)
            return
        self.current_skin = path
        pygame.display.set_caption(f"{os.path.basename(path)}")
        self._save_state()  # 记录当前文件夹+文件名+序号, 下次启动恢复
        self._update_icon()  # 固定用原始皮肤 pic1 的头作图标

    def _choose_folder(self):
        """选择皮肤文件夹。优先用 tkinter 原生对话框(系统 Python 可用),
        兜底用 PowerShell 对话框。返回选中的文件夹路径, 失败返回 None。"""
        # 1) tkinter 原生文件夹选择(系统 Python 自带 Tcl, 最可靠)
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.update()
            folder = filedialog.askdirectory(title="选择皮肤文件夹")
            root.destroy()
            if folder and os.path.isdir(folder):
                return folder
        except Exception:
            pass
        # 2) PowerShell 对话框兜底
        import subprocess

        script = (
            "Add-Type -AssemblyName System.Windows.Forms\n"
            "$f = New-Object System.Windows.Forms.FolderBrowserDialog\n"
            "$f.Description = '\u9009\u62e9\u76ae\u80a4\u6587\u4ef6\u5939'\n"
            "if ($f.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $f.SelectedPath }\n"
        )
        for exe in ("powershell", "pwsh"):
            try:
                r = subprocess.run(
                    [exe, "-NoProfile", "-STA", "-Command", script],
                    capture_output=True,
                    timeout=180,
                )
                out = r.stdout.decode("utf-8", "ignore").strip()
                if out and os.path.isdir(out):
                    return out
            except Exception:
                continue
        print("文件夹选择失败: 请改用命令行指定皮肤文件夹路径")
        return None

    def _import_folder(self):
        """弹出文件夹选择对话框, 导入该文件夹内的皮肤图片列表。"""
        folder = self._choose_folder()
        if folder:
            self.folder = folder
            self.skin_list = self._scan_skins(folder)
            self._update_search()
            if self.skin_list:
                self.skin_index = 0
                self._reload_skin(self.skin_list[0])
                print(f"已导入 {len(self.skin_list)} 张皮肤: {folder}")
            else:
                print(f"该文件夹没有皮肤图片: {folder}")

    def _handle_drop(self, path):
        """处理拖拽到窗口的文件/文件夹: 文件直接加载为当前皮肤,
        文件夹则扫描其中第一张皮肤并设为当前文件夹。"""
        path = str(path)
        if self.edit_mode:  # 编辑中忽略拖拽, 避免未保存的笔迹丢失
            print("编辑中, 已忽略拖入的文件:", path)
            return
        if os.path.isdir(path):
            self.folder = path
            self.skin_list = self._scan_skins(path)
            self._update_search()
            if self.skin_list:
                self.skin_index = 0
                self._reload_skin(self.skin_list[0])
                print(f"已导入 {len(self.skin_list)} 张皮肤: {path}")
            else:
                print(f"该文件夹没有皮肤图片: {path}")
        elif path.lower().endswith((".png", ".jpg", ".jpeg")):
            self.folder = os.path.dirname(os.path.abspath(path))
            self.skin_list = self._scan_skins(self.folder)
            try:
                self.skin_index = self.skin_list.index(path)
            except ValueError:
                self.skin_index = 0
            self._update_search()
            self._reload_skin(path)
        else:
            print("不支持的拖拽文件:", path)

    def _prev(self):
        if self.skin_list:
            self.skin_index = (self.skin_index - 1) % len(self.skin_list)
            self._reload_skin(self.skin_list[self.skin_index])

    def _next(self):
        if self.skin_list:
            self.skin_index = (self.skin_index + 1) % len(self.skin_list)
            self._reload_skin(self.skin_list[self.skin_index])

    def _refresh(self):
        if self.current_skin:
            self._reload_skin(self.current_skin)

    def _hit_search(self, pos):
        px, py = pos
        if px < self.x0:
            return False
        sx, sy, sw, sh = self.search_rect
        gx = self.x0 + sx
        return gx <= px <= gx + sw and sy <= py <= sy + sh

    def _hit_search_list(self, pos):
        px, py = pos
        if px < self.x0:
            return None
        x = self.x0 + 10
        y0 = 400
        ih = 26
        if x <= px <= x + 160 and y0 <= py < self.H - 12:
            r = (py - y0) // ih
            idx = self.search_scroll + r
            if 0 <= idx < len(self.search_results):
                return idx
        return None

    def _mouse_in_list(self):
        return self._hit_search_list(pygame.mouse.get_pos()) is not None

    def _on_list_click(self, index):
        if self.edit_mode:  # 编辑中不允许双击换皮肤, 避免丢失未保存的笔迹
            return
        now = pygame.time.get_ticks()
        if index == self._click_item and (now - self._click_time) < 600:
            # 双击 → 切换到该皮肤
            if 0 <= index < len(self.search_results):
                f = self.search_results[index]
                try:
                    self.skin_index = self.skin_list.index(f)
                except ValueError:
                    pass
                self._reload_skin(f)
            self._click_item = None
        else:
            self._click_item = index
            self._click_time = now

    def _paste_search(self):
        try:
            import pygame.scrap

            pygame.scrap.init()
            for mt in ("text/plain;charset=utf-8", "text/plain", "UTF8_STRING"):
                try:
                    data = pygame.scrap.get(mt)
                    if data:
                        t = (
                            data.decode("utf-8", "ignore")
                            if isinstance(data, bytes)
                            else data
                        )
                        self.search_text += t
                        self._update_search()
                        return
                except Exception:
                    continue
        except Exception:
            pass

    def _hit_button(self, pos):
        px, py = pos
        if px < self.x0:
            return None  # 点在 3D 预览区
        for label, (rx, y, w, h) in self.buttons:
            x = self.x0 + rx
            if x <= px <= x + w and y <= py <= y + h:
                return label
        return None

    def _handle_button(self, label):
        # 编辑模式下除手臂模式外其它按钮都锁住, 避免中途换皮肤丢失编辑
        if self.edit_mode and label not in ("宽形", "纤细"):
            return
        if label == "导入文件夹":
            self._import_folder()
        elif label == "上一张":
            self._prev()
        elif label == "下一张":
            self._next()
        elif label == "刷新":
            self._refresh()
        elif label == "重命名":
            self._rename()
        elif label == "走路":
            self._start_anim("walk")
        elif label == "挥手":
            self._start_anim("wave")
        elif label == "宽形":
            self._set_arm_style("classic")
        elif label == "纤细":
            self._set_arm_style("slim")
        elif label == "编辑皮肤":
            self._enter_edit()

    def _start_anim(self, kind):
        """播放指定动画(走路/挥手), 从当前视角位置开始, 持续一段时间后复位。"""
        if not self.current_skin:
            return
        self.anim = kind
        self.anim_start = pygame.time.get_ticks() / 1000.0

    def _joint_of(self, name):
        """返回部位顶部关节点(肩/髋), 动画绕它旋转。"""
        parts = build_parts(self._arm_style)
        p = parts[name]
        pos = p["pos"]
        h = p["size"][1]
        return (pos[0], pos[1] + h / 2.0, pos[2])

    def _anim_angles(self, t):
        """根据动画类型和时间返回 {部位: (旋转轴, 角度度)}。
        走路: 双腿绕 X 轴前后交替摆动, 手臂同侧反向小幅摆动;
        挥手: 右臂绕 Z 轴在身体侧面抬起挥动。"""
        if self.anim == "walk":
            ph = t * 2 * math.pi / 1.1  # 步频约 1.1 秒一步
            return {
                "right_leg": ((1, 0, 0), 35 * math.sin(ph)),
                "left_leg": ((1, 0, 0), -35 * math.sin(ph)),
                "right_arm": ((1, 0, 0), -20 * math.sin(ph)),
                "left_arm": ((1, 0, 0), 20 * math.sin(ph)),
            }
        if self.anim == "wave":
            # 右臂绕肩朝前上方抬起挥手(像打招呼), 从正面看明显
            ph = t * 2 * math.pi / 0.8
            return {"right_arm": ((1, 0, 0), -45 - 30 * math.sin(ph))}
        return {}

    def _merge_geo(self, parts, angles=None):
        """把按部位几何合并为单一数组; 若给出 angles, 先对各部位绕关节旋转。
        合并交给 merge_parts: 空部位(整层透明的 overlay)按行拼接, 不会变成 (N,0)。"""
        if angles:
            parts = {
                name: (
                    (
                        rotate_verts(v, self._joint_of(name), *angles[name])
                        if name in angles
                        else v
                    ),
                    t_,
                    n_,
                )
                for name, (v, t_, n_) in parts.items()
            }
        return merge_parts(parts)

    def _rename(self):
        """重命名当前皮肤文件(用 tkinter 输入对话框)。"""
        if not self.current_skin:
            return
        try:
            import tkinter as tk
            from tkinter import simpledialog

            root = tk.Tk()
            root.withdraw()
            root.update()
            old = os.path.basename(self.current_skin)
            new = simpledialog.askstring(
                "重命名皮肤", "输入新文件名:", initialvalue=old, parent=root
            )
            root.destroy()
        except Exception as e:
            print("重命名对话框失败:", e)
            return
        if not new:
            return
        new = new.strip()
        if not new or "/" in new or "\\" in new or ":" in new:
            print("文件名非法")
            return
        ext = os.path.splitext(old)[1] or ".png"
        if not new.lower().endswith((".png", ".jpg", ".jpeg")):
            new += ext
        dst = os.path.join(os.path.dirname(self.current_skin), new)
        if os.path.abspath(dst) == os.path.abspath(self.current_skin):
            return
        if os.path.exists(dst):
            print("同名文件已存在")
            return
        try:
            os.rename(self.current_skin, dst)
            self.skin_list[self.skin_index] = dst
            self.current_skin = dst
            self._reload_skin(dst)
            self._save_state()
            print(f"已重命名为: {new}")
        except Exception as e:
            print("重命名失败:", e)

    def _save_state(self):
        """保存当前文件夹+文件名+序号, 下次启动恢复。

        防护: 已被判定出 bug 的皮肤(在黑名单 _failed_skins 里)绝不写入 json ——
        一旦写进去, 下次启动就会恢复到这张坏皮肤、一开就崩, 整个程序打不开。
        这类情况下 skin_name 写成 None, 下次启动改为从文件夹里的正常皮肤开始。"""
        try:
            import json

            safe_skin = self.current_skin
            if safe_skin and safe_skin in getattr(self, "_failed_skins", ()):
                safe_skin = None  # 坏皮肤不进状态文件
            state = {
                "folder": self.folder,
                "skin_name": os.path.basename(safe_skin) if safe_skin else None,
                "index": self.skin_index,
                "arm_style": getattr(self, "_arm_style", "slim"),
            }
            with open(self._state_path(), "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False)
        except Exception:
            pass

    @staticmethod
    def _state_path():
        return os.path.join(_base_dir(), "skin_preview_state.json")

    def _update_icon(self, path=None):
        """用皮肤头部正面作为窗口图标。默认固定用原始皮肤 pic1 的头。"""
        base = _base_dir()
        orig = os.path.join(_skins_dir(), "pic1.png")
        if path is None:
            src = orig if os.path.exists(orig) else self.current_skin
        else:
            src = path
        if not src or not os.path.exists(src):
            return
        try:
            img = pygame.image.load(src).convert_alpha()
            w = img.get_width()
            if w < 64 or w % 16 != 0:
                return
            s = w // 64
            # 头部正面区域(标准 UV: head front 位于 (8,8) 8x8, 按 s 缩放)
            hw = 8 * s
            icon = pygame.Surface((hw, hw), pygame.SRCALPHA)
            icon.blit(img, (0, 0), (hw, hw, hw, hw))  # 头部 base 正面
            # 叠加 overlay 头部(帽子/刘海等), 若该区域有内容
            try:
                ov = img.subsurface((hw + 32 * s, hw, hw, hw)).copy()
                if ov.get_bounding_rect():
                    icon.blit(ov, (0, 0))
            except Exception:
                pass
            icon = pygame.transform.scale(
                icon, (256, 256)
            )  # NEAREST 像素风格放大, 保持清晰
            pygame.display.set_icon(icon)
            self._apply_taskbar_icon(icon)  # 覆盖任务栏图标
        except Exception:
            pass

    def _apply_taskbar_icon(self, icon_surf):
        """用 WM_SETICON 强制把任务栏/标题栏图标都换成皮肤头。
        (pygame.set_icon 只改窗口小图标, 任务栏大图标需底层 API 覆盖)"""
        try:
            from PIL import Image
            import ctypes, tempfile

            user32 = ctypes.windll.user32
            hwnd = pygame.display.get_wm_info().get("window")
            if not hwnd:
                return
            user32.LoadImageW.argtypes = [
                ctypes.c_void_p,
                ctypes.c_wchar_p,
                ctypes.c_uint,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint,
            ]
            user32.LoadImageW.restype = ctypes.c_void_p
            user32.SendMessageW.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_uint,
                ctypes.c_void_p,
            ]
            w, h = icon_surf.get_size()
            data = pygame.image.tobytes(icon_surf, "RGBA")
            pil = Image.frombytes("RGBA", (w, h), data)
            ico_path = os.path.join(tempfile.gettempdir(), "skin_viewer_icon.ico")
            pil.save(
                ico_path,
                format="ICO",
                sizes=[(256, 256), (64, 64), (48, 48), (32, 32), (16, 16)],
            )
            h_icon = user32.LoadImageW(
                None, ico_path, 1, 0, 0, 0x10
            )  # IMAGE_ICON | LR_LOADFROMFILE
            if h_icon:
                WM_SETICON = 0x0080
                user32.SendMessageW(hwnd, WM_SETICON, 0, h_icon)  # ICON_SMALL
                user32.SendMessageW(hwnd, WM_SETICON, 1, h_icon)  # ICON_BIG(任务栏)
        except Exception:
            pass

    def _surface_to_texture(self, surf):
        surf = surf.convert_alpha()
        w, h = surf.get_size()
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        data = pygame.image.tobytes(surf, "RGBA", False)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data
        )
        return tex

    def _make_font(self, size=32):
        """用系统字体文件创建 pygame 字体(中文字体), 避开 SysFont 的 bug。
        用高分辨率渲染后再缩放到目标, 保证文字清晰。"""
        for p in (
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\simsun.ttc",
        ):
            if os.path.exists(p):
                return pygame.font.Font(p, size)
        return pygame.font.Font(None, size)

    def _render_text(self, text, x, y, w, h):
        """在正交投影下绘制按钮文字(纹理四边形)。"""
        try:
            if self._font is None or self._font.size != 32:
                self._font = self._make_font(32)
            surf = self._font.render(text, True, (235, 240, 245))
        except Exception:
            return
        tw, th = surf.get_size()
        scale = min((w - 8) / max(1, tw), (h - 6) / max(1, th))
        if scale < 0.5:
            return
        dw, dh = int(tw * scale), int(th * scale)
        sx, sy = x + (w - dw) / 2, y + (h - dh) / 2
        tex = self._surface_to_texture(surf)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex)
        glColor3f(1, 1, 1)
        # v 上下翻转让文字正立(纹理第0行在四边形顶部)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(sx, sy)
        glTexCoord2f(1, 0)
        glVertex2f(sx + dw, sy)
        glTexCoord2f(1, 1)
        glVertex2f(sx + dw, sy + dh)
        glTexCoord2f(0, 1)
        glVertex2f(sx, sy + dh)
        glEnd()
        glDeleteTextures([tex])
        glDisable(GL_TEXTURE_2D)

    def _render_text_left(self, text, x, y, w, h):
        """左对齐文本(用于侧边栏信息), 超出宽度自动缩放。"""
        try:
            if self._font is None or self._font.size != 20:
                self._font = self._make_font(20)
            surf = self._font.render(text, True, (225, 230, 238))
        except Exception:
            return
        tw, th = surf.get_size()
        scale = min(w / max(1, tw), (h - 4) / max(1, th))
        if scale < 0.25:
            return
        dw, dh = int(tw * scale), int(th * scale)
        sx, sy = x, y + (h - dh) / 2
        tex = self._surface_to_texture(surf)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex)
        glColor3f(1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(sx, sy)
        glTexCoord2f(1, 0)
        glVertex2f(sx + dw, sy)
        glTexCoord2f(1, 1)
        glVertex2f(sx + dw, sy + dh)
        glTexCoord2f(0, 1)
        glVertex2f(sx, sy + dh)
        glEnd()
        glDeleteTextures([tex])
        glDisable(GL_TEXTURE_2D)

    def _render_text_wrap(
        self, text, x, y, w, size, color=(0.85, 0.92, 1.0), line_h=30
    ):
        """大字体多行文本(用于显示长文件名, 自动换行)。"""
        font = self._make_font(size)
        lines, cur = [], ""
        for ch in text:
            if ch == "\n":
                lines.append(cur)
                cur = ""
                continue
            if font.size(cur + ch)[0] > w and cur:
                lines.append(cur)
                cur = ch
            else:
                cur += ch
        if cur:
            lines.append(cur)
        for i, ln in enumerate(lines):
            try:
                surf = font.render(ln, True, tuple(int(c * 255) for c in color))
            except Exception:
                continue
            tex = self._surface_to_texture(surf)
            tw, th = surf.get_size()
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, tex)
            glColor3f(1, 1, 1)
            glBegin(GL_QUADS)
            glTexCoord2f(0, 0)
            glVertex2f(x, y + i * line_h)
            glTexCoord2f(1, 0)
            glVertex2f(x + tw, y + i * line_h)
            glTexCoord2f(1, 1)
            glVertex2f(x + tw, y + i * line_h + th)
            glTexCoord2f(0, 1)
            glVertex2f(x, y + i * line_h + th)
            glEnd()
            glDeleteTextures([tex])
        glDisable(GL_TEXTURE_2D)

    def _draw_buttons(self):
        """右侧独立侧边栏: 正交投影盖在右侧, 竖排按钮 + 当前皮肤信息。"""
        glViewport(0, 0, self.W, self.H)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.W, self.H, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_COLOR_MATERIAL)
        glEnable(GL_BLEND)
        # 侧边栏背景(右侧 x0..W)
        glColor4f(0.16, 0.18, 0.23, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(self.x0, 0)
        glVertex2f(self.W, 0)
        glVertex2f(self.W, self.H)
        glVertex2f(self.x0, self.H)
        glEnd()
        # 左分隔线
        glColor4f(0.45, 0.5, 0.6, 1.0)
        glBegin(GL_LINES)
        glVertex2f(self.x0 + 0.5, 0)
        glVertex2f(self.x0 + 0.5, self.H)
        glEnd()
        # 标题
        self._render_text("皮肤预览", self.x0 + 10, 4, 160, 16)
        # 搜索框
        sx, sy, sw, sh = self.search_rect
        gx = self.x0 + sx
        (
            glColor4f(0.30, 0.34, 0.44, 1.0)
            if not self.search_active
            else glColor4f(0.36, 0.42, 0.55, 1.0)
        )
        glBegin(GL_QUADS)
        glVertex2f(gx, sy)
        glVertex2f(gx + sw, sy)
        glVertex2f(gx + sw, sy + sh)
        glVertex2f(gx, sy + sh)
        glEnd()
        glColor4f(0.6, 0.7, 0.8, 1.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(gx, sy)
        glVertex2f(gx + sw, sy)
        glVertex2f(gx + sw, sy + sh)
        glVertex2f(gx, sy + sh)
        glEnd()
        stext = self.search_text
        if len(stext) > 17:
            stext = "..." + stext[-14:]
        if self.search_active:
            stext += "|"
        self._render_text_left(
            stext if stext else "(搜索文件名)", gx + 4, sy + 2, sw - 8, sh - 4
        )
        # 按钮(坐标加 x0 偏移); 手臂模式按钮当前选中项高亮
        for label, (rx, y, w, h) in self.buttons:
            x = self.x0 + rx
            active = (label == "宽形" and self._arm_style == "classic") or (
                label == "纤细" and self._arm_style == "slim"
            )
            if active:
                glColor4f(0.30, 0.52, 0.78, 1.0)
            else:
                glColor4f(0.24, 0.28, 0.37, 1.0)
            glBegin(GL_QUADS)
            glVertex2f(x, y)
            glVertex2f(x + w, y)
            glVertex2f(x + w, y + h)
            glVertex2f(x, y + h)
            glEnd()
            glColor4f(0.5, 0.6, 0.72, 1.0)
            glBegin(GL_LINE_LOOP)
            glVertex2f(x, y)
            glVertex2f(x + w, y)
            glVertex2f(x + w, y + h)
            glVertex2f(x, y + h)
            glEnd()
            self._render_text(label, x, y, w, h)
        # 图片计数 + 当前文件名
        n = len(self.skin_list) if self.skin_list else 0
        self._render_text_left(
            f"图片 {self.skin_index + 1}/{n}  共{n}张", self.x0 + 10, 334, 160, 18
        )
        cname = os.path.basename(self.current_skin) if self.current_skin else "无"
        if len(cname) > 16:
            cname = cname[:13] + "..."
        self._render_text_left(f"当前: {cname}", self.x0 + 10, 354, 160, 18)
        # 搜索结果列表
        self._render_text_left(
            f"搜索结果 ({len(self.search_results)})", self.x0 + 10, 376, 160, 16
        )
        self._draw_search_list()
        # 编辑模式: 3D 区左上角提示 + 顶部边框标记
        if self.edit_mode:
            glColor4f(0.98, 0.75, 0.15, 1.0)
            glBegin(GL_LINES)
            glVertex2f(0, 0.5)
            glVertex2f(self.x0, 0.5)
            glVertex2f(0, 0.5)
            glVertex2f(0.5, 0.5)
            glEnd()
            glColor4f(0.10, 0.12, 0.16, 0.85)
            glBegin(GL_QUADS)
            glVertex2f(0, 0)
            glVertex2f(430, 0)
            glVertex2f(430, 30)
            glVertex2f(0, 30)
            glEnd()
            hint = "编辑模式 · 左键绘制 / 右键旋转 / 滚轮缩放 · " f"笔刷{self.brush}px"
            if self.eyedropper:
                hint = "编辑模式 · 取色中: 左键点模型吸取颜色"
            self._render_text_left(hint, 8, 2, 414, 26)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def _draw_search_list(self):
        """绘制搜索结果列表(滚动), 每项为一行文件名。"""
        x = self.x0 + 10
        w = 160
        y0 = 400
        ih = 26
        rows = (self.H - y0 - 12) // ih
        if rows < 1:
            rows = 1
        for r in range(rows):
            idx = self.search_scroll + r
            if idx >= len(self.search_results):
                break
            f = self.search_results[idx]
            yy = y0 + r * ih
            if f == self.current_skin:  # 高亮当前皮肤
                glColor4f(0.28, 0.34, 0.46, 1.0)
                glBegin(GL_QUADS)
                glVertex2f(x, yy)
                glVertex2f(x + w, yy)
                glVertex2f(x + w, yy + ih)
                glVertex2f(x, yy + ih)
                glEnd()
            # 项分隔线
            glColor4f(0.24, 0.28, 0.36, 1.0)
            glBegin(GL_LINES)
            glVertex2f(x, yy + ih)
            glVertex2f(x + w, yy + ih)
            glEnd()
            name = os.path.basename(f)
            if len(name) > 16:
                name = name[:13] + "..."
            self._render_text_left(name, x + 2, yy + 2, w - 4, ih - 4)

    def _draw_geo(self, geo):
        """立即模式绘制一组几何 (verts, texs, norms)。
        每帧显式启用 GL_TEXTURE_2D —— 侧边栏绘制会禁用纹理,
        否则从第 2 帧起纹理失效、模型显示为默认白色。"""
        verts, texs, norms = geo
        glEnable(GL_TEXTURE_2D)
        n_v = len(verts)
        # 防御性检查: 顶点/UV/法线三者数量必须一致。二者不一致时 PyOpenGL 会收到
        # 0 字节数组并抛 "Expected 12 byte array, got 0 byte array" —— 皮肤只有
        # 第一层有像素(overlay 整层透明)时就触发过这个崩溃, 见 merge_parts 注释。
        if self.texture is None or n_v == 0 or len(texs) != n_v or len(norms) != n_v:
            return
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glBegin(GL_TRIANGLES)
        for i in range(len(verts)):
            glNormal3fv(norms[i])
            glTexCoord2fv(texs[i])
            glVertex3fv(verts[i])
        glEnd()

    def _setup_gl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_ALPHA_TEST)
        glAlphaFunc(GL_GREATER, 0.5)  # 丢弃 alpha=0(透明)片段, 模拟逐像素渲染, 消除镂空
        glAlphaFunc(GL_GREATER, 0.5)
        glClearColor(0.16, 0.18, 0.22, 1.0)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 2.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.35, 0.35, 0.35, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.9, 0.9, 0.9, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.3, 0.3, 0.3, 1.0])
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    def _resize(self, w, h):
        w = max(int(w), 620)
        h = max(int(h), 400)
        self.W, self.H = w, h
        self.x0 = self.W - self.SW
        if self.x0 < 460:
            self.x0 = 460
            self.W = self.x0 + self.SW
        self.screen = pygame.display.set_mode((self.W, self.H), self._flags)
        self._setup_gl()  # resize 后 GL 上下文可能重置, 重新初始化状态
        if self.current_skin:
            try:
                # 重建纹理: 编辑模式下要保留未保存的笔迹, 直接复用当前显示数组
                if self.skin_arr_display is not None:
                    self.texture = upload_texture(self.skin_arr_display)
                else:
                    self.texture = load_skin(self.current_skin)
            except Exception as e:
                print("重建纹理失败:", e)
            self._update_icon()  # resize 后恢复窗口图标

    def _render_no_skin_hint(self):
        """无初始皮肤时在 3D 预览区中央画提示文字。"""
        glViewport(0, 0, self.W, self.H)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.W, self.H, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_COLOR_MATERIAL)
        glEnable(GL_BLEND)
        cx = self.x0 // 2
        cy = self.H // 2
        self._render_text_wrap(
            "未找到皮肤\n点击右侧「导入文件夹」选择皮肤文件夹",
            cx - 150,
            cy - 40,
            300,
            26,
        )
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def _render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # 只要右侧皮肤列表有图片, 就默认选当前序号皮肤渲染;
        # 仅当完全没有任何皮肤时才显示"未找到皮肤"提示。
        if not self.skin_list:
            glDisable(GL_ALPHA_TEST)
            self._render_no_skin_hint()
            self._draw_buttons()
            pygame.display.flip()
            return
        if self.current_skin is None and self.skin_list:
            idx = min(max(self.skin_index, 0), len(self.skin_list) - 1)
            self._reload_skin(self.skin_list[idx])
        if self.current_skin is None:
            # 仍无法加载(可能是单张皮肤损坏), 显示提示避免空白
            glDisable(GL_ALPHA_TEST)
            self._render_no_skin_hint()
            self._draw_buttons()
            pygame.display.flip()
            return
        # 3D 模型渲染占满 3D 预览区(0..x0), 侧边栏用正交投影盖在右侧
        glViewport(0, 0, self.x0, self.H)
        # 丢弃 alpha=0(透明)片段, 模拟 mod 逐像素渲染: 皮肤透明处不显示, 消除缝隙镂空。
        # 不用背面剔除(cull): 各面绕序不统一会误剔面, 且 mod 本身也不做剔除。
        glEnable(GL_ALPHA_TEST)
        glAlphaFunc(GL_GREATER, 0.5)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(40, self.x0 / float(self.H), 0.1, 500.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(self.pan[0], self.pan[1] - 16.0, -self.dist)
        glRotatef(self.pitch, 1, 0, 0)
        glRotatef(self.yaw, 0, 1, 0)

        if self.wireframe:
            glDisable(GL_LIGHTING)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glColor3f(0.15, 0.6, 0.95)
            self._draw_geo(self.base_geo)
        else:
            if self.lit:
                glEnable(GL_LIGHTING)
            else:
                glDisable(GL_LIGHTING)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            self._draw_geo(self.base_geo)  # 先画第一层(身体)
            self._draw_geo(self.overlay_geo)  # 再画第二层(帽子/外套等)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        # 编辑模式: 悬停像素高亮(在 3D 投影下画, 不移动相机)
        self.hover_texel = None
        if self.edit_mode:
            mx, my = pygame.mouse.get_pos()
            if 0 <= mx < self.x0 and not self.rotate_mode:
                self.hover_texel = self._pick_texel(mx, my)
            # 调色板二维图上悬停 / 选中的像素, 在 3D 里同步高亮(穿透显示)。
            # 二维图上空白像素(alpha=0)也高亮, 因此 include_empty=True
            for pt2 in (self.hover2d, self.sel2d):
                if pt2:
                    for cell in self._texel_cells(pt2[0], pt2[1], include_empty=True):
                        self._draw_cell(
                            cell,
                            fill=(0.2, 0.9, 1.0, 0.5),
                            line=(0.6, 0.95, 1.0, 1.0),
                            through=True,
                        )
            self._draw_hover(self.hover_texel)
        # 侧边栏正交绘制: 关闭 alpha 剔除, 避免影响文字纹理的半透明边缘
        glDisable(GL_ALPHA_TEST)
        self._draw_buttons()
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            self.clock.tick(60)
            # 调色板窗口(tkinter)每帧泵一次事件, 与 3D 编辑保持同步
            if self.palette is not None:
                self.palette.pump()
            # 动画推进: 依据当前时间对部位做关节旋转, 到期后复位静态姿态
            if self.anim:
                t = pygame.time.get_ticks() / 1000.0 - self.anim_start
                if t > self.anim_dur.get(self.anim, 2.0):
                    self.anim = None
                    self.base_geo = self._merge_geo(self._parts_base)
                    self.overlay_geo = self._merge_geo(self._parts_overlay)
                else:
                    a = self._anim_angles(t)
                    self.base_geo = self._merge_geo(self._parts_base, a)
                    self.overlay_geo = self._merge_geo(self._parts_overlay, a)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.VIDEORESIZE:
                    self._resize(ev.w, ev.h)
                elif ev.type == pygame.DROPFILE:
                    self._handle_drop(ev.file)
                elif ev.type == pygame.KEYDOWN:
                    if self.search_active:
                        if ev.key in (
                            pygame.K_RETURN,
                            pygame.K_KP_ENTER,
                            pygame.K_ESCAPE,
                        ):
                            self.search_active = False
                        elif ev.key == pygame.K_BACKSPACE:
                            self.search_text = self.search_text[:-1]
                            self._update_search()
                        elif ev.key == pygame.K_v and (ev.mod & pygame.KMOD_CTRL):
                            self._paste_search()
                        elif ev.unicode:
                            self.search_text += ev.unicode
                            self._update_search()
                    elif ev.key == pygame.K_ESCAPE:
                        if self.edit_mode:
                            self._exit_edit()  # 编辑中 ESC = 退出编辑(不保存)
                        else:
                            running = False
                    elif ev.key == pygame.K_w:
                        self.wireframe = not self.wireframe
                    elif ev.key == pygame.K_r:
                        self.yaw, self.pitch, self.dist = -40.0, -10.0, 68.0
                        self.pan = [0.0, 0.0]
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    if ev.button == 1:
                        pos = pygame.mouse.get_pos()
                        if self.edit_mode and pos[0] < self.x0:
                            if self.rotate_mode:
                                # 旋转模式: 左键拖动 = 转视角(和正常预览一样)
                                self._dragging = "rot"
                                self._last = pos
                            else:
                                # 编辑模式: 3D 区内左键 = 绘制/取色/删除
                                self._painting = True
                                self.begin_stroke()
                                self._paint_at(*pos)
                        elif self._hit_search(pos):
                            self.search_active = True
                        else:
                            hit = self._hit_button(pos)
                            if hit is not None:
                                self._handle_button(hit)
                                self.search_active = False
                            else:
                                item = self._hit_search_list(pos)
                                if item is not None:
                                    self._on_list_click(item)
                                elif not self.edit_mode:
                                    self._dragging = "rot"
                                    self._last = pos
                    elif ev.button == 3:
                        # 编辑模式下右键改为旋转视角(左键已用于绘制);
                        # 开了旋转模式后左键管旋转, 右键改管平移
                        if self.edit_mode:
                            self._dragging = "pan" if self.rotate_mode else "rot"
                        else:
                            self._dragging = "pan"
                        self._last = pygame.mouse.get_pos()
                    elif ev.button == 4:
                        if self._mouse_in_list():
                            self.search_scroll = max(0, self.search_scroll - 1)
                        else:
                            self.dist = max(10.0, self.dist * 0.92)
                    elif ev.button == 5:
                        if self._mouse_in_list():
                            self.search_scroll = min(
                                len(self.search_results) - 1, self.search_scroll + 1
                            )
                        else:
                            self.dist = min(300.0, self.dist * 1.08)
                elif ev.type == pygame.MOUSEBUTTONUP:
                    self._dragging = None
                    if ev.button == 1:
                        if self._painting:
                            self.end_stroke()
                        self._painting = False
                elif ev.type == pygame.MOUSEMOTION:
                    if self._painting and self.edit_mode:  # 按住左键拖动 = 连续绘制
                        x, y = pygame.mouse.get_pos()
                        if x < self.x0:
                            self._paint_at(x, y)
                    if not self._dragging:
                        continue
                    x, y = pygame.mouse.get_pos()
                    dx, dy = x - self._last[0], y - self._last[1]
                    self._last = (x, y)
                    if self._dragging == "rot":
                        self.yaw += dx * 0.5
                        self.pitch += dy * 0.5
                        self.pitch = max(-90, min(90, self.pitch))
                    else:
                        self.pan[0] += dx * 0.02
                        self.pan[1] -= dy * 0.02
            try:
                self._render()
            except Exception as e:
                # 单帧渲染异常(多半是某张皮肤引起)不整个退出: 隔离该皮肤(改名 ERROR_<原名>)
                # 并清掉状态里的记录, 随后自动回落到列表里的下一张, 窗口继续可用。
                msg = f"{type(e).__name__}: {e}"
                del e
                gc.collect()
                self._handle_skin_error(msg, "渲染")
        self._close_palette()
        pygame.quit()


class PaletteWindow:
    """编辑模式的调色板窗口(独立的 tkinter 窗口)。
    提供: 取色(吸管)、RGB 调色、预设色板、笔刷大小、保存并退出 / 退出。"""

    SWATCHES = [
        "#ffffff",
        "#d8d8d8",
        "#a0a0a0",
        "#6c6c6c",
        "#3c3c3c",
        "#141414",
        "#e8c39e",
        "#d3a06c",
        "#b57a4a",
        "#8a5a30",
        "#5c3a1c",
        "#2f1d0e",
        "#e05a5a",
        "#e8a13a",
        "#e8d84a",
        "#8ad04a",
        "#4ac9a0",
        "#3aa8e0",
        "#4a6ce8",
        "#8a4ae8",
        "#c94ad0",
        "#e04a96",
        "#7a4a2c",
        "#2b3a4a",
    ]

    def __init__(self, viewer):
        import tkinter as tk

        self.tk = tk
        self.v = viewer
        self._guard = False
        self.root = tk.Tk()
        self.root.title("皮肤编辑 · 调色板")
        # 尽量贴在屏幕右侧(调色板宽约 420, 高约 850, 顶部留 60px)
        try:
            self.root.update_idletasks()
            ww = max(self.root.winfo_reqwidth(), 420)
            self.root.geometry(
                "+%d+%d" % (max(0, self.root.winfo_screenwidth() - ww - 16), 60)
            )
        except Exception:
            pass
        self.root.resizable(False, False)
        body = tk.Frame(self.root, padx=10, pady=8)
        body.pack()

        # 当前颜色预览 + 十六进制
        top = tk.Frame(body)
        top.pack(fill="x")
        self.canvas = tk.Canvas(
            top,
            width=86,
            height=44,
            bg="#20242c",
            highlightthickness=1,
            highlightbackground="#4a5568",
        )
        self.canvas.pack(side="left")
        self.patch = self.canvas.create_rectangle(
            6, 6, 80, 38, fill="#000000", outline=""
        )
        info = tk.Frame(top)
        info.pack(side="left", fill="x", expand=True)
        tk.Label(info, text="当前颜色 (hex)", anchor="w").pack(fill="x")
        self.hex_var = tk.StringVar()
        self.hex_entry = tk.Entry(info, textvariable=self.hex_var, width=10)
        self.hex_entry.pack(fill="x")
        self.hex_entry.bind("<Return>", self._on_hex)

        # RGB 滑条
        self.scales = {}
        for i, ch in enumerate("RGB"):
            row = tk.Frame(body)
            row.pack(fill="x", pady=1)
            tk.Label(
                row, text=ch, width=2, fg=["#e06a6a", "#6ae08a", "#6a9ae0"][i]
            ).pack(side="left")
            s = tk.Scale(
                row,
                from_=0,
                to=255,
                orient="horizontal",
                length=232,
                showvalue=True,
                command=self._on_scale,
                bg="#f0f0f0",
            )
            s.pack(side="left")
            self.scales[ch] = s

        # 预设色板
        sw_frame = tk.Frame(body)
        sw_frame.pack(pady=6)
        for idx, col in enumerate(self.SWATCHES):
            b = tk.Frame(sw_frame, width=26, height=20, bg=col, relief="raised", bd=1)
            b.grid(row=idx // 6, column=idx % 6, padx=1, pady=1)
            b.bind("<Button-1>", lambda e, c=col: self._on_swatch(c))

        # 源皮肤二维图(与 3D 视图双向联动)
        tk.Label(body, text="源皮肤二维图（左键可在图上直接绘制）", anchor="w").pack(
            fill="x", pady=(4, 2)
        )
        arr = viewer.skin_arr_display
        tex_w = arr.shape[1] if arr is not None else 64
        # 二维图缩放: 64 宽→4x(256), 128 宽→3x(384), 上限 4x
        self.SCALE2D = min(4, max(2, 384 // max(1, tex_w)))
        side = tex_w * self.SCALE2D
        self.canvas2d = tk.Canvas(
            body,
            width=side,
            height=side,
            bg="#20242c",
            highlightthickness=1,
            highlightbackground="#4a5568",
        )
        self.canvas2d.pack()
        tk.Label(
            body,
            text="黄框=鼠标所在像素 · 青框=二维选中(3D 同步)",
            anchor="w",
            fg="#777777",
        ).pack(fill="x")
        self._photo2d = None
        self._ver2d = -1
        self.canvas2d.bind("<Motion>", self._on_2d_motion)
        self.canvas2d.bind("<Button-1>", self._on_2d_click)
        self.canvas2d.bind("<Leave>", self._on_2d_leave)
        self._refresh_2d()

        # 取色 / 笔刷
        ops = tk.Frame(body)
        ops.pack(fill="x", pady=4)
        self.eye_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            ops, text="取色(吸管)", variable=self.eye_var, command=self._on_eye
        ).pack(side="left")
        tk.Label(ops, text=" 笔刷").pack(side="left")
        self.brush_var = tk.IntVar(value=1)
        for n in (1, 2, 3):
            tk.Radiobutton(
                ops,
                text=f"{n}px",
                variable=self.brush_var,
                value=n,
                command=self._on_brush,
            ).pack(side="left")

        # 工具行: 删除像素(橡皮) / 旋转模式 / 撤销
        tools = tk.Frame(body)
        tools.pack(fill="x", pady=3)
        self.erase_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            tools,
            text="删除像素(橡皮)",
            variable=self.erase_var,
            command=self._on_erase,
        ).pack(side="left")
        self.undo_btn = tk.Button(
            tools,
            text="撤销",
            width=8,
            command=self._on_undo,
        )
        self.undo_btn.pack(side="left", padx=(8, 0))

        rot = tk.Frame(body)
        rot.pack(fill="x", pady=(0, 2))
        self.rot_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            rot,
            text="旋转/平移模式（左键拖动转视角，不绘制）",
            variable=self.rot_var,
            command=self._on_rotate,
        ).pack(side="left")

        # 状态行(悬停像素信息)
        self.status = tk.Label(
            body, text="悬停在模型上选择像素", anchor="w", fg="#666666"
        )
        self.status.pack(fill="x")

        # 保存 / 退出
        btns = tk.Frame(body)
        btns.pack(fill="x", pady=(6, 0))
        tk.Button(
            btns,
            text="保存并退出",
            width=12,
            bg="#2f7a4a",
            fg="white",
            command=lambda: self._defer(self.v._save_edit),
        ).pack(side="left")
        tk.Button(
            btns,
            text="退出(不保存)",
            width=12,
            command=lambda: self._defer(self.v._exit_edit),
        ).pack(side="right")

        self.root.protocol("WM_DELETE_WINDOW", lambda: self._defer(self.v._exit_edit))
        self.set_color(self.v.paint_color)

    def _defer(self, fn):
        """把动作推迟到本次 tk 事件结束后执行(避免在按钮回调里销毁窗口)。"""
        try:
            self.root.after(20, fn)
        except Exception:
            pass

    def _on_scale(self, _=None):
        if self._guard:
            return
        rgb = tuple(int(self.scales[c].get()) for c in "RGB")
        self.v._set_color(rgb)
        self._guard = True  # 回写预览时不要再触发滑条回调
        try:
            hexs = "#%02x%02x%02x" % rgb
            self.canvas.itemconfig(self.patch, fill=hexs)
            self.hex_var.set(hexs)
        except Exception:
            pass
        self._guard = False

    def _on_swatch(self, col):
        rgb = tuple(int(col[i : i + 2], 16) for i in (1, 3, 5))
        self.v._set_color(rgb)
        self.set_color(rgb)

    def _on_hex(self, _=None):
        txt = self.hex_var.get().strip().lstrip("#")
        if len(txt) != 6:
            return
        try:
            rgb = tuple(int(txt[i : i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            return
        self.v._set_color(rgb)
        self.set_color(rgb)

    def _on_eye(self):
        self.v.eyedropper = bool(self.eye_var.get())
        if self.v.eyedropper:  # 取色与删除互斥
            self.v.erase_mode = False
            self.erase_var.set(False)

    def _on_erase(self):
        self.v.erase_mode = bool(self.erase_var.get())
        if self.v.erase_mode:
            self.v.eyedropper = False
            self.set_eyedropper(False)

    def _on_rotate(self):
        self.v.rotate_mode = bool(self.rot_var.get())
        if self.v.rotate_mode:
            self.v._painting = False
            self.v.hover_texel = None

    def _on_undo(self):
        n = self.v.undo()
        self._refresh_2d(force=True)
        self._draw_hl()
        try:
            self.status.config(
                text=f"已撤销上一笔（{n} 像素） · 剩余可撤销 {len(self.v.undo_stack)}"
            )
        except Exception:
            pass

    def _on_brush(self):
        self.v.brush = int(self.brush_var.get())

    def set_color(self, rgb):
        """外部(吸管取色)改色后同步 UI。"""
        self._guard = True
        try:
            for i, c in enumerate("RGB"):
                self.scales[c].set(int(rgb[i]))
            hexs = "#%02x%02x%02x" % tuple(rgb)
            self.hex_var.set(hexs)
            self.canvas.itemconfig(self.patch, fill=hexs)
        except Exception:
            pass
        self._guard = False

    def set_eyedropper(self, on):
        self._guard = True
        try:
            self.eye_var.set(bool(on))
        except Exception:
            pass
        self._guard = False

    def _on_2d_leave(self, _evt=None):
        self.v.hover2d = None
        self._draw_hl()

    def _on_2d_motion(self, evt):
        self.v.hover2d = self._pixel_at_2d(evt.x, evt.y)
        self._draw_hl()

    def _on_2d_click(self, evt):
        pt = self._pixel_at_2d(evt.x, evt.y)
        if pt is None:
            return
        self.v.sel2d = pt
        try:
            self.v.begin_stroke()
            self.v._paint_texel(pt[0], pt[1])
            self.v.end_stroke()
        except Exception as e:
            print("二维图绘制失败:", e)
            self.v.end_stroke()
        self._refresh_2d(force=True)
        self._draw_hl()

    def _pixel_at_2d(self, cx, cy):
        """画布坐标 → 皮肤像素坐标; 超出范围返回 None。"""
        arr = self.v.skin_arr_display
        if arr is None:
            return None
        s = self.SCALE2D
        x, y = int(cx // s), int(cy // s)
        h, w = arr.shape[0], arr.shape[1]
        if 0 <= x < w and 0 <= y < h:
            return (x, y)
        return None

    def _refresh_2d(self, force=False):
        """把当前皮肤图(原始数组: 文件内容+已改像素)重画到二维图。"""
        arr = self.v.skin_arr_raw
        if arr is None:
            return
        if not force and self._ver2d == self.v.texture_version:
            return
        try:
            from PIL import ImageTk
        except Exception as e:
            print("二维图初始化失败:", e)
            return
        h, w = arr.shape[0], arr.shape[1]
        # 透明像素铺棋盘格, 便于看清 alpha=0 区域
        check = np.zeros((h, w, 3), np.uint8)
        check[:] = 40
        ys, xs = np.mgrid[0:h, 0:w]
        check[((ys // 4) + (xs // 4)) % 2 == 0] = 68
        a = (arr[:, :, 3:4].astype(np.float32)) / 255.0
        rgb = arr[:, :, :3].astype(np.float32) * a + check.astype(np.float32) * (
            1.0 - a
        )
        img = Image.fromarray(rgb.astype(np.uint8), "RGB")
        s = self.SCALE2D
        img = img.resize((w * s, h * s), Image.NEAREST)
        self._photo2d = ImageTk.PhotoImage(img)
        self.canvas2d.delete("img")
        self.canvas2d.create_image(0, 0, anchor="nw", image=self._photo2d, tags="img")
        self._ver2d = self.v.texture_version
        self._draw_hl()

    def _draw_hl(self):
        """二维图上的高亮: 黄=鼠标所在像素(3D 悬停或二维悬停), 青=二维选中。"""
        c = self.canvas2d
        try:
            c.delete("hl")
        except Exception:
            return
        s = self.SCALE2D
        marks = []
        if self.v.hover_texel:
            marks.append((self.v.hover_texel["x"], self.v.hover_texel["y"], "#ffd24a"))
        elif self.v.hover2d:
            marks.append((self.v.hover2d[0], self.v.hover2d[1], "#ffd24a"))
        if self.v.sel2d:
            marks.append((self.v.sel2d[0], self.v.sel2d[1], "#59e9ff"))
        for x, y, col in marks:
            c.create_rectangle(
                x * s, y * s, (x + 1) * s, (y + 1) * s, outline=col, width=2, tags="hl"
            )

    def pump(self):
        """每帧泵一次 tk 事件并刷新悬停像素信息。"""
        if self.root is None:
            return
        try:
            # 滑条轮询: 部分 Tk 版本里 Scale.set 不触发 command,
            # 这里每帧比对一次, 保证拖动滑条一定改到画笔色。
            if not self._guard:
                rgb = tuple(int(self.scales[c].get()) for c in "RGB")
                if rgb != tuple(self.v.paint_color):
                    self.v.paint_color = rgb
                    hexs = "#%02x%02x%02x" % rgb
                    self.canvas.itemconfig(self.patch, fill=hexs)
                    self.hex_var.set(hexs)
            hit = self.v.hover_texel
            if hit is not None:
                px = self.v.skin_arr_display[hit["y"], hit["x"]]
                txt = (
                    f"像素 UV ({hit['x']}, {hit['y']})  "
                    f"色 #{int(px[0]):02x}{int(px[1]):02x}{int(px[2]):02x}  "
                    f"改动 {len(self.v.edited)}"
                )
            elif self.v.hover2d:
                px = self.v.skin_arr_raw[self.v.hover2d[1], self.v.hover2d[0]]
                if int(px[3]) > 0:
                    col = "色 #%02x%02x%02x" % (int(px[0]), int(px[1]), int(px[2]))
                else:
                    col = "空白像素(alpha=0)"
                txt = (
                    f"二维 ({self.v.hover2d[0]}, {self.v.hover2d[1]})  {col}  "
                    f"改动 {len(self.v.edited)}"
                )
            else:
                txt = f"悬停在模型或二维图上选择像素 · 改动 {len(self.v.edited)}"
            if self.v.erase_mode:
                txt += " · 橡皮模式(删除像素)"
            elif self.v.rotate_mode:
                txt += " · 旋转模式"
            self.status.config(text=txt)
            depth = len(self.v.undo_stack)
            self.undo_btn.config(
                text=("撤销(%d)" % depth) if depth else "撤销",
                state=("normal" if depth else "disabled"),
            )
            self._refresh_2d()
            self._draw_hl()
            self.root.update()
        except Exception:
            self.root = None

    def close(self):
        if self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass
            self.root = None


def _base_dir():
    """返回运行目录(脚本或打包后 exe 所在目录), 供定位状态文件(持久)。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _skins_dir():
    """返回皮肤目录: 优先 exe/脚本 旁的 skins, 否则用打包内置的 skins。"""
    base = _base_dir()
    d = os.path.join(base, "skins")
    if os.path.isdir(d):
        return d
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass and os.path.isdir(os.path.join(meipass, "skins")):
            return os.path.join(meipass, "skins")
    return base


def find_default_skin():
    # 优先恢复上次打开的状态(文件夹+文件名)
    state_path = os.path.join(_base_dir(), "skin_preview_state.json")
    if os.path.exists(state_path):
        try:
            import json

            with open(state_path, encoding="utf-8") as f:
                st = json.load(f)
            folder = st.get("folder")
            name = st.get("skin_name")
            if folder and name and os.path.isdir(folder):
                p = os.path.join(folder, name)
                if os.path.exists(p):
                    return p
        except Exception:
            pass
    folder = _skins_dir()
    if os.path.isdir(folder):
        for f in sorted(os.listdir(folder)):
            if f.lower() == "pic1.png":
                return os.path.join(folder, f)
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ) and not f.upper().startswith(ERROR_PREFIX):
                return os.path.join(folder, f)
    return None


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = find_default_skin()
    if not path or not os.path.exists(path):
        # 无默认皮肤也照常启动, 用户可在侧边栏点「导入文件夹」自行选择
        print("未找到默认皮肤, 程序以空界面启动, 可在侧边栏点「导入文件夹」选择")
        Viewer(None).run()
        return
    print(f"加载皮肤: {path}")
    Viewer(path).run()


if __name__ == "__main__":
    main()
