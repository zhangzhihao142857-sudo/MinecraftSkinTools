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
    # 右臂(角色右侧): slim(Alex)纤细手臂, 宽3; front/back/top/bottom 取3px
    "right_arm": dict(
        size=(3, 12, 4),
        pos=(-5.5, 18, 0),
        uv=dict(
            base=dict(
                top=_r(44, 16, 3, 4),
                bottom=_r(48, 16, 3, 4),
                right=_r(40, 20, 4, 12),
                front=_r(44, 20, 3, 12),
                left=_r(48, 20, 4, 12),
                back=_r(52, 20, 3, 12),
            ),
            overlay=dict(  # 袖子, 整体 +16y
                top=_r(44, 32, 3, 4),
                bottom=_r(48, 32, 3, 4),
                right=_r(40, 36, 4, 12),
                front=_r(44, 36, 3, 12),
                left=_r(48, 36, 4, 12),
                back=_r(52, 36, 3, 12),
            ),
        ),
    ),
    # 左臂(角色左侧): slim(Alex)纤细手臂, 宽3
    "left_arm": dict(
        size=(3, 12, 4),
        pos=(5.5, 18, 0),
        uv=dict(
            base=dict(
                top=_r(36, 48, 3, 4),
                bottom=_r(40, 48, 3, 4),
                right=_r(32, 52, 4, 12),
                front=_r(36, 52, 3, 12),
                left=_r(40, 52, 4, 12),
                back=_r(44, 52, 3, 12),
            ),
            overlay=dict(  # 袖子, 整体 +16x
                top=_r(52, 48, 3, 4),
                bottom=_r(56, 48, 3, 4),
                right=_r(48, 52, 4, 12),
                front=_r(52, 52, 3, 12),
                left=_r(56, 52, 4, 12),
                back=_r(60, 52, 3, 12),
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


def detect_arm_style(a):
    """检测皮肤手臂宽度: slim(Alex)=3px宽, 经典(Steve)=4px宽。
    依据手臂 front 面的有效宽度(从区域起点连续有像素的列数):
    恰好4列=经典; 3列或装饰性涂满(超宽)=slim。"""
    def eff_width(y0, y1, x0):
        rows = a[y0:y1, x0:x0 + 8].sum(axis=0)
        w = 0
        for c in rows:
            if c > 0:
                w += 1
            else:
                break
        return w
    rw = eff_width(20, 32, 44)   # 右臂 front
    lw = eff_width(52, 64, 36)   # 左臂 front
    w = max(rw, lw)
    return "classic" if w == 4 else "slim"


def build_parts(arm_style="slim"):
    """按手臂宽度类型返回 PARTS。经典(Steve)手臂宽4px, slim(Alex)宽3px。"""
    import copy

    parts = copy.deepcopy(PARTS)
    if arm_style == "classic":
        for side, arm in (("right", "right_arm"), ("left", "left_arm")):
            parts[arm]["size"] = (4, 12, 4)
            parts[arm]["pos"] = (-6.0 if side == "right" else 6.0, 18, 0)
        parts["right_arm"]["uv"]["base"].update(
            top=_r(44, 16, 4, 4), bottom=_r(48, 16, 4, 4),
            front=_r(44, 20, 4, 12), back=_r(52, 20, 4, 12))
        parts["right_arm"]["uv"]["overlay"].update(
            top=_r(44, 32, 4, 4), bottom=_r(48, 32, 4, 4),
            front=_r(44, 36, 4, 12), back=_r(52, 36, 4, 12))
        parts["left_arm"]["uv"]["base"].update(
            top=_r(36, 48, 4, 4), bottom=_r(40, 48, 4, 4),
            front=_r(36, 52, 4, 12), back=_r(44, 52, 4, 12))
        parts["left_arm"]["uv"]["overlay"].update(
            top=_r(52, 48, 4, 4), bottom=_r(56, 48, 4, 4),
            front=_r(52, 52, 4, 12), back=_r(60, 52, 4, 12))
    return parts


def build_geometry(layer, alpha=None, arm_style="slim"):
    """构建某一层(base 或 overlay)的三角形几何。
    返回 (vertices, texcoords, normals)。overlay 立方体比 base 略大一圈。
    alpha: 可选, 皮肤 alpha 布尔数组(64x64 布局)。参考 SkinLayers3D 的
    SolidPixelWrapper —— 只画皮肤上有非透明像素的面; 全透明的面跳过,
    从而消除镂空(透明面不再被画出, 避免透过模型看到内部)。"""
    verts, texs, norms = [], [], []
    # overlay 每维外扩 0.5(即每面外移 0.25), 形成略大的第二层
    thick = 0.5 if layer == "overlay" else 0.0

    def add_face(p, region, nx, ny, nz):
        w, h, d = p["size"]
        w, h, d = w + thick, h + thick, d + thick
        rx, ry, rw, rh = region
        # 面级 alpha 过滤: 该面在皮肤上全透明则跳过(不画), 避免镂空
        if (
            alpha is not None
            and 0 <= rx and 0 <= ry
            and rx + rw <= alpha.shape[1]
            and ry + rh <= alpha.shape[0]
            and not alpha[ry:ry + rh, rx:rx + rw].any()
        ):
            return
        u0, v0 = rx / SKIN_SIZE, ry / SKIN_SIZE
        u1, v1 = (rx + rw) / SKIN_SIZE, (ry + rh) / SKIN_SIZE
        # (64x64 新版皮肤左臂/左腿有独立UV区域, 直接取纹理, 不镜像
        #  legacyMirrorOf 仅用于旧版 64x32 皮肤, 此处无需处理)

        # 纹理 v 方向: v0 对应皮肤图像顶部(较小的 y), 赋给方块顶部顶点, 保证正立。
        if nx != 0:  # 左右面: 纹理宽=深度d, 高=高度h
            px_ = nx * w / 2
            quad = [
                (px_, -h / 2, -d / 2, u0, v1),
                (px_, -h / 2, d / 2, u1, v1),
                (px_, h / 2, d / 2, u1, v0),
                (px_, h / 2, -d / 2, u0, v0),
            ]
        elif nz != 0:  # 前后面: 纹理宽=宽度w, 高=高度h
            pz_ = nz * d / 2
            quad = [
                (-w / 2, -h / 2, pz_, u0, v1),
                (w / 2, -h / 2, pz_, u1, v1),
                (w / 2, h / 2, pz_, u1, v0),
                (-w / 2, h / 2, pz_, u0, v0),
            ]
        else:  # 顶底面: 纹理宽=宽度w, 高=深度d
            py_ = ny * h / 2
            quad = [
                (-w / 2, py_, -d / 2, u0, v1),
                (w / 2, py_, -d / 2, u1, v1),
                (w / 2, py_, d / 2, u1, v0),
                (-w / 2, py_, d / 2, u0, v0),
            ]

        cx, cy, cz = p["pos"]
        tri_order = (0, 1, 2, 0, 2, 3)
        for idx in tri_order:
            x, y, z, u, v = quad[idx]
            verts.append((x + cx, y + cy, z + cz))
            texs.append((u, v))
            norms.append((nx, ny, nz))

    parts = build_parts(arm_style)
    for name, p in parts.items():
        uv = p["uv"][layer]
        add_face(p, uv["right"], 1, 0, 0)
        add_face(p, uv["left"], -1, 0, 0)
        add_face(p, uv["front"], 0, 0, 1)
        add_face(p, uv["back"], 0, 0, -1)
        add_face(p, uv["top"], 0, 1, 0)
        add_face(p, uv["bottom"], 0, -1, 0)

    return (
        np.array(verts, np.float32),
        np.array(texs, np.float32),
        np.array(norms, np.float32),
    )


def read_skin_alpha(path):
    """读取皮肤 alpha 通道并归一化到 64x64 经典布局, 返回 np 布尔数组。
    高分辨率皮肤(128x128 等)按 64 布局放大, 用 NEAREST 缩回保留每格像素。"""
    img = Image.open(path).convert("RGBA")
    a = np.array(img)[:, :, 3]
    if a.shape[0] != 64 or a.shape[1] != 64:
        a = np.array(img.resize((64, 64), Image.NEAREST))[:, :, 3]
    return a > 0


def _extract_skin_tone(arr):
    """从头部前脸(base)提取该皮肤的肤色(肉色调), 兜底默认浅肤色。
    用于填充身体面透明空洞, 使手臂/腿看起来是皮肤色而非灰色。"""
    front = arr[8:16, 8:16]  # 头 front (64 布局)
    if arr.shape[0] != 64:
        s = arr.shape[0] / 64.0
        x0, y0, w, h = int(8 * s), int(8 * s), int(8 * s), int(8 * s)
        front = arr[y0:y0 + h, x0:x0 + w]
    a = front[:, :, 3]
    px = front[a > 0]
    tones = []
    for p in px:
        r, g, b = int(p[0]), int(p[1]), int(p[2])
        # 肉色启发: R>G>B, 偏暖, 非发色/眼白
        if r > g > b and r > 140 and (r - b) > 20 and abs(r - g) < 60:
            tones.append((r, g, b))
    if len(tones) >= 3:
        return tuple(int(x) for x in np.median(tones, axis=0))
    return (226, 208, 180)  # MC 默认浅肤色


def _fill_from_left(seg, fallback):
    """水平延拓: 每个透明像素取它左侧(UV -x)最近非透明像素的颜色填充。
    对右臂/左臂, 3D 空间的"右侧"(+x, right面)在 UV 上正是 -x 方向。"""
    a = seg[:, :, 3]
    h, w = seg.shape[0], seg.shape[1]
    for y in range(h):
        last = None
        for x in range(w):
            if a[y, x] > 0:
                last = seg[y, x, :3].copy()
            else:
                col = last if last is not None else fallback
                seg[y, x, :3] = col
                seg[y, x, 3] = 255
    return seg


def _fill_skin(seg, fallback):
    """直接用肤色填充透明处(非手臂部位)。"""
    a = seg[:, :, 3]
    m = a == 0
    seg[m, :3] = fallback
    seg[m, 3] = 255
    return seg


def _patch_body_faces(arr, layer="base", skin_tone=(226, 208, 180)):
    """补全身体层(base)面的部分透明空洞。
    手臂(right_arm/left_arm)按 3D 空间"右侧"(+x, right面)方向延拓填充:
    在 UV 上对应 -x, 即每个透明像素取左侧色块颜色, 使背面空洞与右侧色块连贯。
    其余部位直接用肤色填充。支持高分辨率皮肤: 坐标按 64 布局随实际尺寸缩放。"""
    sx = arr.shape[1] / 64.0
    sy = arr.shape[0] / 64.0
    for name, p in PARTS.items():
        for d, (rx, ry, rw, rh) in p["uv"][layer].items():
            X0 = int(rx * sx)
            Y0 = int(ry * sy)
            W = int(rw * sx)
            H = int(rh * sy)
            if X0 < 0 or Y0 < 0 or X0 + W > arr.shape[1] or Y0 + H > arr.shape[0]:
                continue
            seg = arr[Y0:Y0 + H, X0:X0 + W]
            a = seg[:, :, 3]
            n = int((a > 0).sum())
            if 0 < n < a.size:
                if name in ("right_arm", "left_arm"):
                    _fill_from_left(seg, skin_tone)   # 3D 右侧 = UV -x = 取左侧
                else:
                    _fill_skin(seg, skin_tone)
    return arr


def load_skin(path):
    """读取皮肤 png 并生成 OpenGL 纹理 id。
    Minecraft 皮肤语义: alpha==0 为透明, alpha>0 一律按不透明显示(半透明像素也显示)。
    这里把 alpha>0 的像素二值化为 255, 避免半透明像素被 alpha test 丢弃造成镂空。"""
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    # 补全 base 层身体面的部分透明空洞(用皮肤肤色填充), 消除手臂/腿缝隙镂空
    _patch_body_faces(arr, "base", _extract_skin_tone(arr))
    if (
        arr.shape[2] == 4
        and int((arr[:, :, 3] > 0).sum())
        and int((arr[:, :, 3] < 255).sum())
    ):
        arr[arr[:, :, 3] > 0, 3] = 255
        img = Image.fromarray(arr)
    tw, th = img.size
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    # 不要用 tobytes(...) 带 flip=-1(垂直翻转会导致 UV 错位); 按原样上传。
    data = img.tobytes("raw", "RGBA", 0, 0)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tw, th, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    return texture


class Viewer:
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
        ]
        # 搜索框: 输入关键字实时过滤当前文件夹皮肤
        self.search_active = False
        self.search_text = ""
        self.search_rect = (10, 22, 160, 30)
        self.search_results = []   # 过滤后的匹配文件列表
        self.search_scroll = 0     # 列表滚动偏移
        self._click_item = None    # 记录单击的列表项(用于双击检测)
        self._click_time = 0.0
        pygame.display.set_caption(
            os.path.basename(self.current_skin)
            if self.current_skin
            else "Minecraft 皮肤预览 - 点击右侧「导入文件夹」选择皮肤"
        )
        self._setup_gl()

        if self.current_skin:
            self.texture = load_skin(self.current_skin)
            self._skin_alpha = read_skin_alpha(self.current_skin)  # 面级过滤用
            self._arm_style = detect_arm_style(self._skin_alpha)  # slim/经典(宽体)
        else:
            self.texture = None
            self._skin_alpha = None
            self._arm_style = "slim"
        # 双层几何: base(第一层) + overlay(第二层, 外扩), 透明面按皮肤 alpha 跳过
        empty = np.zeros((64, 64), bool)
        self.base_geo = build_geometry(
            "base",
            self._skin_alpha if self._skin_alpha is not None else empty,
            self._arm_style,
        )
        self.overlay_geo = build_geometry(
            "overlay",
            self._skin_alpha if self._skin_alpha is not None else empty,
            self._arm_style,
        )

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

    def _scan_skins(self, folder):
        """扫描文件夹内的皮肤图片(png/jpg/jpeg)。"""
        if not folder or not os.path.isdir(folder):
            return []
        return [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def _update_search(self):
        """按搜索关键字过滤当前文件夹的皮肤列表(大小写不敏感)。"""
        kw = self.search_text.strip().lower()
        if kw:
            self.search_results = [
                f for f in self.skin_list
                if kw in os.path.basename(f).lower()
            ]
        else:
            self.search_results = list(self.skin_list)
        if self.search_scroll >= len(self.search_results):
            self.search_scroll = max(0, len(self.search_results) - 1)

    def _reload_skin(self, path):
        """重新加载指定皮肤图片(删除旧纹理, 创建新纹理)。
        加载失败时保留旧纹理并打印原因, 避免界面空白且无提示。"""
        if not path or not os.path.exists(path):
            print("皮肤不存在:", path)
            return
        try:
            if getattr(self, "texture", None) is not None:
                glDeleteTextures([self.texture])
            self.texture = load_skin(path)
            self._skin_alpha = read_skin_alpha(path)  # 面级过滤依赖具体皮肤
            self._arm_style = detect_arm_style(self._skin_alpha)  # 自动识别手臂宽/细
            self.base_geo = build_geometry("base", self._skin_alpha, self._arm_style)
            self.overlay_geo = build_geometry("overlay", self._skin_alpha, self._arm_style)
        except Exception as e:
            print(f"加载皮肤失败: {path} -> {e}")
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
        y0 = 262
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
                        t = data.decode("utf-8", "ignore") if isinstance(data, bytes) else data
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
        """保存当前文件夹+文件名+序号, 下次启动恢复。"""
        try:
            import json

            state = {
                "folder": self.folder,
                "skin_name": (
                    os.path.basename(self.current_skin) if self.current_skin else None
                ),
                "index": self.skin_index,
            }
            with open(self._state_path(), "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False)
        except Exception:
            pass

    @staticmethod
    def _state_path():
        return os.path.join(
            _base_dir(), "skin_preview_state.json"
        )

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
        glColor4f(0.30, 0.34, 0.44, 1.0) if not self.search_active else glColor4f(0.36, 0.42, 0.55, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(gx, sy); glVertex2f(gx + sw, sy)
        glVertex2f(gx + sw, sy + sh); glVertex2f(gx, sy + sh)
        glEnd()
        glColor4f(0.6, 0.7, 0.8, 1.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(gx, sy); glVertex2f(gx + sw, sy)
        glVertex2f(gx + sw, sy + sh); glVertex2f(gx, sy + sh)
        glEnd()
        stext = self.search_text
        if len(stext) > 17:
            stext = "..." + stext[-14:]
        if self.search_active:
            stext += "|"
        self._render_text_left(stext if stext else "(搜索文件名)", gx + 4, sy + 2, sw - 8, sh - 4)
        # 按钮(坐标加 x0 偏移)
        for label, (rx, y, w, h) in self.buttons:
            x = self.x0 + rx
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
        self._render_text_left(f"图片 {self.skin_index + 1}/{n}  共{n}张", self.x0 + 10, 196, 160, 18)
        cname = os.path.basename(self.current_skin) if self.current_skin else "无"
        if len(cname) > 16:
            cname = cname[:13] + "..."
        self._render_text_left(f"当前: {cname}", self.x0 + 10, 216, 160, 18)
        # 搜索结果列表
        self._render_text_left(f"搜索结果 ({len(self.search_results)})", self.x0 + 10, 238, 160, 16)
        self._draw_search_list()
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def _draw_search_list(self):
        """绘制搜索结果列表(滚动), 每项为一行文件名。"""
        x = self.x0 + 10
        w = 160
        y0 = 262
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
                glVertex2f(x, yy); glVertex2f(x + w, yy)
                glVertex2f(x + w, yy + ih); glVertex2f(x, yy + ih)
                glEnd()
            # 项分隔线
            glColor4f(0.24, 0.28, 0.36, 1.0)
            glBegin(GL_LINES)
            glVertex2f(x, yy + ih); glVertex2f(x + w, yy + ih)
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
        if self.texture is None or len(verts) == 0:
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
                self.texture = load_skin(self.current_skin)  # 重建纹理
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
        self._render_text_wrap("未找到皮肤\n点击右侧「导入文件夹」选择皮肤文件夹", cx - 150, cy - 40, 300, 26)
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
        # 侧边栏正交绘制: 关闭 alpha 剔除, 避免影响文字纹理的半透明边缘
        glDisable(GL_ALPHA_TEST)
        self._draw_buttons()
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            self.clock.tick(60)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.VIDEORESIZE:
                    self._resize(ev.w, ev.h)
                elif ev.type == pygame.DROPFILE:
                    self._handle_drop(ev.file)
                elif ev.type == pygame.KEYDOWN:
                    if self.search_active:
                        if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_ESCAPE):
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
                        running = False
                    elif ev.key == pygame.K_w:
                        self.wireframe = not self.wireframe
                    elif ev.key == pygame.K_r:
                        self.yaw, self.pitch, self.dist = -40.0, -10.0, 68.0
                        self.pan = [0.0, 0.0]
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    if ev.button == 1:
                        pos = pygame.mouse.get_pos()
                        if self._hit_search(pos):
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
                                else:
                                    self._dragging = "rot"
                                    self._last = pos
                    elif ev.button == 3:
                        self._dragging = "pan"
                        self._last = pygame.mouse.get_pos()
                    elif ev.button == 4:
                        if self._mouse_in_list():
                            self.search_scroll = max(0, self.search_scroll - 1)
                        else:
                            self.dist = max(10.0, self.dist * 0.92)
                    elif ev.button == 5:
                        if self._mouse_in_list():
                            self.search_scroll = min(len(self.search_results) - 1, self.search_scroll + 1)
                        else:
                            self.dist = min(300.0, self.dist * 1.08)
                elif ev.type == pygame.MOUSEBUTTONUP:
                    self._dragging = None
                elif ev.type == pygame.MOUSEMOTION and self._dragging:
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
            self._render()
        pygame.quit()


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
    state_path = os.path.join(
        _base_dir(), "skin_preview_state.json"
    )
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
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
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
