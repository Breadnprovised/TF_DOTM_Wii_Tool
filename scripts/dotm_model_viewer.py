#!/usr/bin/env python3
"""
dotm_model_viewer.py - model viewer for Transformers: Dark of the Moon (Wii)

Indexes every .wii load unit in the game folder and shows every model
textured in a 3D view. Textures are resolved across the whole folder, so a
model whose textures are stored in another file still shows them.

    python3 dotm_model_viewer.py [GAME_FOLDER]

Without a folder argument a folder picker opens (the last folder is
remembered). Requires Python 3.8+, numpy, Pillow and pyglet 2.x
(pip install numpy pillow "pyglet>=2.1"), with the other scripts of this
folder next to it.

Controls
  Up / Down, wheel over list   previous / next model
  PgUp / PgDn, Home / End      jump through the list
  click a row                  select it
  / or Ctrl+F, or click the search box   search the list (Esc clears)
  drag the scroll bar / panel edge       scroll the list / resize the panel
  left drag                    orbit        right/middle drag   pan
  wheel over view              zoom         F                   frame model
  Space                        auto-rotate
  T textures   L lighting   V vertex colors   W wireframe   S skeleton
  C backface culling   B background color
  D hide duplicate models (on by default)   N hide empty models (on by default)
  A animations ([ and ] step one frame)   I piece inspector   O export settings
  E / Ctrl+E / Export button   export current model (.glb with skeleton + PNG textures)
  F12 screenshot (screenshots folder)   H or the Help button: key list
"""

import json
import math
import os
import queue
import re
import sys
import threading
import time
import ctypes
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
# exports/screenshots go next to the launchers when the scripts live in "scripts"
BASE = os.path.dirname(HERE) if os.path.basename(HERE).lower() == "scripts" else HERE
sys.path.insert(0, HERE)

import numpy as np
from PIL import Image

try:
    import pyglet
    from pyglet import gl
    from pyglet.graphics.shader import Shader, ShaderProgram
    from pyglet.window import key, mouse
except ImportError:
    sys.exit('pyglet 2.x is required:  pip install "pyglet>=2.1"')

from gameindex import GameIndex, find_wii_files
from lufile import GX_FORMATS
from gxtex import decode as gx_decode
from model import Model
from export import export_model, safe
import anim as anim_mod

CONFIG = os.path.join(os.path.expanduser("~"), ".dotm_model_viewer.json")
PANEL_W = 330          # default side-panel width (resizable, remembered)
PANEL_MIN, VIEW_MIN = 200, 260
SB_W = 10              # scrollbar width
SEARCH_H = 26
ROW_H = 20
FONT = 10
BGS = [(0.16, 0.17, 0.19), (0.45, 0.47, 0.5), (0.02, 0.02, 0.02), (0.85, 0.86, 0.88)]

VERT = """#version 330 core
in vec3 position;
in vec3 normal;
in vec2 uv;
in vec4 color;
uniform mat4 mvp;
uniform mat4 bone;          // the piece's joint: bind pose -> animated pose
out vec3 v_pos;
out vec3 v_nrm;
out vec2 v_uv;
out vec4 v_col;
void main() {
    vec4 p = bone * vec4(position, 1.0);
    v_pos = p.xyz;
    v_nrm = mat3(bone) * normal;
    v_uv = uv;
    v_col = color;
    gl_Position = mvp * p;
}
"""

FRAG = """#version 330 core
in vec3 v_pos;
in vec3 v_nrm;
in vec2 v_uv;
in vec4 v_col;
uniform sampler2D diffuse_tex;
uniform int has_tex;
uniform int use_tex;
uniform int use_light;
uniform int use_col;
uniform vec3 eye;
uniform vec3 tint;
out vec4 frag;
void main() {
    vec4 base = vec4(tint, 1.0);
    if (use_tex == 1 && has_tex == 1) {
        base = texture(diffuse_tex, v_uv);
        if (base.a < 0.35) discard;
    }
    if (use_col == 1) base.rgb *= v_col.rgb;
    vec3 n;
    if (length(v_nrm) > 0.01) n = normalize(v_nrm);
    else n = normalize(cross(dFdx(v_pos), dFdy(v_pos)));
    vec3 v = normalize(eye - v_pos);
    if (dot(n, v) < 0.0) n = -n;                    // two-sided lighting
    vec3 rgb = base.rgb;
    if (use_light == 1) {
        vec3 l1 = normalize(v + vec3(0.3, 0.6, 0.2));
        vec3 l2 = normalize(vec3(-0.5, 0.3, -0.6));
        float d = 0.38 + 0.62 * max(dot(n, l1), 0.0) + 0.18 * max(dot(n, l2), 0.0);
        float spec = pow(max(dot(n, normalize(l1 + v)), 0.0), 32.0) * 0.18;
        rgb = rgb * d + spec;
    }
    frag = vec4(rgb, 1.0);
}
"""

LINE_VERT = """#version 330 core
in vec3 position;
uniform mat4 mvp;
void main() { gl_Position = mvp * vec4(position, 1.0); gl_PointSize = 7.0; }
"""

LINE_FRAG = """#version 330 core
uniform vec4 line_color;
out vec4 frag;
void main() { frag = line_color; }
"""


def output_dir(kind):
    """Folder for exports / screenshots: next to the launchers if writable,
    otherwise in the user's home folder."""
    for base in (BASE, os.path.join(os.path.expanduser("~"), "DOTM_Wii_Viewer")):
        d = os.path.join(base, kind)
        try:
            os.makedirs(d, exist_ok=True)
            probe = os.path.join(d, ".write_test")
            open(probe, "w").close()
            os.remove(probe)
            return d
        except OSError:
            continue
    raise OSError(f"no writable folder for {kind}")


# ----------------------------------------------------------------------------
# small matrix helpers (row-major numpy; transposed when uploaded)
# ----------------------------------------------------------------------------
def perspective(fov, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov) / 2)
    m = np.zeros((4, 4), np.float32)
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def look_at(eye, target, up):
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= max(np.linalg.norm(s), 1e-9)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def gl_mat(m):
    return tuple(np.asarray(m, np.float32).T.flatten().tolist())


def _set_label(lbl, text, color=None):
    """Change a label only if needed: every text change re-lays it out,
    which is the slow part of redrawing the model list."""
    if lbl.text != text:
        lbl.text = text
    if color is not None and tuple(lbl.color) != color:
        lbl.color = color


def _n(count, word, plural=None):
    return f"{count:,} {word if count == 1 else (plural or word + 's')}"


def tex_key(t):
    return (t.lu.path, t.off)


def read_rgba(t):
    """Decode a texture object (level 0) to an HxWx4 uint8 array."""
    tx = t.tex
    data = t.lu.read(t.off + tx["data_off"], t.size - tx["data_off"])
    return gx_decode(data, tx["fmt"], tx["w"], tx["h"])


# ----------------------------------------------------------------------------
# Catalogue of everything in the game folder
# ----------------------------------------------------------------------------
class ModelEntry:
    __slots__ = ("obj", "rel", "name")

    def __init__(self, obj, rel):
        self.obj, self.rel, self.name = obj, rel, obj.name or f"model {obj.id:08x}"


class Catalogue:
    def __init__(self, root):
        self.index = GameIndex(root)
        self.models = []        # [ModelEntry]
        self.lock = threading.Lock()
        self.files_done = 0
        self.files_total = 0
        self.skipped = []       # (rel path, error)
        self.done = False
        self.stop = False
        self.anim_objs = []     # type-79 objects
        self.anims = []         # parsed anim_mod.Anim (after the scan)
        self.anims_done = False
        self.dups = {}          # model index -> index of the identical copy that is kept
        self.empty = set()      # model indices with nothing to draw (no triangles)
        self.dups_done = False  # duplicate / empty check finished

    @property
    def texture_count(self):
        with self.lock:
            return sum(len(lu.textures) for lu in self.index.lus)

    def surface_texture(self, obj, surface):
        with self.lock:
            return self.index.surface_texture(obj, surface)

    def scan(self, root):
        files = find_wii_files(root)
        self.files_total = len(files)
        for path in files:
            if self.stop:
                break
            rel = os.path.relpath(path, root)
            lu, err = GameIndex.open_file(path)
            if err:
                print(f"skipped {rel}: {err}", file=sys.stderr)
                self.skipped.append((rel, err))
            else:
                where = os.path.splitext(rel)[0].replace("\\", "/")
                with self.lock:
                    self.index.register(lu)
                    self.models.extend(ModelEntry(o, where) for o in lu.models)
                    self.anim_objs.extend(o for o in lu.objects if o.type == anim_mod.T_ANIM)
            self.files_done += 1
        self.done = True
        parsed = []
        for o in self.anim_objs:
            if self.stop:
                break
            try:
                parsed.append(anim_mod.Anim(o))
            except Exception as e:          # noqa: BLE001
                print(f"skipped animation {o.name}: {e}", file=sys.stderr)
        with self.lock:
            self.anims = parsed
            self.model_names = sorted({m.name for m in self.models})
        self.anims_done = True
        if not self.stop:
            try:
                self.dups = self.find_duplicates()
            except Exception as e:
                print(f"warning: duplicate check failed: {e}", file=sys.stderr)
                self.dups = {}
        self.dups_done = True

    def find_duplicates(self):
        """Models that are byte-for-byte copies of another model (ignoring the
        object ID) and resolve to identical textures (also ignoring their
        object IDs). The copy kept is the one in the model's own file (see
        own_file), otherwise the one with the shortest path. Also
        fills self.empty with the models that have no triangles."""
        with self.lock:
            models = list(self.models)
        groups = {}
        empty = set()
        for i, e in enumerate(models):
            if self.stop:
                return {}
            try:
                d = e.obj.read()
                if Model(d, e.obj.name).triangle_count == 0:
                    empty.add(i)        # e.g. shadow / spawner / camera helpers
            except Exception:
                continue
            # the object ID (+0x0C) differs between copies stored in different files
            groups.setdefault((len(d), zlib.crc32(d[0x10:])), []).append(i)
        self.empty = empty
        tex_sig = {}

        def sig(i):
            obj = models[i].obj
            out = []
            for s in Model(obj.read(), obj.name).surfaces:
                t = self.surface_texture(obj, s)
                if t is None:
                    out.append(None)
                    continue
                k = tex_key(t)
                if k not in tex_sig:
                    # whole texture object apart from its object ID (+0x0C), which
                    # differs between copies stored in different files
                    tex_sig[k] = (t.tex["w"], t.tex["h"], t.tex["fmt"], zlib.crc32(t.read()[0x10:]))
                out.append(tex_sig[k])
            return tuple(out)

        def own_file(e):
            """The model's own load unit: the game names files after their
            character in 8 letters (a_bumblebee_ref -> abumbleb, d_megatron_ref
            -> dmegatro)."""
            mn = re.sub(r"[^a-z0-9]", "", re.sub(r"_ref$", "", e.name.lower()))
            lu = re.sub(r"[^a-z0-9]", "", e.obj.lu.name.lower().split(":")[0])
            return len(lu) >= 4 and mn.startswith(lu)

        dups = {}
        for members in groups.values():
            if len(members) < 2 or self.stop:
                continue
            by_sig = {}
            for i in members:
                try:
                    by_sig.setdefault(sig(i), []).append(i)
                except Exception:
                    continue
            for same in by_sig.values():
                if len(same) < 2:
                    continue
                same.sort(key=lambda i: (not own_file(models[i]), len(models[i].rel), models[i].rel.lower()))
                for i in same[1:]:
                    dups[i] = same[0]
        return dups


# ----------------------------------------------------------------------------
# GPU resources for one model
# ----------------------------------------------------------------------------
class GpuModel:
    def __init__(self, program, lprogram, entry, model, surf_tex, tex_arrays):
        """surf_tex: texture object (or None) per surface; tex_arrays: tex key ->
        HxWx4 uint8 numpy (decoded in a worker)."""
        self.entry = entry
        self.model = model
        self.surf_tex = surf_tex
        self.parts = []
        self.textures = {}
        self.arrays = tex_arrays          # kept for the piece panel thumbnails
        for k, arr in tex_arrays.items():
            self.textures[k] = upload_texture(arr)
        for s, t in zip(model.surfaces, surf_tex):
            n = len(s.pos)
            idx = s.tris.ravel().tolist()
            if not idx:
                continue
            vl = program.vertex_list_indexed(
                n, gl.GL_TRIANGLES, idx,
                position=("f", s.pos.ravel().tolist()),
                normal=("f", s.nrm.ravel().tolist()),
                uv=("f", s.uv.ravel().tolist()),
                color=("f", (s.col.astype(np.float32) / 255.0).ravel().tolist()))
            tid = self.textures.get(tex_key(t)) if t is not None and s.has_uv else None
            self.parts.append((vl, s, tid))
        # skeleton: a line from every joint to its parent, plus a point per joint
        self.bones = self.joints = None
        self.bind_world = np.asarray(model.world, np.float64).reshape(-1, 4, 4) if len(model.joints) else np.zeros((0, 4, 4))
        self.inv_bind = np.linalg.inv(self.bind_world) if len(self.bind_world) else self.bind_world
        J = model.joint_positions().astype(np.float32)
        self.bone_pairs = [(i, p) for i, p in enumerate(model.parents) if 0 <= p < len(J) and p != i]
        if len(J):
            if self.bone_pairs:
                L = np.array([np.r_[J[i], J[p]] for i, p in self.bone_pairs], np.float32).reshape(-1, 3)
                self.bones = lprogram.vertex_list(len(L), gl.GL_LINES, position=("f", L.ravel().tolist()))
            self.joints = lprogram.vertex_list(len(J), gl.GL_POINTS, position=("f", J.ravel().tolist()))
        self.posed = False
        lo, hi = model.bounds()
        if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
            lo, hi = np.zeros(3), np.ones(3)
        self.center = ((lo + hi) / 2).astype(np.float32)
        self.radius = max(float(np.linalg.norm(hi - lo)) / 2, 1e-3)

    def set_skeleton(self, world):
        """Move the skeleton overlay to model-space joint matrices (None: bind pose)."""
        if self.joints is None:
            return
        W = self.bind_world if world is None else world
        J = W[:, 3, :3].astype(np.float32)
        self.joints.position[:] = J.ravel().tolist()
        if self.bones is not None and self.bone_pairs:
            L = np.array([np.r_[J[i], J[p]] for i, p in self.bone_pairs], np.float32)
            self.bones.position[:] = L.ravel().tolist()
        self.posed = world is not None

    def delete(self):
        for vl, *_ in self.parts:
            vl.delete()
        for vl in (self.bones, self.joints):
            if vl is not None:
                vl.delete()
        ids = [t for t in self.textures.values() if t]
        if ids:
            arr = (gl.GLuint * len(ids))(*ids)
            gl.glDeleteTextures(len(ids), arr)
        self.parts = []
        self.textures = {}
        self.bones = self.joints = None


def upload_texture(arr):
    h, w = arr.shape[:2]
    tid = gl.GLuint()
    gl.glGenTextures(1, ctypes.byref(tid))
    gl.glBindTexture(gl.GL_TEXTURE_2D, tid)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    # the arrays are upright images; the game's v = 0 is the bottom row
    # (OpenGL convention), so upload bottom row first
    data = np.ascontiguousarray(arr[::-1], np.uint8)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, w, h, 0, gl.GL_RGBA,
                    gl.GL_UNSIGNED_BYTE, data.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
    gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
    return tid.value


# ----------------------------------------------------------------------------
# Piece inspector (right-hand panel)
# ----------------------------------------------------------------------------
def _thumb_image(arr, size):
    """numpy RGBA -> pyglet image, fitted into size x size over a checkerboard."""
    im = Image.fromarray(np.ascontiguousarray(arr), "RGBA")
    im.thumbnail((size, size), Image.LANCZOS if max(im.size) > size else Image.NEAREST)
    if max(im.size) < size:                       # small textures: scale up crisply
        k = size // max(im.size)
        im = im.resize((im.width * k, im.height * k), Image.NEAREST)
    bg = Image.new("RGBA", im.size, (150, 150, 150, 255))
    sq = max(4, size // 8)
    for yy in range(0, im.height, sq):
        for xx in range(0, im.width, sq):
            if (xx // sq + yy // sq) % 2:
                bg.paste((200, 200, 200, 255), (xx, yy, min(xx + sq, im.width), min(yy + sq, im.height)))
    bg.alpha_composite(im)
    return pyglet.image.ImageData(bg.width, bg.height, "RGBA", bg.tobytes(), pitch=-bg.width * 4)


class PiecePanel:
    W = 410
    THUMB = 44
    LINE = 16

    def __init__(self, win):
        self.win = win
        self.visible = False
        self.gpu = None
        self.highlight = None        # piece index shown alone in the view
        self.scroll = 0
        self.batch = pyglet.graphics.Batch()
        self.g0, self.g1, self.g2 = (pyglet.graphics.Group(order=i) for i in range(3))
        self.objs = []
        self.hits = []               # (x, y, w, h, action, arg)
        self.thumbs = {}             # (tex key, size) -> pyglet image
        self.preview = None          # texture key shown large
        self.pbatch = pyglet.graphics.Batch()
        self.pobjs = []
        self.sb_drag = None          # grab offset inside the thumb while dragging the scroll bar
        self._sb = None              # (top, avail, total, thumb_h, thumb_y) of the scroll bar

    # ------------------------------------------------------------------
    def toggle(self):
        self.visible = not self.visible
        if self.visible:
            self.win.close_panels(self)
        if not self.visible:
            self.highlight = None
            self.preview = None
        self.rebuild()
        self.win.flash("Piece inspector " + ("open - click a piece to show only its faces" if self.visible else "closed"))

    def set_model(self, gpu):
        self.gpu = gpu
        self.highlight = None
        self.preview = None
        self.scroll = 0
        self.thumbs.clear()
        self.rebuild()

    def on_escape(self):
        if self.preview is not None:
            self.preview = None
            self.rebuild()
            return True
        if self.highlight is not None:
            self.highlight = None
            self.rebuild()
            return True
        return False

    def layout(self):
        self.rebuild()

    @property
    def rect(self):
        w = self.win
        x0 = max(w.panel_w + 260, w.width - self.W)
        return x0, 28, w.width - x0, w.height - 46 - 28

    # ------------------------------------------------------------------
    def _surface_lines(self, s):
        head = f"piece {s.index}  ·  joint {s.joint}"
        sub = f"{len(s.tris):,} tris · {len(s.pos):,} verts"
        rows = [f"material @ 0x{s.material:x}"]
        if not (s.has_uv and s.has_nrm):
            rows.append("vertex: position" + (", normal" if s.has_nrm else "") + ", colour"
                        + ("" if s.has_uv else " (no UVs)"))
        if s.texrefs:
            rows.append("texture refs  " + "  ".join(f"{r:08x}" for r in s.texrefs))
        else:
            rows.append("no texture reference")
        return head, sub, rows

    def _card_height(self, rows):
        return 8 + 20 + len(rows) * self.LINE + 6 + self.THUMB + 6 + 8

    def rebuild(self):
        for o in self.objs + self.pobjs:
            o.delete()
        self.objs, self.pobjs, self.hits = [], [], []
        if not self.visible:
            return
        B, G0, G1, G2 = self.batch, self.g0, self.g1, self.g2
        x0, y0, pw, ph = self.rect
        top = y0 + ph
        add = self.objs.append
        add(pyglet.shapes.Rectangle(x0, y0, pw, ph, color=(22, 24, 28, 238), batch=B, group=G0))
        add(pyglet.shapes.Rectangle(x0, y0, 2, ph, color=(60, 64, 72, 255), batch=B, group=G1))
        gpu = self.gpu
        if gpu is None or not gpu.model.surfaces:
            add(pyglet.text.Label("No pieces" if gpu else "No model loaded", x=x0 + 14, y=top - 24,
                                  font_size=FONT, color=(190, 190, 190, 255), batch=B, group=G2))
            return
        m = gpu.model
        obj = gpu.entry.obj
        cards = []
        for s, t in zip(m.surfaces, gpu.surf_tex):
            head, sub, rows = self._surface_lines(s)
            cards.append((self._card_height(rows), s, t, head, sub, rows))
        total = sum(c[0] for c in cards) + 30
        self.scroll = max(0, min(self.scroll, max(0, total - ph)))
        cw = pw - 16 - (12 if total > ph else 0)     # card width (room for the scroll bar)
        ntex = len({tex_key(t) for t in gpu.surf_tex if t is not None})
        # title
        y = top + self.scroll - 6
        add(pyglet.text.Label(f"Pieces ({len(cards)})  ·  {ntex} texture{'s' if ntex != 1 else ''}",
                              x=x0 + 14, y=y - 14, font_size=FONT + 1, weight="bold",
                              color=(235, 235, 235, 255), batch=B, group=G2))
        y -= 26
        chars = int((pw - 30) / self.win.char_w)
        for ch, s, t, head, sub, rows in cards:
            ctop, cbot = y, y - ch
            y = cbot
            if cbot > top or ctop < y0:       # off screen
                continue
            sel = self.highlight == s.index
            add(pyglet.shapes.Rectangle(x0 + 8, cbot + 3, cw, ch - 6,
                                        color=(48, 70, 104, 255) if sel else (34, 37, 43, 255), batch=B, group=G0))
            self.hits.append((x0 + 8, cbot + 3, cw, ch - 6, "card", s.index))
            cy = ctop - 22
            add(pyglet.text.Label(head, x=x0 + 18, y=cy, font_size=FONT, weight="bold",
                                  color=(255, 255, 255, 255), batch=B, group=G2))
            add(pyglet.text.Label(sub, x=x0 + 8 + cw - 10, y=cy, font_size=FONT - 1, anchor_x="right",
                                  color=(170, 180, 195, 255), batch=B, group=G2))
            cy -= 4
            for r in rows:
                cy -= self.LINE
                add(pyglet.text.Label(r[:chars], x=x0 + 18, y=cy, font_size=FONT - 1,
                                      color=(205, 205, 210, 255), batch=B, group=G2))
            # texture slot
            cy -= 6 + self.THUMB + 6
            tx_ = x0 + 18
            k = tex_key(t) if t is not None else None
            arr = gpu.arrays.get(k) if k is not None else None
            if arr is not None:
                key_ = (k, self.THUMB)
                if key_ not in self.thumbs:
                    self.thumbs[key_] = _thumb_image(arr, self.THUMB)
                img = self.thumbs[key_]
                add(pyglet.sprite.Sprite(img, x=tx_ + (self.THUMB - img.width) // 2,
                                         y=cy + (self.THUMB - img.height) // 2, batch=B, group=G2))
                self.hits.append((tx_, cy, self.THUMB, self.THUMB, "thumb", k))
            else:
                add(pyglet.shapes.Rectangle(tx_, cy, self.THUMB, self.THUMB, color=(55, 58, 64, 255), batch=B, group=G1))
                add(pyglet.text.Label("—" if not s.texrefs else "?", x=tx_ + self.THUMB // 2, y=cy + self.THUMB // 2,
                                      anchor_x="center", anchor_y="center", font_size=FONT + 2,
                                      color=(150, 150, 155, 255), batch=B, group=G2))
            if t is not None:
                tx = t.tex
                line1 = f"diffuse · {t.name}"
                line2 = f"{tx['w']}×{tx['h']} {GX_FORMATS[tx['fmt']]} · " + \
                        ("this file" if t.lu is obj.lu else f"from {t.lu.name}")
            elif s.texrefs:
                line1 = f"diffuse · {s.texrefs[0]:08x}"
                line2 = "not in game folder"
            else:
                line1, line2 = "diffuse", "untextured"
            lc = int((pw - self.THUMB - 46) / self.win.char_w)
            add(pyglet.text.Label(line1[:lc], x=tx_ + self.THUMB + 10, y=cy + self.THUMB - 16,
                                  font_size=FONT - 1, color=(235, 235, 235, 255), batch=B, group=G2))
            add(pyglet.text.Label(line2[:lc], x=tx_ + self.THUMB + 10, y=cy + 6, font_size=FONT - 1,
                                  color=(165, 170, 180, 255) if t is not None or not s.texrefs else (220, 150, 120, 255),
                                  batch=B, group=G2))
        # scroll bar: drag the thumb, or click the track to jump there
        self._sb = None
        if total > ph:
            th = max(30, int(ph * ph / total))
            ty = top - th - int((ph - th) * self.scroll / max(1, total - ph))
            sx = x0 + pw - 12
            add(pyglet.shapes.Rectangle(sx, y0, 8, ph, color=(36, 39, 45, 255), batch=B, group=G1))
            col = (150, 156, 168, 255) if self.sb_drag is not None else (110, 115, 125, 255)
            add(pyglet.shapes.Rectangle(sx + 1, ty, 6, th, color=col, batch=B, group=G2))
            self._sb = (top, ph, total, th, ty)
            self.hits.append((sx - 4, y0, 16, ph, "sbar", None))   # checked first (hits run in reverse)
        self._build_preview()

    def _build_preview(self):
        k = self.preview
        if k is None or self.gpu is None or k not in self.gpu.arrays:
            return
        w = self.win
        x0, y0, pw, ph = self.rect
        area_l, area_r = w.panel_w, x0
        size = int(min(512, area_r - area_l - 60, w.height - 140))
        if size < 64:
            return
        arr = self.gpu.arrays[k]
        key_ = (k, size)
        if key_ not in self.thumbs:
            self.thumbs[key_] = _thumb_image(arr, size)
        img = self.thumbs[key_]
        cx, cy = (area_l + area_r) // 2, w.height // 2
        B = self.pbatch
        self.pobjs.append(pyglet.shapes.Rectangle(area_l, 0, area_r - area_l, w.height, color=(0, 0, 0, 170), batch=B, group=self.g0))
        self.pobjs.append(pyglet.shapes.Rectangle(cx - img.width // 2 - 8, cy - img.height // 2 - 50,
                                                  img.width + 16, img.height + 66, color=(30, 32, 37, 255), batch=B, group=self.g1))
        self.pobjs.append(pyglet.sprite.Sprite(img, x=cx - img.width // 2, y=cy - img.height // 2, batch=B, group=self.g2))
        t = next((t for t in self.gpu.surf_tex if t is not None and tex_key(t) == k), None)
        cap = ""
        if t is not None:
            tx = t.tex
            cap = (f"{t.name}  ·  {tx['w']}×{tx['h']} {GX_FORMATS[tx['fmt']]}  ·  "
                   f"{w.cat.index.rel(t.lu)}")
        self.pobjs.append(pyglet.text.Label(cap, x=cx, y=cy - img.height // 2 - 20,
                                            anchor_x="center", font_size=FONT, color=(230, 230, 230, 255),
                                            batch=B, group=self.g2))
        self.pobjs.append(pyglet.text.Label("click another thumbnail to switch  ·  click elsewhere or Esc to close",
                                            x=cx, y=cy - img.height // 2 - 38, anchor_x="center", font_size=FONT - 1,
                                            color=(160, 165, 175, 255), batch=B, group=self.g2))

    # ------------------------------------------------------------------
    def draw(self, scale):
        if not self.visible:
            return
        x0, y0, pw, ph = self.rect
        if self.preview is not None:
            self.pbatch.draw()
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(int(x0 * scale), int(y0 * scale), int(pw * scale) + 1, int(ph * scale) + 1)
        self.batch.draw()
        gl.glDisable(gl.GL_SCISSOR_TEST)

    def on_press(self, x, y, button):
        if not self.visible:
            return False
        if self.preview is not None:
            # another thumbnail -> show that one; the same one or anywhere else -> close
            hit = next((arg for hx, hy, hw, hh, act, arg in self.hits
                        if act == "thumb" and hx <= x <= hx + hw and hy <= y <= hy + hh), None)
            self.preview = hit if (hit is not None and hit != self.preview) else None
            self.rebuild()
            return True
        x0, y0, pw, ph = self.rect
        if not (x0 <= x <= x0 + pw and y0 <= y <= y0 + ph):
            return False
        if button != mouse.LEFT:
            return True
        for hx, hy, hw, hh, act, arg in reversed(self.hits):   # thumbs before cards
            if hx <= x <= hx + hw and hy <= y <= hy + hh:
                if act == "sbar":
                    _top, _ph, _tot, th_, ty_ = self._sb
                    # grab the thumb where it was clicked; a click on the track centres it there
                    self.sb_drag = (y - ty_) if ty_ <= y <= ty_ + th_ else th_ / 2
                    self._scroll_to_thumb(y)
                    return True
                if act == "thumb":
                    self.preview = arg
                elif act == "card":
                    self.highlight = None if self.highlight == arg else arg
                    if self.highlight is not None:
                        s = next(s for s in self.gpu.model.surfaces if s.index == arg)
                        self.win.flash(f"Showing only piece {arg}: {len(s.tris):,} triangles "
                                       f"(click again or Esc to show all)")
                self.rebuild()
                return True
        return True

    def _scroll_to_thumb(self, y):
        """Scroll so the thumb's grabbed point sits at mouse height y."""
        top, avail, total, th_, _ty = self._sb
        travel = max(1, avail - th_)
        frac = (top - th_ - (y - self.sb_drag)) / travel
        self.scroll = int(round(max(0.0, min(1.0, frac)) * (total - avail)))
        self.rebuild()

    def on_drag(self, x, y):
        if self.sb_drag is not None and self._sb:
            self._scroll_to_thumb(y)
            return True
        return False

    def on_release(self):
        if self.sb_drag is not None:
            self.sb_drag = None
            self.rebuild()
            return True
        return False

    def on_scroll(self, x, y, sy):
        if not self.visible:
            return False
        x0, y0, pw, ph = self.rect
        if not (x0 <= x <= x0 + pw and y0 <= y <= y0 + ph):
            return False
        self.scroll = max(0, self.scroll - int(sy * 40))
        self.rebuild()
        return True


class AnimPanel:
    """Side panel listing the animations that fit the current model, with
    playback controls. The pose is applied on the GPU (rigid skinning: every
    piece follows its joint)."""
    W = 410
    ROW = 20
    CTRL_H = 112
    FPS = 30

    def __init__(self, win):
        self.win = win
        self.visible = False
        self.gpu = None
        self.items = []              # [(anim_mod.Anim, score)]
        self.cur = None              # index into items
        self.time = 0.0              # seconds
        self.playing = False
        self.loop = True
        self.speed = 1.0
        self.scroll = 0
        self.batch = pyglet.graphics.Batch()
        self.g0, self.g1, self.g2 = (pyglet.graphics.Group(order=i) for i in range(3))
        self.objs, self.hits = [], []
        self.dyn = []                # timeline objects updated every frame
        self.dragging = False
        self.sb_drag = None          # grab offset inside the thumb while dragging the scroll bar
        self._sb = None              # (list_top, avail, total, thumb_h, thumb_y) of the list scroll bar
        self._pose_key = None
        self._pose = None

    @property
    def rect(self):
        w = self.win
        x0 = max(w.panel_w + 260, w.width - self.W)
        return x0, 28, w.width - x0, w.height - 46 - 28

    # ------------------------------------------------------------------
    def toggle(self):
        self.visible = not self.visible
        if self.visible:
            self.win.close_panels(self)
        self.rebuild()
        if self.visible:
            n = len(self.items)
            if not self.win.cat.anims_done:
                self.win.flash("Animations: still scanning the folder...")
            else:
                self.win.flash(f"Animations: {n} fit this model - click one to play" if n else
                               "Animations: none found for this model in the scanned folder")
        else:
            self.win.flash("Animations closed (model shown in its bind pose)")

    def set_model(self, gpu):
        self.gpu = gpu
        self.cur = None
        self.playing = False
        self.time = 0.0
        self.scroll = 0
        self.items = []
        self._pose_key = None
        if gpu is not None and len(gpu.model.joints) and self.win.cat.anims_done:
            with self.win.cat.lock:
                anims, names = list(self.win.cat.anims), list(self.win.cat.model_names)
            self.items = anim_mod.matching_anims(gpu.model, gpu.entry.name, anims, names)
        self.rebuild()

    def refresh_list(self):
        """The animations were scanned after the model was shown."""
        self.set_model(self.gpu)

    def on_escape(self):
        return False

    def layout(self):
        self.rebuild()

    # ------------------------------------------------------------------
    @property
    def anim(self):
        return self.items[self.cur][0] if self.cur is not None and self.cur < len(self.items) else None

    def frames(self):
        a = self.anim
        return max(1, int(round(a.duration * self.FPS)) + 1) if a else 1

    def frame(self):
        return min(self.time * self.FPS, self.frames() - 1)

    def pose(self):
        """(skinning matrices, model-space joint matrices) for the current
        time, or None (bind pose: panel closed or no animation picked)."""
        a = self.anim
        if not self.visible or a is None or self.gpu is None:
            return None
        key = (id(a), round(self.time, 5))
        if key != self._pose_key:
            m = self.gpu.model
            world = anim_mod.world_matrices(m, a.local_matrices(m, min(self.time, a.duration)))
            skin = np.einsum("nij,njk->nik", self.gpu.inv_bind, world)    # row-vector: inv(bind) . pose
            self._pose_key, self._pose = key, (skin, world)
        return self._pose

    def play(self, i):
        a = self.items[i][0]
        self.cur, self.time, self.playing = i, 0.0, True
        self.win.flash(f"{a.name}: {self.frames()} frames ({a.duration:.2f} s), "
                       f"{len(a.joints)} joints animated, from {self.win.cat.index.rel(a.obj.lu)}")
        self.rebuild()

    def stop(self):
        self.cur, self.playing, self.time = None, False, 0.0
        self.rebuild()

    def tick(self, dt):
        if not (self.visible and self.playing and self.anim):
            return
        length = self.anim.duration
        self.time += dt * self.speed
        if self.time >= length - 1e-6:
            if self.loop:
                self.time = self.time % max(length, 1e-6)
            else:
                self.time = length
                self.playing = False
                self.rebuild()
        self._update_timeline()

    def step(self, n):
        if not self.anim:
            return
        self.playing = False
        f = int(round(self.frame())) + n
        self.time = min((f % self.frames()) / self.FPS, self.anim.duration)
        self.rebuild()

    # ------------------------------------------------------------------
    def _timeline_rect(self):
        x0, y0, pw, ph = self.rect
        top = y0 + ph
        return x0 + 16, top - 84, pw - 32, 14

    def _update_timeline(self):
        for o in self.dyn:
            o.delete()
        self.dyn = []
        if not self.visible:
            return
        tx, ty, tw, th = self._timeline_rect()
        B = self.batch
        a = self.anim
        frac = min(1.0, self.time / a.duration) if a and a.duration > 0 else 0
        self.dyn.append(pyglet.shapes.Rectangle(tx, ty, max(1, int(tw * frac)), th, color=(70, 120, 190, 255), batch=B, group=self.g1))
        self.dyn.append(pyglet.shapes.Rectangle(tx + int(tw * frac) - 2, ty - 3, 4, th + 6, color=(235, 235, 235, 255), batch=B, group=self.g2))
        if a:
            txt = f"frame {int(self.frame()) + 1} / {self.frames()}   ·   {self.time:.2f} / {a.duration:.2f} s"
        else:
            txt = "bind pose - pick an animation below"
        self.dyn.append(pyglet.text.Label(txt, x=tx, y=ty - 18, font_size=FONT - 1, color=(190, 195, 205, 255),
                                          batch=B, group=self.g2))

    def rebuild(self):
        for o in self.objs + self.dyn:
            o.delete()
        self.objs, self.hits, self.dyn = [], [], []
        if not self.visible:
            return
        B, G0, G1, G2 = self.batch, self.g0, self.g1, self.g2
        x0, y0, pw, ph = self.rect
        top = y0 + ph
        add = self.objs.append
        add(pyglet.shapes.Rectangle(x0, y0, pw, ph, color=(22, 24, 28, 238), batch=B, group=G0))
        add(pyglet.shapes.Rectangle(x0, y0, 2, ph, color=(60, 64, 72, 255), batch=B, group=G1))
        chars = int((pw - 40) / self.win.char_w)
        gpu = self.gpu
        if gpu is None:
            add(pyglet.text.Label("No model loaded", x=x0 + 14, y=top - 24, font_size=FONT,
                                  color=(190, 190, 190, 255), batch=B, group=G2))
            return
        a = self.anim
        title = f"Animations ({len(self.items)})" + (f"  ·  {a.name}" if a else "")
        add(pyglet.text.Label(title[:chars], x=x0 + 14, y=top - 20, font_size=FONT + 1, weight="bold",
                              color=(235, 235, 235, 255), batch=B, group=G2))
        # transport buttons
        bx, by = x0 + 14, top - 54
        for name, text, w_ in (("play", "Pause" if self.playing else "Play", 64), ("prev", "<", 30), ("next", ">", 30),
                               ("loop", "Loop", 50), ("speed", f"{self.speed:g}x", 44), ("bind", "Bind pose", 78)):
            on = (name == "loop" and self.loop) or (name == "play" and self.playing)
            add(pyglet.shapes.Rectangle(bx, by, w_, 22, color=(52, 80, 120, 255) if on else (48, 52, 60, 255), batch=B, group=G1))
            add(pyglet.text.Label(text, x=bx + w_ // 2, y=by + 11, anchor_x="center", anchor_y="center",
                                  font_size=FONT - 1, color=(235, 235, 235, 255), batch=B, group=G2))
            self.hits.append((bx, by, w_, 22, name, None))
            bx += w_ + 5
        tx, ty, tw, th = self._timeline_rect()
        add(pyglet.shapes.Rectangle(tx, ty, tw, th, color=(45, 48, 55, 255), batch=B, group=G0))
        self.hits.append((tx, ty - 6, tw, th + 12, "timeline", None))
        self._update_timeline()
        # list
        list_top = top - self.CTRL_H
        add(pyglet.shapes.Rectangle(x0 + 8, list_top + 4, pw - 16, 1, color=(60, 64, 72, 255), batch=B, group=G1))
        msg = None
        if not len(gpu.model.joints):
            msg = ["This model has no skeleton."]
        elif not self.win.cat.anims_done:
            msg = ["Still scanning the folder..."]
        elif not self.items:
            msg = ["No animations for this model in the scanned folder.",
                   "A model's animations are usually in its own file",
                   "(e.g. dmegatro.wii for d_megatron_ref) and in the",
                   "levels / mission files (cutscenes)."]
        if msg:
            for k, line in enumerate(msg):
                add(pyglet.text.Label(line[:chars + 4], x=x0 + 14, y=list_top - 20 - k * 16, font_size=FONT - 1,
                                      color=(190, 190, 190, 255), batch=B, group=G2))
            return
        rows = []
        last = None
        for i, (an, _sc) in enumerate(self.items):
            grp = self.win.cat.index.rel(an.obj.lu).replace("\\", "/")
            if grp != last:
                rows.append(("head", grp, None))
                last = grp
            rows.append(("clip", an, i))
        total = len(rows) * self.ROW + 10
        avail = list_top - y0
        self.scroll = max(0, min(self.scroll, max(0, total - avail)))
        y = list_top + self.scroll
        for kind_, val, i in rows:
            y -= self.ROW
            if y > list_top - self.ROW + 2 or y < y0:
                continue
            if kind_ == "head":
                add(pyglet.text.Label(val[:int(chars * 0.92)], x=x0 + 12, y=y + 5, font_size=FONT - 1, weight="bold",
                                      color=(150, 175, 210, 255), batch=B, group=G2))
                continue
            sel = i == self.cur
            if sel:
                add(pyglet.shapes.Rectangle(x0 + 8, y, pw - 16, self.ROW, color=(52, 101, 164, 255), batch=B, group=G1))
            nfr = max(1, int(round(val.duration * self.FPS)) + 1)
            secs = f"{nfr} f · {val.duration:.1f} s"
            add(pyglet.text.Label(val.name[:max(8, chars - len(secs) - 2)], x=x0 + 22, y=y + 5, font_size=FONT,
                                  color=(255, 255, 255, 255) if sel else (215, 215, 215, 255), batch=B, group=G2))
            add(pyglet.text.Label(secs, x=x0 + pw - 18, y=y + 5, anchor_x="right", font_size=FONT - 1,
                                  color=(160, 165, 175, 255), batch=B, group=G2))
            self.hits.append((x0 + 8, y, pw - 16, self.ROW, "clip", i))
        self._sb = None
        if total > avail:
            th_ = max(30, int(avail * avail / total))
            ty_ = list_top - th_ - int((avail - th_) * self.scroll / max(1, total - avail))
            sx = x0 + pw - 10
            add(pyglet.shapes.Rectangle(sx, y0, 8, avail, color=(36, 39, 45, 255), batch=B, group=G1))
            col = (150, 156, 168, 255) if self.sb_drag is not None else (110, 115, 125, 255)
            add(pyglet.shapes.Rectangle(sx + 1, ty_, 6, th_, color=col, batch=B, group=G2))
            self._sb = (list_top, avail, total, th_, ty_)
            # checked before the clip rows, which reach under the bar
            self.hits.insert(0, (sx - 4, y0, 14, avail, "sbar", None))

    # ------------------------------------------------------------------
    def draw(self, scale):
        if not self.visible:
            return
        x0, y0, pw, ph = self.rect
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(int(x0 * scale), int(y0 * scale), int(pw * scale) + 1, int(ph * scale) + 1)
        self.batch.draw()
        gl.glDisable(gl.GL_SCISSOR_TEST)

    def _scrub(self, x):
        tx, ty, tw, th = self._timeline_rect()
        if not self.anim:
            return
        frac = max(0.0, min(1.0, (x - tx) / max(tw, 1)))
        self.time = frac * self.anim.duration
        self.playing = False
        self._update_timeline()

    def on_press(self, x, y, button):
        if not self.visible:
            return False
        x0, y0, pw, ph = self.rect
        if not (x0 <= x <= x0 + pw and y0 <= y <= y0 + ph):
            return False
        if button != mouse.LEFT:
            return True
        for hx, hy, hw, hh, act, arg in self.hits:
            if hx <= x <= hx + hw and hy <= y <= hy + hh:
                if act == "clip":
                    self.play(arg)
                elif act == "play":
                    if self.anim:
                        if not self.playing and self.time >= self.anim.duration - 1e-6 and not self.loop:
                            self.time = 0.0
                        self.playing = not self.playing
                    elif self.items:
                        self.play(self.cur if self.cur is not None else 0)
                        return True
                elif act == "prev":
                    self.step(-1)
                    return True
                elif act == "next":
                    self.step(1)
                    return True
                elif act == "loop":
                    self.loop = not self.loop
                elif act == "speed":
                    sp = [0.25, 0.5, 1.0, 2.0]
                    self.speed = sp[(sp.index(self.speed) + 1) % len(sp)] if self.speed in sp else 1.0
                elif act == "bind":
                    self.stop()
                    return True
                elif act == "sbar":
                    if self._sb:
                        _lt, _av, _tot, th_, ty_ = self._sb
                        # grab the thumb where it was clicked; a click on the track centres it there
                        self.sb_drag = (y - ty_) if ty_ <= y <= ty_ + th_ else th_ / 2
                        self._scroll_to_thumb(y)
                    return True
                elif act == "timeline":
                    self.dragging = True
                    self._scrub(x)
                    return True
                self.rebuild()
                return True
        return True

    def _scroll_to_thumb(self, y):
        """Scroll so the thumb's grabbed point sits at mouse height y."""
        list_top, avail, total, th_, _ty = self._sb
        travel = max(1, avail - th_)
        frac = (list_top - th_ - (y - self.sb_drag)) / travel
        self.scroll = int(round(max(0.0, min(1.0, frac)) * (total - avail)))
        self.rebuild()

    def on_drag(self, x, y):
        if self.sb_drag is not None and self._sb:
            self._scroll_to_thumb(y)
            return True
        if self.dragging:
            self._scrub(x)
            return True
        return False

    def on_release(self):
        if self.sb_drag is not None:
            self.sb_drag = None
            self.rebuild()
            return True
        if self.dragging:
            self.dragging = False
            self.rebuild()
            return True
        return False

    def on_scroll(self, x, y, sy):
        if not self.visible:
            return False
        x0, y0, pw, ph = self.rect
        if not (x0 <= x <= x0 + pw and y0 <= y <= y0 + ph):
            return False
        self.scroll = max(0, self.scroll - int(sy * 40))
        self.rebuild()
        return True


class SettingsPanel:
    """Side panel with export options (saved between sessions)."""
    W = 410
    OPTIONS = (("merge_pieces", "Merge Pieces", True,
                "Automatically merge the model's individual pieces into a single mesh."),
               ("export_skeleton", "Export Skeleton", True,
                "Export the model's bones and weights, if they exist."),
               ("export_animations", "Export Animations", True,
                "Export the model's animations, if they exist. Needs Export Skeleton."))

    def __init__(self, win):
        self.win = win
        self.visible = False
        self.batch = pyglet.graphics.Batch()
        self.g0, self.g1, self.g2 = (pyglet.graphics.Group(order=i) for i in range(3))
        self.objs, self.hits = [], []
        self.values = {k: bool(win.cfg.get(k, d)) for k, _t, d, _h in self.OPTIONS}

    def get(self, k):
        return self.values[k]

    @property
    def rect(self):
        w = self.win
        x0 = max(w.panel_w + 260, w.width - self.W)
        return x0, 28, w.width - x0, w.height - 46 - 28

    def toggle(self):
        self.visible = not self.visible
        if self.visible:
            self.win.close_panels(self)
        self.rebuild()
        self.win.flash("Export settings open" if self.visible else "Export settings closed")

    def layout(self):
        self.rebuild()

    def set_model(self, gpu):
        self.rebuild()

    def on_escape(self):
        return False

    def _save(self):
        self.win.cfg.update(self.values)
        try:
            cfg = json.load(open(CONFIG)) if os.path.exists(CONFIG) else {}
            cfg.update(self.values)
            json.dump(cfg, open(CONFIG, "w"))
        except Exception:
            pass

    @staticmethod
    def _wrap(text, chars):
        lines, cur = [], ""
        for word in text.split():
            if cur and len(cur) + 1 + len(word) > chars:
                lines.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}".strip()
        if cur:
            lines.append(cur)
        return lines

    def rebuild(self):
        for o in self.objs:
            o.delete()
        self.objs, self.hits = [], []
        if not self.visible:
            return
        B, G0, G1, G2 = self.batch, self.g0, self.g1, self.g2
        x0, y0, pw, ph = self.rect
        top = y0 + ph
        add = self.objs.append
        add(pyglet.shapes.Rectangle(x0, y0, pw, ph, color=(22, 24, 28, 238), batch=B, group=G0))
        add(pyglet.shapes.Rectangle(x0, y0, 2, ph, color=(60, 64, 72, 255), batch=B, group=G1))
        add(pyglet.text.Label("Export Settings", x=x0 + 14, y=top - 20, font_size=FONT + 1, weight="bold",
                              color=(235, 235, 235, 255), batch=B, group=G2))
        add(pyglet.text.Label("Export (E)", x=x0 + 14, y=top - 48, font_size=FONT - 1, weight="bold",
                              color=(150, 175, 210, 255), batch=B, group=G2))
        chars = max(20, int((pw - 60) / (self.win.char_w * 0.92)))
        y = top - 62
        for k, title, _d, help_ in self.OPTIONS:
            on = self.values[k]
            y -= 26
            add(pyglet.shapes.Rectangle(x0 + 16, y + 3, 16, 16, color=(90, 96, 108, 255), batch=B, group=G1))
            add(pyglet.shapes.Rectangle(x0 + 17, y + 4, 14, 14, color=(52, 101, 164, 255) if on else (70, 74, 82, 255),
                                        batch=B, group=G1))
            if on:
                add(pyglet.shapes.Rectangle(x0 + 20, y + 7, 8, 8, color=(235, 240, 250, 255), batch=B, group=G2))
            add(pyglet.text.Label(title, x=x0 + 42, y=y + 6, font_size=FONT,
                                  color=(235, 235, 235, 255), batch=B, group=G2))
            self.hits.append((x0 + 8, y, pw - 16, 24, k))
            for line in self._wrap(help_, chars):
                y -= 16
                add(pyglet.text.Label(line, x=x0 + 42, y=y + 4, font_size=FONT - 2,
                                      color=(160, 165, 175, 255), batch=B, group=G2))
            y -= 8

    def draw(self, scale):
        if not self.visible:
            return
        x0, y0, pw, ph = self.rect
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(int(x0 * scale), int(y0 * scale), int(pw * scale) + 1, int(ph * scale) + 1)
        self.batch.draw()
        gl.glDisable(gl.GL_SCISSOR_TEST)

    def on_press(self, x, y, button):
        if not self.visible:
            return False
        x0, y0, pw, ph = self.rect
        if not (x0 <= x <= x0 + pw and y0 <= y <= y0 + ph):
            return False
        if button == mouse.LEFT:
            for hx, hy, hw, hh, k in self.hits:
                if hx <= x <= hx + hw and hy <= y <= hy + hh:
                    self.values[k] = not self.values[k]
                    self._save()
                    self.rebuild()
                    title = next(t for kk, t, _d, _h in self.OPTIONS if kk == k)
                    self.win.flash(f"{title}: {'on' if self.values[k] else 'off'}")
                    break
        return True

    def on_drag(self, x, y):
        return False

    def on_release(self):
        return False

    def on_scroll(self, x, y, sy):
        if not self.visible:
            return False
        x0, y0, pw, ph = self.rect
        return x0 <= x <= x0 + pw and y0 <= y <= y0 + ph


# ----------------------------------------------------------------------------
# Viewer window
# ----------------------------------------------------------------------------
class Viewer(pyglet.window.Window):
    def __init__(self, root, auto_shots=None):
        cfg = pyglet.gl.Config(double_buffer=True, depth_size=24, major_version=3, minor_version=3,
                               forward_compatible=True, sample_buffers=1, samples=4)
        try:
            super().__init__(1400, 860, caption="Dark of the Moon Model Viewer",
                             resizable=True, config=cfg)
        except pyglet.window.NoSuchConfigException:
            super().__init__(1400, 860, caption="Dark of the Moon Model Viewer",
                             resizable=True)
        self.root = root
        self.cat = Catalogue(root)
        self.program = ShaderProgram(Shader(VERT, "vertex"), Shader(FRAG, "fragment"))
        self.lprogram = ShaderProgram(Shader(LINE_VERT, "vertex"), Shader(LINE_FRAG, "fragment"))

        self.flags = dict(tex=True, light=True, col=True, wire=False, cull=False, skel=False,
                          spin=False, help=False, nodup=True, noempty=True)
        self.bg = 0
        self.vis_rows = []          # indices into cat.models
        self.sel = -1              # row in visible
        self.scroll = 0
        self.filter = ""

        self.gpu = None            # GpuModel being shown
        self.missing = set()
        self.prefetch = []         # model indices the worker decodes when idle
        self.current_mi = None     # index (in cat.models) of the model shown
        self.loading = None        # model index being loaded
        self.load_gen = 0
        self.results = queue.Queue()
        self.cam = dict(yaw=35.0, pitch=18.0, dist=10.0, target=np.zeros(3, np.float32))
        self.msg = ""
        self.msg_time = 0
        self.msg_error = False

        try:
            self.cfg = json.load(open(CONFIG))
        except Exception:
            self.cfg = {}
        self.panel_w = int(self.cfg.get("panel_width", PANEL_W))
        self.search_focus = False
        self.drag = None           # "resize" | "thumb" | "view" | None
        self.thumb_grab = 0
        self.hover_div = False

        self.ui = pyglet.graphics.Batch()
        self.ui_bg = pyglet.graphics.Group(order=0)
        self.ui_mid = pyglet.graphics.Group(order=1)
        self.ui_fg = pyglet.graphics.Group(order=2)
        B, M, F = self.ui, self.ui_bg, self.ui_fg
        self.panel = pyglet.shapes.Rectangle(0, 0, 10, 10, color=(28, 30, 34, 255), batch=B, group=M)
        self.divider = pyglet.shapes.Rectangle(0, 0, 3, 10, color=(55, 58, 64, 255), batch=B, group=self.ui_mid)
        self.sel_bar = pyglet.shapes.Rectangle(0, 0, 10, ROW_H, color=(52, 101, 164, 255), batch=B, group=self.ui_mid)
        self.head = pyglet.text.Label("", x=10, y=0, font_size=FONT + 1, weight="bold",
                                      color=(235, 235, 235, 255), batch=B, group=F)
        # search box
        self.search_border = pyglet.shapes.Rectangle(0, 0, 10, SEARCH_H, color=(70, 74, 82, 255), batch=B, group=M)
        self.search_bg = pyglet.shapes.Rectangle(0, 0, 10, SEARCH_H - 2, color=(16, 17, 20, 255), batch=B, group=self.ui_mid)
        self.search_lbl = pyglet.text.Label("", x=0, y=0, font_size=FONT, anchor_y="center",
                                            color=(235, 235, 235, 255), batch=B, group=F)
        self.search_clear_bg = pyglet.shapes.Circle(0, 0, 8, color=(60, 63, 70, 255), batch=B, group=self.ui_mid)
        self.search_clear = pyglet.text.Label("×", x=0, y=0, font_size=FONT + 2, anchor_x="center",
                                              anchor_y="center", color=(170, 170, 175, 255), batch=B, group=F)
        # scrollbar
        self.sb_track = pyglet.shapes.Rectangle(0, 0, SB_W, 10, color=(38, 40, 45, 255), batch=B, group=self.ui_mid)
        self.sb_thumb = pyglet.shapes.Rectangle(0, 0, SB_W, 10, color=(95, 100, 110, 255), batch=B, group=F)
        self.rows = []
        self.info = pyglet.text.Label("", x=0, y=0, font_size=FONT, multiline=True,
                                      width=700, color=(230, 230, 230, 255), anchor_y="top", batch=B, group=F)
        self.status_lbl = pyglet.text.Label("", x=0, y=8, font_size=FONT,
                                            color=(200, 200, 200, 255), batch=B, group=F)
        self.help_bg = pyglet.shapes.Rectangle(0, 0, 10, 10, color=(0, 0, 0, 200), batch=B, group=M)
        self.help_lbl = pyglet.text.Label(HELP_TEXT, x=0, y=0, font_size=FONT + 1, multiline=True,
                                          width=260, color=(240, 240, 240, 255), anchor_y="top", batch=B, group=F)
        self.buttons = []          # (name, rect, label)
        for name, text in (("anims", "Animations  (A)"), ("pieces", "Pieces  (I)"), ("export", "Export  (E)"),
                           ("exportdir", "Export Folder"), ("settings", "Export Settings  (O)"),
                           ("help", "Help  (H)")):
            r = pyglet.shapes.Rectangle(0, 0, 10, 26, color=(50, 54, 62, 235), batch=B, group=M)
            lb = pyglet.text.Label(text, x=0, y=0, font_size=FONT, anchor_x="center", anchor_y="center",
                                   color=(235, 235, 235, 255), batch=B, group=F)
            self.buttons.append((name, r, lb))
        probe = pyglet.text.Label("abcdefghijklmnopqrstuvwxyz_0123456789 #·/", font_size=FONT)
        self.char_w = max(probe.content_width / 41.0, 4.0)
        probe.delete()
        self.piecepanel = PiecePanel(self)
        self.setpanel = SettingsPanel(self)
        self.animpanel = AnimPanel(self)
        self.export_job = None     # background export: dict(name, out, result, error, finished)
        self.cur_resize = self.get_system_mouse_cursor(self.CURSOR_SIZE_LEFT_RIGHT)
        self.cur_text = self.get_system_mouse_cursor(self.CURSOR_TEXT)
        self._layout()

        self.auto_shots = auto_shots
        threading.Thread(target=self.cat.scan, args=(root,), daemon=True).start()
        self.worker_q = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()
        pyglet.clock.schedule_interval(self.update, 1 / 60)
        self._last_model_count = -1
        self._rows_dirty = False
        self._scan_was_done = False

    # ------------------------------------------------------------- layout --
    def _clamp_panel(self):
        self.panel_w = int(max(PANEL_MIN, min(int(self.panel_w), self.width - VIEW_MIN)))

    def _layout(self):
        self._clamp_panel()
        w, h, pw = self.width, self.height, self.panel_w
        self.panel.width, self.panel.height = pw, h
        self.divider.x, self.divider.height = pw - 2, h
        self.divider.color = (90, 130, 190, 255) if (self.hover_div or self.drag == "resize") else (55, 58, 64, 255)
        self.head.y = h - 22
        # search box
        sy = h - 36 - SEARCH_H
        self.search_rect = (8, sy, pw - 16, SEARCH_H)
        self.search_border.x, self.search_border.y = 8, sy
        self.search_border.width = pw - 16
        self.search_border.color = (90, 140, 220, 255) if self.search_focus else (70, 74, 82, 255)
        self.search_bg.x, self.search_bg.y, self.search_bg.width = 9, sy + 1, pw - 18
        self.search_lbl.x, self.search_lbl.y = 16, sy + SEARCH_H // 2
        self.search_clear.x, self.search_clear.y = pw - 22, sy + SEARCH_H // 2 + 1
        self.search_clear_bg.x, self.search_clear_bg.y = pw - 22, sy + SEARCH_H // 2
        # list + scrollbar
        top = sy - 8
        bottom = 6
        nrows = max(1, (top - bottom) // ROW_H)
        while len(self.rows) < nrows:
            self.rows.append(pyglet.text.Label("", x=12, y=0, font_size=FONT,
                                               color=(215, 215, 215, 255), batch=self.ui, group=self.ui_fg))
        for i, r in enumerate(self.rows):
            r.y = top - (i + 1) * ROW_H + 5
            r.visible = i < nrows
        self.nrows = nrows
        self.list_top = top
        self.list_bottom = top - nrows * ROW_H
        self.sb_x = pw - SB_W - 5
        self.sb_track.x, self.sb_track.y = self.sb_x, self.list_bottom
        self.sb_track.height = nrows * ROW_H
        self.sel_bar.x, self.sel_bar.width = 0, self.sb_x - 3
        self.row_chars = max(8, int((self.sb_x - 22) / self.char_w))
        # buttons, top-right of the 3D view
        bx = w - 10
        for name, r, lb in reversed(self.buttons):
            bw = max(110, int(lb.content_width) + 24)
            bx -= bw
            r.x, r.y, r.width, r.height = bx, h - 36, bw, 26
            lb.x, lb.y = bx + bw // 2, h - 23
            bx -= 8
        for pnl in self._panels():
            pnl.layout()
        # right side
        self.info.x, self.info.y = pw + 14, h - 10
        self.status_lbl.x = pw + 14
        self._place_help()
        self._refresh_rows()

    def _place_help(self):
        """Help box under the top-right buttons (left of the side panel if open)."""
        h = self.height
        right = self._view_right() - 10
        hh = int(self.help_lbl.content_height) + 28
        hw = int(self.help_lbl.content_width) + 40
        self.help_bg.width, self.help_bg.height = hw, hh
        self.help_bg.x, self.help_bg.y = right - hw, h - 46 - hh
        self.help_lbl.x, self.help_lbl.y = right - hw + 20, h - 46 - 14

    def _panels(self):
        return [p for p in (getattr(self, "animpanel", None), getattr(self, "piecepanel", None),
                            getattr(self, "setpanel", None)) if p is not None]

    def _view_right(self):
        for pnl in self._panels():
            if pnl.visible:
                return pnl.rect[0]
        return self.width

    def close_panels(self, keep):
        """Side panels share the right edge: opening one closes the others."""
        for pnl in self._panels():
            if pnl is not keep and pnl.visible:
                pnl.toggle()

    def on_resize(self, w, h):
        super().on_resize(w, h)
        self._layout()

    # ------------------------------------------------------------- list --
    def _rebuild_visible(self):
        with self.cat.lock:
            models = list(self.cat.models)
        dups = self.cat.dups if self.cat.dups_done else {}
        empty = self.cat.empty if self.cat.dups_done else set()
        terms = self.filter.lower().split()
        cur = self.current_mi            # model on screen (or loading)
        vis = []
        for i, m in enumerate(models):
            if self.flags["nodup"] and i in dups:
                continue
            if self.flags["noempty"] and i in empty:
                continue
            if terms:
                hay = f"{m.rel.lower()} {m.name.lower()}"
                if not all(t in hay for t in terms):
                    continue
            vis.append(i)
        self.vis_rows = vis
        if cur is None:
            self.sel = -1
            if vis:
                self.select(0)
                return
        elif cur in vis:
            self.sel = vis.index(cur)
            if not (self.scroll <= self.sel < self.scroll + self.nrows):
                self.scroll = max(0, self.sel - self.nrows // 2)
        elif cur in dups and dups[cur] in vis:
            # the shown model is now hidden as a duplicate: show the kept copy
            self.sel = vis.index(dups[cur])
            self.scroll = max(0, self.sel - self.nrows // 2)
            self.request_load(dups[cur])
        elif vis:
            # the shown model was filtered out: jump to the top match
            # (loaded after a short pause so typing doesn't trigger a load per key)
            self.sel, self.scroll = 0, 0
            pyglet.clock.unschedule(self._load_selected)
            pyglet.clock.schedule_once(self._load_selected, 0.25)
        else:
            self.sel = -1                # no matches
        self._refresh_rows()

    def _refresh_rows(self):
        self._rows_dirty = False
        models = self.cat.models
        n = len(self.vis_rows)
        self.scroll = max(0, min(self.scroll, max(0, n - self.nrows)))
        lim = self.row_chars
        for i, r in enumerate(self.rows[:self.nrows]):
            k = self.scroll + i
            if k < n:
                m = models[self.vis_rows[k]]
                txt = f"{m.name}  ·  {m.rel}"
                _set_label(r, txt if len(txt) <= lim else txt[:lim - 1] + "…",
                           (255, 255, 255, 255) if k == self.sel else (205, 205, 210, 255))
            else:
                _set_label(r, "")
        if 0 <= self.sel - self.scroll < self.nrows and self.sel >= 0:
            self.sel_bar.visible = True
            self.sel_bar.y = self.list_top - (self.sel - self.scroll + 1) * ROW_H
        else:
            self.sel_bar.visible = False
        # scrollbar
        track = self.nrows * ROW_H
        if n > self.nrows:
            th = max(24, int(track * self.nrows / n))
            frac = self.scroll / max(1, n - self.nrows)
            self.sb_thumb.height = th
            self.sb_thumb.x = self.sb_x
            self.sb_thumb.y = self.list_top - th - int((track - th) * frac)
            self.sb_thumb.color = (130, 136, 148, 255) if self.drag == "thumb" else (95, 100, 110, 255)
            self.sb_thumb.visible = self.sb_track.visible = True
        else:
            self.sb_thumb.visible = self.sb_track.visible = False
        total = len(models)
        _set_label(self.head, f"Models  {n:,}" + (f" of {total:,}" if n != total else ""))
        # search box text (placeholder when empty)
        max_chars = max(4, int((self.search_rect[2] - 40) / self.char_w))
        if self.filter:
            t = self.filter if len(self.filter) <= max_chars else "…" + self.filter[-(max_chars - 1):]
            caret = "|" if (self.search_focus and int(time.time() * 2) % 2 == 0) else ""
            _set_label(self.search_lbl, t + caret, (235, 235, 235, 255))
        else:
            _set_label(self.search_lbl,
                       ("|" if (self.search_focus and int(time.time() * 2) % 2 == 0) else "") +
                       ("" if self.search_focus else "Search models…  ( / )"),
                       (235, 235, 235, 255) if self.search_focus else (120, 124, 132, 255))
        self.search_clear.visible = self.search_clear_bg.visible = bool(self.filter)

    def set_filter(self, text):
        self.filter = text
        self.scroll = 0
        self._rebuild_visible()

    def _load_selected(self, dt=None):
        pyglet.clock.unschedule(self._load_selected)
        if 0 <= self.sel < len(self.vis_rows) and self.vis_rows[self.sel] != self.current_mi:
            self.request_load(self.vis_rows[self.sel])

    def select(self, row):
        if not self.vis_rows:
            return
        row = int(max(0, min(row, len(self.vis_rows) - 1)))
        changed = row != self.sel
        self.sel = row
        if self.sel < self.scroll:
            self.scroll = self.sel
        elif self.sel >= self.scroll + self.nrows:
            self.scroll = self.sel - self.nrows + 1
        self._refresh_rows()
        mi = self.vis_rows[row]
        if changed or self.gpu is None or self.gpu.entry is not self.cat.models[mi]:
            self.request_load(mi)

    # ------------------------------------------------------------- loading --
    def request_load(self, mi):
        self.current_mi = mi
        self.load_gen += 1
        self.loading = mi
        self.worker_q.put((self.load_gen, mi))

    TEX_CACHE_BYTES = 200 * 1024 * 1024

    def _decode(self, e, model_cache, tex_cache):
        """Parse a model and decode its textures (worker thread only)."""
        k = (e.obj.lu.path, e.obj.off)
        m = model_cache.get(k)
        if m is None:
            m = Model(e.obj.read(), e.obj.name)
            model_cache[k] = m
            while len(model_cache) > 64:
                model_cache.pop(next(iter(model_cache)))
        surf_tex, arrays, missing = [], {}, set()
        for s in m.surfaces:
            t = self.cat.surface_texture(e.obj, s)
            surf_tex.append(t)
            if t is None:
                missing.update(s.texrefs[:1])
                continue
            tk = tex_key(t)
            if tk in arrays:
                continue
            if tk in tex_cache:
                arrays[tk] = tex_cache[tk]
                continue
            try:
                arrays[tk] = read_rgba(t)
            except Exception:
                continue
            tex_cache[tk] = arrays[tk]
            self._tex_bytes += arrays[tk].nbytes
            while self._tex_bytes > self.TEX_CACHE_BYTES and len(tex_cache) > 1:
                old = tex_cache.pop(next(iter(tex_cache)))
                self._tex_bytes -= old.nbytes
        return m, surf_tex, arrays, missing

    def _worker(self):
        """Decode meshes and texture pixels off the UI thread. When idle,
        pre-decode the models next to the selection so Up/Down is quicker."""
        tex_cache, model_cache = {}, {}
        self._tex_bytes = 0
        self._model_cache = model_cache
        while True:
            try:
                gen, mi = self.worker_q.get(timeout=0.15)
            except queue.Empty:
                try:
                    mj = self.prefetch.pop(0)
                except (IndexError, AttributeError):
                    continue
                try:
                    self._decode(self.cat.models[mj], model_cache, tex_cache)
                except Exception:
                    pass
                continue
            # skip stale requests when the user is scrolling quickly
            while not self.worker_q.empty():
                gen, mi = self.worker_q.get()
            if gen != self.load_gen:
                continue
            e = self.cat.models[mi]
            try:
                self.results.put((gen, mi) + self._decode(e, model_cache, tex_cache) + (None,))
            except Exception as ex:
                self.results.put((gen, mi, None, [], {}, set(), ex))

    def _finish_load(self, gen, mi, model, surf_tex, arrays, missing, err):
        if gen != self.load_gen:
            return
        self.loading = None
        e = self.cat.models[mi]
        if self.gpu:
            self.gpu.delete()
            self.gpu = None
        if err:
            self.piecepanel.set_model(None)
            self.animpanel.set_model(None)
            self.flash(f"Could not load {e.name}: {err}", error=True)
            self.info.text = f"{e.name}\n{e.rel}\nerror: {err}"
            return
        self.gpu = GpuModel(self.program, self.lprogram, e, model, surf_tex, arrays)
        self.missing = missing
        # neighbours to pre-decode while the user looks at this one
        if 0 <= self.sel < len(self.vis_rows):
            mc = getattr(self, "_model_cache", {})
            self.prefetch = [self.vis_rows[r] for r in (self.sel + 1, self.sel - 1, self.sel + 2)
                             if 0 <= r < len(self.vis_rows)
                             and (self.cat.models[self.vis_rows[r]].obj.lu.path,
                                  self.cat.models[self.vis_rows[r]].obj.off) not in mc]
        self.piecepanel.set_model(self.gpu)
        self.animpanel.set_model(self.gpu)
        self.frame()
        self._update_info(e, model, surf_tex, missing)

    def _update_info(self, e, m, surf_tex, missing=()):
        lo, hi = m.bounds()
        size = hi - lo
        texs = {tex_key(t): t for t in surf_tex if t is not None}
        other = sum(1 for t in texs.values() if t.lu is not e.obj.lu)
        lines = [f"{e.name}", f"{e.rel}.wii  ·  object 0x{e.obj.id:08x}",
                 f"{_n(len(m.surfaces), 'piece')} · {_n(m.vertex_count, 'vertex', 'vertices')} · "
                 f"{_n(m.triangle_count, 'triangle')}",
                 f"{_n(len(m.joints), 'joint')} · size {size[0]:.2f} × {size[1]:.2f} × {size[2]:.2f}",
                 _n(len(texs), 'texture') + (f" ({other} from other files)" if other else "")]
        if missing:
            lines.append(f"{len(missing)} texture(s) not found in this folder"
                         + ("" if self.cat.done else " yet (still scanning)"))
        if not m.surfaces:
            lines.append("nothing to draw: no renderable pieces (e.g. a shadow-only model)")
        if m.warnings:
            lines.append("warnings: " + "; ".join(sorted(set(m.warnings))[:3]))
        self.info.text = "\n".join(lines)

    # ------------------------------------------------------------- camera --
    def frame(self):
        if not self.gpu:
            return
        pose = self.animpanel.pose() if hasattr(self, "animpanel") else None
        if pose is not None:
            # centre on the animated pose (cutscenes move the model around)
            skin = pose[0]
            pts = []
            for s in self.gpu.model.surfaces:
                if len(s.pos):
                    K = skin[s.joint] if 0 <= s.joint < len(skin) else np.eye(4)
                    lo, hi = s.pos.min(0), s.pos.max(0)
                    pts.append(np.c_[np.array([lo, hi]), np.ones(2)] @ K)
            if pts:
                P = np.vstack(pts)[:, :3]
                self.cam["target"] = ((P.min(0) + P.max(0)) / 2).astype(np.float32)
                self.cam["dist"] = max(self.gpu.radius, float(np.linalg.norm(P.max(0) - P.min(0))) / 2) * 2.6
                return
        self.cam["target"] = self.gpu.center.copy()
        self.cam["dist"] = self.gpu.radius * 2.6
        self.cam["yaw"], self.cam["pitch"] = 35.0, 15.0

    def _eye(self):
        y, p = math.radians(self.cam["yaw"]), math.radians(self.cam["pitch"])
        d = self.cam["dist"]
        off = np.array([math.cos(p) * math.sin(y), math.sin(p), math.cos(p) * math.cos(y)], np.float32) * d
        return self.cam["target"] + off

    # ------------------------------------------------------------- frame --
    def update(self, dt):
        while not self.results.empty():
            self._finish_load(*self.results.get())
        n = (len(self.cat.models), self.cat.dups_done)
        if n != self._last_model_count:
            self._last_model_count = n
            self._rebuild_visible()
        if self.cat.done and not self._scan_was_done:
            self._scan_was_done = True
            # textures stored in files indexed after this model: load it again
            if self.missing and self.current_mi is not None:
                self.request_load(self.current_mi)
            if not self.cat.files_total:
                self.info.text = ("No .wii files in this folder.\n"
                                  "Pick the game folder (\"transformers dark of the moon\").")
        if self.cat.anims_done and not getattr(self, "_anims_seen", False):
            self._anims_seen = True
            self.animpanel.refresh_list()       # animations were still being scanned
        self.animpanel.tick(dt)
        if self.flags["spin"]:
            self.cam["yaw"] += 20 * dt
        self._check_export()
        if self.search_focus:
            self._refresh_rows()          # caret blink
        # bottom line: only the result of the last action (export, screenshot,
        # toggles, errors), for 10 seconds
        recent = bool(self.msg) and time.time() - self.msg_time < 10
        _set_label(self.status_lbl, self.msg if recent else "",
                   (255, 120, 110, 255) if self.msg_error else (150, 230, 150, 255))
        if self.auto_shots is not None:
            self._auto_step()

    def on_draw(self):
        if self._rows_dirty:
            self._refresh_rows()
        bg = BGS[self.bg]
        gl.glClearColor(bg[0], bg[1], bg[2], 1)
        self.clear()
        fbw, fbh = self.get_framebuffer_size()
        sx = fbw / max(self.width, 1)
        right = self._view_right()
        vx, vw = int(self.panel_w * sx), max(1, int(right * sx) - int(self.panel_w * sx))
        if self.gpu:
            gl.glViewport(vx, 0, vw, fbh)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthFunc(gl.GL_LEQUAL)
            gl.glDisable(gl.GL_BLEND)
            if self.flags["cull"]:
                gl.glEnable(gl.GL_CULL_FACE)
                gl.glCullFace(gl.GL_BACK)
                gl.glFrontFace(gl.GL_CW)
            if self.flags["wire"]:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            eye = self._eye()
            view = look_at(eye, self.cam["target"], np.array([0, 1, 0], np.float32))
            r = self.gpu.radius
            near = max(self.cam["dist"] - r * 1.5, self.cam["dist"] * 0.002, 1e-3)
            proj = perspective(45, vw / max(fbh, 1), near, self.cam["dist"] + r * 3)
            mvp = gl_mat(proj @ view)
            p = self.program
            p.use()
            p["mvp"] = mvp
            p["eye"] = tuple(float(x) for x in eye)
            p["use_tex"] = int(self.flags["tex"])
            p["use_light"] = int(self.flags["light"])
            p["use_col"] = int(self.flags["col"])
            p["diffuse_tex"] = 0
            hl = self.piecepanel.highlight
            pose = self.animpanel.pose()
            skin = pose[0] if pose is not None else None
            ident = tuple(np.eye(4, dtype=np.float32).ravel().tolist())
            nj = len(skin) if skin is not None else 0
            gl.glActiveTexture(gl.GL_TEXTURE0)
            for vl, s, tid in self.gpu.parts:
                if hl is not None and s.index != hl:
                    continue                  # surface selected: show only its faces
                # row-vector skinning matrix, row by row = GLSL's column-major K^T
                p["bone"] = tuple(skin[s.joint].astype(np.float32).ravel().tolist()) \
                    if skin is not None and 0 <= s.joint < nj else ident
                p["has_tex"] = int(bool(tid))
                p["tint"] = (0.75, 0.75, 0.78) if s.texrefs else (0.55, 0.6, 0.7)
                gl.glBindTexture(gl.GL_TEXTURE_2D, tid or 0)
                vl.draw(gl.GL_TRIANGLES)
            p.stop()
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            gl.glDisable(gl.GL_CULL_FACE)
            if self.flags["skel"] and self.gpu.joints is not None:
                if pose is not None:
                    self.gpu.set_skeleton(pose[1])
                elif self.gpu.posed:
                    self.gpu.set_skeleton(None)
                gl.glDisable(gl.GL_DEPTH_TEST)
                gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
                lp = self.lprogram
                lp.use()
                lp["mvp"] = mvp
                if self.gpu.bones is not None:
                    lp["line_color"] = (1.0, 0.85, 0.2, 1.0)
                    self.gpu.bones.draw(gl.GL_LINES)
                lp["line_color"] = (1.0, 0.3, 0.2, 1.0)
                self.gpu.joints.draw(gl.GL_POINTS)
                lp.stop()
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glViewport(0, 0, fbw, fbh)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self.help_bg.visible = self.help_lbl.visible = self.flags["help"]
        if self.flags["help"]:
            self._place_help()
        self.ui.draw()
        for pnl in self._panels():
            pnl.draw(fbw / max(self.width, 1))

    # ------------------------------------------------------------- input --
    def flash(self, text, error=False):
        self.msg, self.msg_time = text, time.time()
        self.msg_error = error

    def on_key_press(self, sym, mods):
        if sym == key.E and mods & (key.MOD_CTRL | key.MOD_COMMAND):
            self.export_current()
            return pyglet.event.EVENT_HANDLED
        if sym == key.F12:
            self.screenshot()
            return pyglet.event.EVENT_HANDLED
        if self.search_focus:
            if sym in (key.ENTER, key.RETURN):
                self._load_selected()        # open the highlighted match now
                self.set_search_focus(False)
                return pyglet.event.EVENT_HANDLED
            if sym == key.ESCAPE:
                if self.filter:
                    self.set_filter("")
                else:
                    self.set_search_focus(False)
                return pyglet.event.EVENT_HANDLED
            if sym == key.BACKSPACE:
                if mods & key.MOD_CTRL:
                    self.set_filter(self.filter.rstrip().rsplit(" ", 1)[0] if " " in self.filter.rstrip() else "")
                else:
                    self.set_filter(self.filter[:-1])
                return pyglet.event.EVENT_HANDLED
            if sym == key.A and mods & key.MOD_CTRL:
                self.set_filter("")
                return pyglet.event.EVENT_HANDLED
            if sym == key.V and mods & (key.MOD_CTRL | key.MOD_COMMAND):
                try:
                    clip = self.get_clipboard_text() or ""
                except Exception:
                    clip = ""
                if clip:
                    self.set_filter(self.filter + clip.splitlines()[0])
                return pyglet.event.EVENT_HANDLED
            if sym not in (key.UP, key.DOWN, key.PAGEUP, key.PAGEDOWN):
                return pyglet.event.EVENT_HANDLED   # typed characters arrive in on_text
        if sym == key.ESCAPE and self.piecepanel.on_escape():
            return pyglet.event.EVENT_HANDLED
        if sym == key.ESCAPE:
            if self.filter:
                self.set_filter("")
                return pyglet.event.EVENT_HANDLED
            if self.flags["help"]:
                self.flags["help"] = False
                return pyglet.event.EVENT_HANDLED
            return pyglet.event.EVENT_HANDLED   # don't quit on Esc
        step = {key.UP: -1, key.DOWN: 1, key.PAGEUP: -self.nrows, key.PAGEDOWN: self.nrows}
        if sym in step:
            self.select(self.sel + step[sym])
        elif sym == key.HOME:
            self.select(0)
        elif sym == key.END:
            self.select(len(self.vis_rows) - 1)
        elif sym == key.SLASH or (sym == key.F and mods & (key.MOD_CTRL | key.MOD_COMMAND)):
            self.set_search_focus(True)
        elif sym in (key.F, key.R):
            self.frame()
        elif sym == key.SPACE:
            self.flags["spin"] = not self.flags["spin"]
        elif sym == key.B:
            self.bg = (self.bg + 1) % len(BGS)
        elif sym == key.I:
            self.piecepanel.toggle()
        elif sym == key.A:
            self.animpanel.toggle()
        elif sym in (key.BRACKETLEFT, key.BRACKETRIGHT) and self.animpanel.visible:
            self.animpanel.step(-1 if sym == key.BRACKETLEFT else 1)
        elif sym == key.O:
            self.setpanel.toggle()
        elif sym == key.E:
            self.export_current()
        else:
            names = {key.T: "tex", key.L: "light", key.V: "col", key.W: "wire", key.S: "skel",
                     key.C: "cull", key.H: "help", key.D: "nodup", key.N: "noempty"}
            if sym in names:
                f = names[sym]
                self.flags[f] = not self.flags[f]
                if f in ("nodup", "noempty"):
                    self._rebuild_visible()
                    self.flash(f"{LABELS[f]}: {'on' if self.flags[f] else 'off'}")
                elif f == "skel" and self.flags[f] and self.gpu is not None and self.gpu.joints is None:
                    self.flash("Skeleton: on (this model has no joints)")
                elif f != "help":
                    self.flash(f"{LABELS[f]}: {'on' if self.flags[f] else 'off'}")
        return pyglet.event.EVENT_HANDLED

    def on_text(self, text):
        if self.search_focus and text.isprintable():
            if text == "/" and not self.filter and getattr(self, "_just_focused", False):
                return
            self.set_filter(self.filter + text)

    def set_search_focus(self, on):
        if on != self.search_focus:
            self.search_focus = on
            self._just_focused = on
            pyglet.clock.schedule_once(lambda dt: setattr(self, "_just_focused", False), 0.05)
            self._layout()

    def _on_divider(self, x):
        return abs(x - self.panel_w) <= 5

    def _in_search(self, x, y):
        sx, sy, sw, sh = self.search_rect
        return sx <= x <= sx + sw and sy <= y <= sy + sh

    def on_mouse_motion(self, x, y, dx, dy):
        x, y = int(x), int(y)          # pyglet 2.1 reports float coordinates
        hov = self._on_divider(x)
        if hov != self.hover_div:
            self.hover_div = hov
            self._layout()
        if hov:
            self.set_mouse_cursor(self.cur_resize)
        elif self._in_search(x, y):
            self.set_mouse_cursor(self.cur_text)
        else:
            self.set_mouse_cursor(None)

    def _scroll_to_thumb(self, y):
        n = len(self.vis_rows)
        if n <= self.nrows:
            return
        track = self.nrows * ROW_H
        th = self.sb_thumb.height
        # y of the thumb's top edge, relative to the top of the track
        rel = (self.list_top - (y + self.thumb_grab)) / max(1, track - th)
        self.scroll = int(round(max(0.0, min(1.0, rel)) * (n - self.nrows)))
        self._rows_dirty = True          # redrawn once per frame (see on_draw)

    def on_mouse_press(self, x, y, button, mods):
        x, y = int(x), int(y)
        self.drag = None
        if any(pnl.on_press(x, y, button) for pnl in self._panels()):
            return
        if self._on_divider(x) and button == mouse.LEFT:
            self.drag = "resize"
            self._layout()
            return
        if x < self.panel_w:
            if button != mouse.LEFT:
                return
            if self._in_search(x, y):
                if self.filter and x >= self.search_clear.x - 10:
                    self.set_filter("")
                self.set_search_focus(True)
                return
            self.set_search_focus(False)
            n = len(self.vis_rows)
            if x >= self.sb_x - 2 and n > self.nrows and self.list_bottom <= y <= self.list_top:
                t = self.sb_thumb
                if t.y <= y <= t.y + t.height:          # grab the thumb
                    self.thumb_grab = t.y + t.height - y
                else:                                    # click on the track: jump there
                    self.thumb_grab = t.height // 2
                    self._scroll_to_thumb(y)
                self.drag = "thumb"
                self._refresh_rows()
                return
            row = (self.list_top - y) // ROW_H
            if 0 <= row < self.nrows and self.scroll + row < n:
                self.select(self.scroll + row)
            return
        for name, r, lb in self.buttons:
            if button == mouse.LEFT and r.x <= x <= r.x + r.width and r.y <= y <= r.y + r.height:
                r.color = (80, 110, 160, 255)
                pyglet.clock.schedule_once(lambda dt, r=r: setattr(r, "color", (50, 54, 62, 235)), 0.15)
                if name == "anims":
                    self.animpanel.toggle()
                elif name == "pieces":
                    self.piecepanel.toggle()
                elif name == "export":
                    self.export_current()
                elif name == "exportdir":
                    self.open_export_folder()
                elif name == "settings":
                    self.setpanel.toggle()
                elif name == "help":
                    self.flags["help"] = not self.flags["help"]
                return
        self.set_search_focus(False)
        self.drag = "view"

    def on_mouse_release(self, x, y, button, mods):
        x, y = int(x), int(y)
        was = self.drag
        self.drag = None
        if any(pnl.on_release() for pnl in self._panels()):
            return
        if was == "resize":
            self.cfg["panel_width"] = self.panel_w
            try:
                cfg = json.load(open(CONFIG)) if os.path.exists(CONFIG) else {}
                cfg["panel_width"] = self.panel_w
                json.dump(cfg, open(CONFIG, "w"))
            except Exception:
                pass
            self.hover_div = self._on_divider(x)
            self._layout()
        elif was == "thumb":
            self._refresh_rows()

    def on_mouse_drag(self, x, y, dx, dy, buttons, mods):
        x, y = int(x), int(y)
        if any(pnl.on_drag(x, y) for pnl in self._panels()):
            return
        if self.drag == "resize":
            self.panel_w = x
            self._layout()
            return
        if self.drag == "thumb":
            self._scroll_to_thumb(y)
            return
        if self.drag != "view":
            return
        if buttons & mouse.LEFT and not mods & key.MOD_SHIFT:
            self.cam["yaw"] -= dx * 0.4
            self.cam["pitch"] = max(-89, min(89, self.cam["pitch"] - dy * 0.4))
        elif buttons & (mouse.RIGHT | mouse.MIDDLE | mouse.LEFT):
            y_, p_ = math.radians(self.cam["yaw"]), math.radians(self.cam["pitch"])
            right = np.array([math.cos(y_), 0, -math.sin(y_)], np.float32)
            up = np.array([-math.sin(p_) * math.sin(y_), math.cos(p_), -math.sin(p_) * math.cos(y_)], np.float32)
            k = self.cam["dist"] * 0.0015
            self.cam["target"] = self.cam["target"] - right * dx * k - up * dy * k

    def on_mouse_scroll(self, x, y, sx, sy):
        x, y = int(x), int(y)
        if any(pnl.on_scroll(x, y, sy) for pnl in self._panels()):
            return
        if x < self.panel_w:
            step = int(round(sy * 3)) or (1 if sy > 0 else -1)
            self.scroll = max(0, self.scroll - step)
            self._rows_dirty = True
        else:
            self.cam["dist"] *= 0.88 ** sy

    def on_close(self):
        self.cat.stop = True
        super().on_close()

    # ------------------------------------------------------------- output --
    def screenshot(self, path=None):
        try:
            if path is None:
                e = self.gpu.entry if self.gpu else None
                name = safe(e.name) if e else "viewer"
                path = os.path.join(output_dir("screenshots"), f"{name}_{time.strftime('%H%M%S')}.png")
            pyglet.image.get_buffer_manager().get_color_buffer().save(path)
        except Exception as e:
            self.flash(f"Screenshot failed: {e}", error=True)
            return None
        self.flash(f"Screenshot saved: {os.path.abspath(path)}")
        print("screenshot:", os.path.abspath(path))
        return path

    @staticmethod
    def _export_dir(e):
        """.model_exports/<file>/<model>/: the .glb (with the skeleton) + its PNG textures."""
        return os.path.join(output_dir(".model_exports"), safe(e.obj.lu.name), safe(e.name))

    def open_export_folder(self):
        """Open the selected model's export folder in the system file browser."""
        if not (0 <= self.sel < len(self.vis_rows)):
            self.flash("No model selected", error=True)
            return
        e = self.cat.models[self.vis_rows[self.sel]]
        try:
            d = self._export_dir(e)
        except Exception as ex:
            self.flash(f"Could not open the export folder: {ex}", error=True)
            return
        if not os.path.isdir(d):
            self.flash(f"{e.name} has not been exported yet - press E (or Export) first", error=True)
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(d)                    # noqa: (Windows only)
            else:
                import subprocess
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", d],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as ex:
            self.flash(f"Could not open {d}: {ex}", error=True)
            return
        self.flash(f"Opened {os.path.abspath(d)}")

    def export_current(self):
        if self.export_job is not None:
            self.flash("An export is still running…")
            return
        if not self.gpu:
            self.flash("Nothing to export - select a model first", error=True)
            return
        e = self.gpu.entry
        if not self.gpu.model.surfaces:
            self.flash(f"Nothing to export - {e.name} has no renderable pieces", error=True)
            return
        try:
            out = self._export_dir(e)
        except Exception as ex:
            self.flash(f"Export failed: {ex}", error=True)
            return
        ntex = len({tex_key(t) for t in self.gpu.surf_tex if t is not None})
        job = dict(name=e.name, out=out, ntex=ntex, missing=len(self.missing),
                   result=None, error=None, finished=False)
        self.export_job = job

        skel, merge = self.setpanel.get("export_skeleton"), self.setpanel.get("merge_pieces")
        anims = [a for a, _s in self.animpanel.items] \
            if skel and self.setpanel.get("export_animations") and self.animpanel.gpu is self.gpu else []

        def run():
            try:
                with self.cat.lock:
                    job["result"] = export_model(self.cat.index, e.obj, out, skeleton=skel, merge=merge, anims=anims)
            except Exception as ex:          # noqa: BLE001
                job["error"] = ex
            job["finished"] = True

        self.flash(f"Exporting {e.name}…")
        threading.Thread(target=run, daemon=True).start()

    def _check_export(self):
        job = self.export_job
        if job is None or not job["finished"]:
            return
        self.export_job = None
        if job["error"] is not None:
            self.flash(f"Export failed: {job['error']}", error=True)
            print("export failed:", job["error"], file=sys.stderr)
            return
        path, info = job["result"]
        parts = [_n(info["meshes"], "mesh", "meshes"), _n(info["textures"], "texture PNG")]
        if info["joints"]:
            parts.append(_n(info["joints"], "bone"))
        if info.get("animations"):
            parts.append(_n(info["animations"], "animation"))
        msg = f"Exported {os.path.basename(path)} ({', '.join(parts)}) to {os.path.abspath(job['out'])}"
        if job["missing"]:
            msg += f"  ({job['missing']} referenced texture(s) not in the game folder)"
        self.flash(msg)
        print("exported:", os.path.abspath(path))

    # ------------------------------------------------------------- testing --
    def _auto_step(self):
        """--shots mode: render every model to PNG, then quit."""
        st = self.auto_shots
        if not self.cat.dups_done:
            return
        if "i" not in st:
            self._rebuild_visible()
            st["i"] = 0
            st["state"] = "select"
        if st["i"] >= len(self.vis_rows):
            print(f"saved {st['i']} screenshot(s) to {st['dir']}")
            pyglet.app.exit()
            return
        if st["state"] == "select":
            self.select(st["i"])
            st["state"] = "wait"
        elif st["state"] == "wait" and self.loading is None and self.gpu is not None:
            st["state"] = "shoot"
            st["frames"] = 0
        elif st["state"] == "shoot":
            st["frames"] += 1
            if st["frames"] >= 2:
                e = self.gpu.entry
                os.makedirs(st["dir"], exist_ok=True)
                self.screenshot(os.path.join(st["dir"], f"{st['i']:03d}_{safe(e.name)}.png"))
                st["i"] += 1
                st["state"] = "select"


LABELS = dict(tex="Textures", light="Lighting", col="Vertex colors", wire="Wireframe",
              skel="Skeleton", cull="Backface culling", nodup="Hide duplicate models",
              noempty="Hide empty models")

HELP_TEXT = """T - textures
L - lighting
V - vertex colors
W - wireframe
S - skeleton
C - backface culling
B - background color
D - hide duplicate models
N - hide empty models
A - animations ( [ ] step)
I - piece inspector
O - export settings
F - frame model
Space - auto-rotate
E - export (.glb + PNG)
F12 - screenshot"""


def pick_folder(initial):
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    r = tk.Tk()
    r.withdraw()
    d = filedialog.askdirectory(title='Select the game folder ("transformers dark of the moon")',
                                initialdir=initial or os.getcwd())
    r.destroy()
    return d or None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    shots = None
    if "--shots" in sys.argv:
        i = sys.argv.index("--shots")
        shots = {"dir": sys.argv[i + 1] if i + 1 < len(sys.argv) else "shots"}
        args = [a for a in args if a != shots["dir"]]
    cfg = {}
    try:
        cfg = json.load(open(CONFIG))
    except Exception:
        pass
    folder = args[0] if args else pick_folder(cfg.get("last_folder"))
    if not folder:
        if cfg.get("last_folder") and os.path.isdir(cfg["last_folder"]):
            folder = cfg["last_folder"]
        else:
            sys.exit("usage: dotm_model_viewer.py GAME_FOLDER")
    folder = os.path.abspath(folder)
    if os.path.isfile(folder):           # a .wii dropped on the launcher: use its folder
        folder = os.path.dirname(folder)
    if not os.path.isdir(folder):
        sys.exit(f"not a folder: {folder}")
    cfg["last_folder"] = folder
    try:
        json.dump(cfg, open(CONFIG, "w"))
    except Exception:
        pass
    Viewer(folder, shots)
    pyglet.app.run()


if __name__ == "__main__":
    main()
