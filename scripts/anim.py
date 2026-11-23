"""Animation reader for Transformers: Dark of the Moon (Wii) - object type 79.

An animation drives joints by index: every joint has 9 channels - scale
x/y/z (0-2), translation x/y/z (3-5) and rotation x/y/z (6-8, Euler angles in
radians, X then Y then Z). All values are stored multiplied by 32.

    header  +0x30 f32 duration (s)
            +0x3C u16 animated tracks, u16 constant translations
            +0x40 u16 constant rotations, u16 constant scales
            +0x48 offsets: track table, constant translations, rotations,
                  scales, events (0xFFFFFFFF = none)
    constant record (8 bytes): 3 x half x/y/z, u8 joint, u8 joint slot
    track: u16 (joint << 8 | type), u8 channel, u8 0, f32 length of the
           last segment, u32 (slot << 24 | key count), then keys:
           f32 time + coefficients of the segment's polynomial in u = 0..1
           (type 0x88: 4 halves a,b,c,d -> a u^3 + b u^2 + c u + d;
            0x80: the same as 4 floats; 0x68: 2 halves c,d -> c u + d)
Channels other than 0-8 (e.g. 22, used by effects) are skipped.

A joint's local matrix (row-vector, like the model's bind matrices) is
diag(scale) . Rx . Ry . Rz with the translation in the last row; channels an
animation doesn't set keep the bind pose's values. (Verified against the
rotated / scaled bind poses of the game's props and effects.)
"""
import re
import struct

import numpy as np

T_ANIM = 79
SKELETAL_TYPES = {0x88: (4, ">e"), 0x80: (4, ">f"), 0x68: (2, ">e")}   # coefficients, value type


class AnimError(Exception):
    pass


class Track:
    __slots__ = ("joint", "ch", "keys", "last", "times", "coefs")

    def value(self, t):
        """Value (unscaled, i.e. x32) at time t: hold before the first key,
        polynomial per segment, the last segment lasts `last` seconds."""
        times, coefs = self.times, self.coefs
        if t <= times[0]:
            return float(coefs[0][-1])
        i = int(np.searchsorted(times, t, side="right")) - 1
        t0 = times[i]
        t1 = times[i + 1] if i + 1 < len(times) else t0 + self.last
        u = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
        u = 0.0 if u < 0 else (1.0 if u > 1 else u)
        r = 0.0
        for c in coefs[i]:
            r = r * u + c
        return float(r)


class Anim:
    """A parsed type-79 animation."""

    def __init__(self, obj):
        self.obj = obj
        self.name = obj.name
        d = obj.read()
        import zlib
        self.signature = zlib.crc32(d[0x10:])        # ignoring the object ID
        if len(d) < 0x60:
            raise AnimError("too short")
        self.duration = struct.unpack_from(">f", d, 0x30)[0]
        w = struct.unpack_from(">12I", d, 0x34)
        n_tracks, n_t = w[2] >> 16, w[2] & 0xFFFF
        n_r, n_s = w[3] >> 16, w[3] & 0xFFFF
        o_tracks, o_t, o_r, o_s = w[5:9]
        self.T = self._consts(d, o_t, n_t)
        self.R = self._consts(d, o_r, n_r)
        self.S = self._consts(d, o_s, n_s)
        self.tracks = []
        if n_tracks and o_tracks != 0xFFFFFFFF:
            table = struct.unpack_from(">%dI" % n_tracks, d, o_tracks)
            for a in table:
                try:
                    tr = self._track(d, a)
                except (struct.error, ValueError):
                    continue
                if tr is not None:
                    self.tracks.append(tr)
        if not np.isfinite(self.duration) or self.duration <= 0:
            self.duration = max([0.0] + [tr.times[-1] + tr.last for tr in self.tracks]) or 1 / 30
        self.joints = set(self.T) | set(self.R) | set(self.S) | {tr.joint for tr in self.tracks}
        self.by_joint = {}
        for tr in self.tracks:
            self.by_joint.setdefault(tr.joint, []).append(tr)

    @staticmethod
    def _consts(d, off, n):
        out = {}
        if off == 0xFFFFFFFF or not n:
            return out
        for i in range(n):
            x, y, z, j, _slot = struct.unpack_from(">3eBB", d, off + 8 * i)
            out[j] = np.array([x, y, z], np.float64) / 32
        return out

    @staticmethod
    def _track(d, a):
        fl, ch, _x, last, cnt = struct.unpack_from(">HBBfI", d, a)
        typ = fl & 0xFF
        if typ not in SKELETAL_TYPES or ch > 8:
            return None
        ncoef, fmt = SKELETAL_TYPES[typ]
        size = 2 if fmt == ">e" else 4
        n = cnt & 0xFFFFFF
        if not 0 < n < 100000:
            return None
        per = 4 + ncoef * size
        if a + 12 + n * per > len(d):
            raise ValueError("track runs past the end")
        tr = Track()
        tr.joint, tr.ch, tr.last = fl >> 8, ch, last if np.isfinite(last) else 0.0
        times, coefs = [], []
        for k in range(n):
            p = a + 12 + k * per
            times.append(struct.unpack_from(">f", d, p)[0])
            coefs.append([float(c) for c in struct.unpack_from(">%d%s" % (ncoef, fmt[1]), d, p + 4)])
        tr.times = np.array(times, np.float64)
        tr.coefs = coefs
        if not np.all(np.isfinite(tr.times)):
            return None
        return tr

    # ------------------------------------------------------------------
    def local_matrices(self, model, t):
        """Local joint matrices (n x 4 x 4, row-vector) of `model` at time t."""
        bind = np.asarray(model.joints, np.float64)
        out = bind.copy()
        for j in self.joints:
            if j >= len(bind):
                continue
            B = bind[j]
            b3 = B[:3, :3]
            bs = np.linalg.norm(b3, axis=1)
            trs = self.by_joint.get(j, ())
            s = self.S[j].copy() if j in self.S else bs.copy()
            tr = self.T[j].copy() if j in self.T else B[3, :3].copy()
            has_rot = j in self.R or any(6 <= k.ch <= 8 for k in trs)
            e = self.R[j].copy() if j in self.R else np.zeros(3)
            for k in trs:
                v = k.value(t) / 32
                if k.ch < 3:
                    s[k.ch] = v
                elif k.ch < 6:
                    tr[k.ch - 3] = v
                else:
                    e[k.ch - 6] = v
            if has_rot:
                rot = euler_matrix(e)
            else:
                rot = b3 / np.where(bs > 1e-12, bs, 1)[:, None]
            M = np.eye(4)
            M[:3, :3] = s[:, None] * rot
            M[3, :3] = tr
            out[j] = M
        return out


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def euler_matrix(e):
    """Row-vector rotation (p' = p . M) from X, Y, Z angles in radians."""
    return _rx(e[0]) @ _ry(e[1]) @ _rz(e[2])


def world_matrices(model, local):
    """Model-space joint matrices from local ones (parents come first)."""
    world = np.array(local, np.float64)
    for j, p in enumerate(model.parents):
        if 0 <= p < len(world) and p != j:
            world[j] = world[j] @ world[p]
    return world


# ----------------------------------------------------------------------------
# which animations go with which model
# ----------------------------------------------------------------------------
def _compact(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _base(model_name):
    """Model name without _ref and the a_/d_ faction prefix."""
    n = re.sub(r"_ref$", "", model_name.lower())
    return re.sub(r"^[ad]_", "", n)


COMMON_WORDS = {"generic", "secondary", "weapon", "intro", "outro", "spawn", "stand"}


def _named_after(base, a):
    """Is the animation (compacted name `a`) named after a model base name?
    A trailing number must match (energoncube02 is not energoncube01)."""
    if not base:
        return False
    if base in a:
        return True
    if base[-1].isdigit():
        return False
    # names are often cut short (beamoflighttut -> beamoflight_...)
    n = max(8, len(base) - 3)
    return len(base) >= 8 and base[:n] in a


def name_relation(model_name, anim_name):
    """2: the animation is named after the model; 1: they share a word; 0: unrelated."""
    base, a = _compact(_base(model_name)), _compact(anim_name)
    if _named_after(base, a):
        return 2
    if base.startswith("camera") and "camera" in a:
        return 2
    for t in re.split(r"[^a-z]+", _base(model_name)):
        if len(t) >= 4 and t not in COMMON_WORDS and (t in a or (len(t) >= 6 and t[:max(6, len(t) - 5)] in a)):
            return 1
    return 0


def _more_specific_owner(model_name, anim_name, all_model_names):
    """True when another model with a longer name is what the animation is
    named after (megatronsecondaryweap_stop belongs to
    megatronsecondaryweap_ref, not d_megatron_ref)."""
    a = _compact(anim_name)
    mine = _compact(_base(model_name))
    for other in all_model_names:
        ob = _compact(_base(other))
        if len(ob) > len(mine) and (ob in a or (ob.rstrip("0123456789") or ob) in a):
            return True
    return False


def compatibility(anim, model):
    """0..1: share of an animation's constant translations equal to the
    model's bind translations (0.5 when it has none); 0 when it drives
    joints the model doesn't have."""
    nj = len(model.joints)
    if not anim.joints or max(anim.joints) >= nj:
        return 0.0
    if not anim.T:
        return 0.5
    bind = np.asarray(model.joints)
    ok = 0
    for j, v in anim.T.items():
        b = bind[j][3, :3]
        ok += bool(np.all(np.abs(v - b) <= 0.01 + 0.01 * np.abs(b)))
    return ok / len(anim.T)


def matching_anims(model, model_name, anims, all_model_names=()):
    """[(anim, score)] that go with a model, best first:
    - named after the model (militaryjeep01_break for militaryjeep01_ref) and
      only driving joints the model has;
    - sharing a word with it (map01m01_intro_optimus for optimusprime_ref),
      fitting its skeleton (most constant translations equal its bind pose)
      and driving at least 15% of its joints;
    - otherwise only for big skeletons it fits almost exactly and drives
      at least a quarter of.
    Copies of the same animation stored in several files are listed once."""
    nj = len(model.joints)
    out, seen = [], set()
    for a in anims:
        if not a.joints or max(a.joints) >= nj:
            continue
        rel = name_relation(model_name, a.name)
        if rel and _more_specific_owner(model_name, a.name, all_model_names):
            continue
        sc = compatibility(a, model)
        cover = len(a.joints) / max(nj, 1)
        if rel == 2:
            ok = True
        elif rel == 1:
            ok = bool(a.T) and sc >= 0.5 and (cover >= 0.15 or nj <= 6)
        else:
            ok = nj >= 12 and sc >= 0.9 and cover >= 0.25
        if not ok:
            continue
        key = (a.name, a.obj.size, a.signature)
        if key in seen:
            continue
        seen.add(key)
        out.append((a, rel + sc))
    out.sort(key=lambda x: x[0].name.lower())
    return out
