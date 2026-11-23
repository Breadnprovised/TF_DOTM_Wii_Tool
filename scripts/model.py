"""Type-91 model parser (numpy). Works on the bytes of a single object (offsets are
object-relative). See FORMAT.md."""
import struct
import numpy as np

# vertex attributes, in stored order: (flag bit, field, dtype, count). A surface's attribute
# flags say which are present; the vertex is the present ones packed together (0xF = all four,
# 0x24 bytes) and the display list has one u16 index per present attribute.
ATTRS = ((1, 'p', '>f4', 3), (2, 'n', '>f4', 3), (4, 'c', 'u1', 4), (8, 'uv', '>f4', 2))


def vertex_dtype(flags):
    """numpy dtype of one vertex for these attribute flags (None if unsupported)."""
    if not flags & 1 or flags & ~0xF:
        return None
    return np.dtype([(f, t, n) for bit, f, t, n in ATTRS if flags & bit])


def _valid_record(d, o):
    flags, stride = _u32(d, o + 4), _u32(d, o + 8)
    dt = vertex_dtype(flags)
    return dt is not None and dt.itemsize == stride
PRIMS = (0x80, 0x90, 0x98, 0xA0, 0xA8, 0xB0, 0xB8)


def _u32(d, o):
    return struct.unpack_from('>I', d, o)[0]


def find_surface_table(d):
    size = len(d)
    for o in range(0x50, size - 12, 4):
        cnt, off = struct.unpack_from('>II', d, o)
        if 0 < cnt < 2000 and 0 < off and off % 4 == 0 and off + 0x44 <= size:
            if _valid_record(d, off):
                if all(off + k * 0x44 + 12 <= size and _valid_record(d, off + k * 0x44)
                       for k in range(min(cnt, 4))):
                    return cnt, off
    return None


def parse_display_list(dl, nattr=4):
    """Triangles (position indices) of a GX display list with nattr u16 indices per vertex."""
    out = []
    i, n_dl = 0, len(dl)
    while i + 3 <= n_dl:
        op = dl[i]
        if op == 0:
            i += 1
            continue
        prim = op & 0xF8
        if prim not in PRIMS:
            break
        n = struct.unpack_from('>H', dl, i + 1)[0]
        i += 3
        if i + n * 2 * nattr > n_dl:
            break
        idx = np.frombuffer(dl, '>u2', n * nattr, i).reshape(n, nattr)[:, 0].astype(np.int64)
        i += n * 2 * nattr
        if prim == 0x90:
            m = n // 3 * 3
            out.append(idx[:m].reshape(-1, 3))
        elif prim == 0x98 and n >= 3:
            a, b, c = idx[:-2], idx[1:-1], idx[2:]
            odd = (np.arange(n - 2) & 1).astype(bool)
            out.append(np.stack([np.where(odd, b, a), np.where(odd, a, b), c], 1))
        elif prim == 0xA0 and n >= 3:
            out.append(np.stack([np.full(n - 2, idx[0]), idx[1:-1], idx[2:]], 1))
    return np.concatenate(out) if out else np.zeros((0, 3), np.int64)


class Surface:
    __slots__ = ('index', 'material', 'joint', 'pos', 'nrm', 'col', 'uv', 'tris', 'texrefs',
                 'has_uv', 'has_nrm')


class Model:
    def __init__(self, data, name=''):
        d = data
        size = len(d)
        self.name = name or d[0x10:0x30].split(b'\0')[0].decode('latin-1')
        self.sphere = struct.unpack_from('>4f', d, 0x40) if size >= 0x50 else (0, 0, 0, 1)
        self.joints, self.parents = [], []
        self.surfaces = []
        self.warnings = []
        j0, j1 = (_u32(d, 0x50), _u32(d, 0x54)) if size >= 0x58 else (0, 0)
        if 0 < j0 < j1 <= size and (j1 - j0) % 0x50 == 0:
            for k in range((j1 - j0) // 0x50):
                o = j0 + k * 0x50
                self.joints.append(np.array(struct.unpack_from('>16f', d, o), np.float64).reshape(4, 4))
                self.parents.append(struct.unpack_from('>h', d, o + 0x40)[0])
        else:
            j1 = 0
        self.world = self._world_matrices()
        st = find_surface_table(d)
        if not st:
            return
        cnt, off = st
        surf_joint = self._surface_joints(d, j1, cnt, off)
        for k in range(cnt):
            r = struct.unpack_from('>17I', d, off + k * 0x44)
            mat, flags, stride, nv, voff, dlsz, dloff = r[0], r[1], r[2], r[3], r[4], r[13], r[14]
            dt = vertex_dtype(flags)
            if dt is None or dt.itemsize != stride:
                self.warnings.append(f'surface {k}: unsupported vertex format '
                                     f'(flags 0x{flags:x}, stride 0x{stride:x}) skipped')
                continue
            if voff + nv * stride > size or dloff + dlsz > size:
                self.warnings.append(f'surface {k}: out-of-range data skipped')
                continue
            v = np.frombuffer(d, dt, nv, voff)
            names = dt.names
            s = Surface()
            s.index, s.material, s.joint = k, mat, surf_joint[k]
            p = v['p'].astype(np.float64)
            n = v['n'].astype(np.float64) if 'n' in names else np.zeros((nv, 3))
            if 0 <= s.joint < len(self.world):
                M = self.world[s.joint]
                p = p @ M[:3, :3] + M[3, :3]
                n = n @ M[:3, :3]
                ln = np.linalg.norm(n, axis=1, keepdims=True)
                n = n / np.where(ln == 0, 1, ln)
            s.pos, s.nrm = p.astype(np.float32), n.astype(np.float32)
            s.col = v['c'].copy() if 'c' in names else np.full((nv, 4), 255, np.uint8)
            s.uv = v['uv'].astype(np.float32) if 'uv' in names else np.zeros((nv, 2), np.float32)
            s.has_uv, s.has_nrm = 'uv' in names, 'n' in names
            tris = parse_display_list(d[dloff:dloff + dlsz], len(names))
            s.tris = tris[(tris < nv).all(1)].astype(np.uint32)
            s.texrefs = []
            if mat + 0x90 <= size:
                for q in range(0, 0x90, 4):
                    val = _u32(d, mat + q)
                    if val >> 24 == 0x22:
                        s.texrefs.append(val)
            self.surfaces.append(s)

    def _world_matrices(self):
        W = [None] * len(self.joints)

        def get(i, depth=0):
            if W[i] is None:
                p = self.parents[i]
                if 0 <= p < len(self.joints) and p != i and depth < 256:
                    W[i] = self.joints[i] @ get(p, depth + 1)
                else:
                    W[i] = self.joints[i]
            return W[i]
        for i in range(len(self.joints)):
            get(i)
        return W

    @staticmethod
    def _surface_joints(d, j1, cnt, off):
        """{1, ngroups, joint_list_off, next_off}; next_off+0x20 -> ngroups x {first, count}."""
        sj = [-1] * cnt
        if not j1:
            return sj
        for o in range(j1, off - 12, 4):
            a, ng, lo, nx = struct.unpack_from('>4I', d, o)
            if a == 1 and 0 < ng <= cnt and j1 <= lo < nx <= off:
                joints = struct.unpack_from('>%dH' % ng, d, lo)
                ok = nx + 0x24 <= len(d) and _u32(d, nx + 4) == cnt and _u32(d, nx + 8) == off \
                    and _u32(d, nx + 12) == ng
                go = _u32(d, nx + 0x20) if ok else 0
                if ok and 0 < go and go + ng * 8 <= len(d):
                    for g in range(ng):
                        first, n = struct.unpack_from('>2I', d, go + g * 8)
                        for s in range(first, min(first + n, cnt)):
                            sj[s] = joints[g]
                elif ng == cnt:
                    sj = list(joints)
                break
        return sj

    @property
    def triangle_count(self):
        return sum(len(s.tris) for s in self.surfaces)

    @property
    def vertex_count(self):
        return sum(len(s.pos) for s in self.surfaces)

    def bounds(self):
        if not self.surfaces:
            return np.zeros(3), np.ones(3)
        P = np.concatenate([s.pos for s in self.surfaces])
        return P.min(0), P.max(0)

    def joint_positions(self):
        return np.array([W[3, :3] for W in self.world]) if self.world else np.zeros((0, 3))
