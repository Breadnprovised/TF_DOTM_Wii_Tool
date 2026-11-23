"""glTF (.glb) / OBJ / PNG export shared by the viewer and the command-line tool."""
import json, os, re, struct
import numpy as np
from PIL import Image


def safe(name):
    return re.sub(r'[^\w.-]+', '_', name) or 'unnamed'


def save_texture_png(index, t, path):
    Image.fromarray(index.texture_rgba(t), 'RGBA').save(path)


def _surface_textures(index, obj, m, out_dir):
    """Resolve every surface's texture and write each one once as a PNG.
    Returns ([texture object or None per surface], {texture key: PNG file name})."""
    surf_tex = [index.surface_texture(obj, s) for s in m.surfaces]
    pngs, used = {}, set()
    for t in surf_tex:
        if t is None or (t.lu.path, t.off) in pngs:
            continue
        name = safe(t.name)
        if name.lower() in used:                     # two different textures, same name
            name = f'{name}_{t.id:08x}'
        used.add(name.lower())
        pngs[(t.lu.path, t.off)] = name + '.png'
        save_texture_png(index, t, os.path.join(out_dir, name + '.png'))
    return surf_tex, pngs


# ----------------------------------------------------------------------------
# glTF binary
# ----------------------------------------------------------------------------
class _Glb:
    def __init__(self):
        self.bin = bytearray()
        self.views, self.accessors = [], []

    def view(self, data, target=None):
        while len(self.bin) % 4:
            self.bin += b'\0'
        v = dict(buffer=0, byteOffset=len(self.bin), byteLength=len(data))
        if target:
            v['target'] = target
        self.bin += data
        self.views.append(v)
        return len(self.views) - 1

    def accessor(self, arr, ctype, typ, target=None, normalized=False, minmax=False):
        arr = np.ascontiguousarray(arr)
        a = dict(bufferView=self.view(arr.tobytes(), target), componentType=ctype,
                 count=int(arr.shape[0]), type=typ)
        if normalized:
            a['normalized'] = True
        if minmax:
            a['min'] = [float(x) for x in arr.reshape(len(arr), -1).min(0)]
            a['max'] = [float(x) for x in arr.reshape(len(arr), -1).max(0)]
        self.accessors.append(a)
        return len(self.accessors) - 1


FLOAT, UBYTE, USHORT, UINT = 5126, 5121, 5123, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963


def quat_from_rows(R):
    """Unit quaternion (x, y, z, w) of a row-vector rotation matrix (p' = p . R),
    i.e. of the column-vector matrix R^T that glTF uses."""
    m = np.asarray(R, np.float64).T
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / S, (m[0, 2] - m[2, 0]) / S, (m[1, 0] - m[0, 1]) / S, 0.25 * S]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * S, (m[0, 1] + m[1, 0]) / S, (m[0, 2] + m[2, 0]) / S, (m[2, 1] - m[1, 2]) / S]
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / S, 0.25 * S, (m[1, 2] + m[2, 1]) / S, (m[0, 2] - m[2, 0]) / S]
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / S, (m[1, 2] + m[2, 1]) / S, 0.25 * S, (m[1, 0] - m[0, 1]) / S]
    q = np.array(q)
    return q / np.linalg.norm(q)


def trs(L):
    """Translation, rotation quaternion and scale of a local joint matrix
    (row-vector: diag(scale) . rotation, translation in row 3)."""
    sc = np.linalg.norm(L[:3, :3], axis=1)
    R = L[:3, :3] / np.where(sc > 1e-12, sc, 1)[:, None]
    return L[3, :3].copy(), quat_from_rows(R), sc


FPS = 30


def _write_animations(g, gl, m, j0, anims):
    """One glTF animation per game animation, sampled at 30 fps: translation,
    rotation and scale of every joint the animation drives."""
    names = {}
    written = 0
    for a in anims:
        nframes = max(1, int(round(a.duration * FPS))) + 1
        times = np.minimum(np.arange(nframes, dtype=np.float64) / FPS, a.duration).astype(np.float32)
        joints = sorted(j for j in a.joints if j < len(m.joints))
        if not joints:
            continue
        T = np.zeros((nframes, len(joints), 3), np.float32)
        Q = np.zeros((nframes, len(joints), 4), np.float32)
        S = np.zeros((nframes, len(joints), 3), np.float32)
        for f, t in enumerate(times):
            loc = a.local_matrices(m, float(t))
            for k, j in enumerate(joints):
                T[f, k], Q[f, k], S[f, k] = trs(loc[j])
        for k in range(len(joints)):                 # shortest path between keys
            for f in range(1, nframes):
                if np.dot(Q[f, k], Q[f - 1, k]) < 0:
                    Q[f, k] = -Q[f, k]
        inp = g.accessor(times, FLOAT, 'SCALAR', minmax=True)
        samplers, channels = [], []
        for k, j in enumerate(joints):
            for path, data, typ in (('translation', T[:, k], 'VEC3'), ('rotation', Q[:, k], 'VEC4'), ('scale', S[:, k], 'VEC3')):
                samplers.append(dict(input=inp, output=g.accessor(np.ascontiguousarray(data), FLOAT, typ),
                                     interpolation='LINEAR'))
                channels.append(dict(sampler=len(samplers) - 1, target=dict(node=j0 + j, path=path)))
        name = a.name
        if name in names:                            # same name, different animation
            names[name] += 1
            name = f'{name} ({names[name]})'
        else:
            names[name] = 1
        gl.setdefault('animations', []).append(dict(name=name, samplers=samplers, channels=channels))
        written += 1
    return written


def export_model(index, obj, out_dir, skeleton=True, merge=True, anims=()):
    """Writes <name>.glb plus the PNGs of its textures into out_dir.

    With skeleton=True (and a model that has joints) the file contains the
    joint hierarchy in its bind pose and every piece is skinned to the
    joint it is attached to with weight 1, which is how the game draws it.
    `anims` (anim.Anim objects that fit the model) are written as glTF
    animations, sampled at 30 fps, when the skeleton is exported.
    With merge=True the pieces are one mesh (one primitive per material);
    otherwise every piece is its own mesh, named pieceNNN_jointJ.
    The .glb refers to the PNGs by file name. Returns (glb path, info dict)."""
    m = index.load_model(obj)
    os.makedirs(out_dir, exist_ok=True)
    base = safe(m.name)
    surf_tex, pngs = _surface_textures(index, obj, m, out_dir)
    g = _Glb()
    gl = dict(asset=dict(version='2.0', generator='DOTM Wii Viewer'), scene=0,
              scenes=[dict(name=m.name, nodes=[0])], nodes=[dict(name=m.name, children=[])],
              meshes=[], materials=[])

    # materials: one per texture (alpha cut-out like the game), one untextured
    mat_of = {}
    if pngs:
        gl['samplers'] = [dict(magFilter=9729, minFilter=9987, wrapS=10497, wrapT=10497)]
        gl['images'], gl['textures'] = [], []
    for t in surf_tex:
        if t is None or (t.lu.path, t.off) in mat_of:
            continue
        k = (t.lu.path, t.off)
        gl['images'].append(dict(uri=pngs[k], name=t.name))
        gl['textures'].append(dict(sampler=0, source=len(gl['images']) - 1))
        mat = dict(name=t.name, doubleSided=True,
                   pbrMetallicRoughness=dict(baseColorTexture=dict(index=len(gl['textures']) - 1),
                                             metallicFactor=0.0, roughnessFactor=1.0))
        if index.texture_rgba(t)[..., 3].min() < 255:
            mat.update(alphaMode='MASK', alphaCutoff=0.35)
        gl['materials'].append(mat)
        mat_of[k] = len(gl['materials']) - 1

    def untextured():
        if None not in mat_of:
            gl['materials'].append(dict(name='untextured', doubleSided=True, pbrMetallicRoughness=dict(
                baseColorFactor=[0.75, 0.75, 0.78, 1.0], metallicFactor=0.0, roughnessFactor=1.0)))
            mat_of[None] = len(gl['materials']) - 1
        return mat_of[None]

    # skeleton
    nj = len(m.joints)
    skinned = skeleton and nj > 0
    if skinned:
        j0 = len(gl['nodes'])
        for k, L in enumerate(m.joints):
            node = dict(name=f'joint_{k:03d}')
            # translation / rotation / scale (not a matrix, so animations can drive it)
            t, q, sc = trs(np.asarray(L, np.float64))
            if np.any(np.abs(t) > 0):
                node['translation'] = [float(x) for x in t]
            if abs(q[3] - 1) > 1e-9:
                node['rotation'] = [float(x) for x in q]
            if np.any(np.abs(sc - 1) > 1e-9):
                node['scale'] = [float(x) for x in sc]
            gl['nodes'].append(node)
        for k, p in enumerate(m.parents):
            if 0 <= p < nj and p != k:
                gl['nodes'][j0 + p].setdefault('children', []).append(j0 + k)
            else:
                gl['nodes'][0]['children'].append(j0 + k)
        ibm = np.array([np.linalg.inv(W) for W in m.world], np.float64)
        ibm[:, :, 3] = (0, 0, 0, 1)                  # exact affine row (no rounding noise)
        ibm = ibm.reshape(nj, 16).astype(np.float32)
        gl['skins'] = [dict(name=f'{m.name}_skeleton', joints=list(range(j0, j0 + nj)),
                            inverseBindMatrices=g.accessor(ibm, FLOAT, 'MAT4'))]
        roots = [j0 + k for k, p in enumerate(m.parents) if not (0 <= p < nj) or p == k]
        if len(roots) == 1:
            gl['skins'][0]['skeleton'] = roots[0]

    def primitive(pieces, skin):
        """One glTF primitive from pieces that share a material and vertex layout."""
        P, N, U, C, J, I = [], [], [], [], [], []
        base = 0
        for s in pieces:
            P.append(s.pos.astype(np.float32))
            nrm = s.nrm.astype(np.float32)
            ln = np.linalg.norm(nrm, axis=1, keepdims=True)
            N.append(np.where(ln > 1e-6, nrm / np.where(ln > 1e-6, ln, 1), np.float32([0, 1, 0])))
            uv = s.uv.astype(np.float32).copy()
            uv[:, 1] = 1.0 - uv[:, 1]                # game v = 0 is the bottom row; glTF's is the top
            U.append(uv)
            C.append(s.col.astype(np.uint8))
            J.append(np.full(len(s.pos), max(s.joint, 0)))
            I.append(s.tris.reshape(-1).astype(np.int64) + base)
            base += len(s.pos)
        s0 = pieces[0]
        attrs = dict(POSITION=g.accessor(np.concatenate(P), FLOAT, 'VEC3', ARRAY_BUFFER, minmax=True))
        if s0.has_nrm:
            attrs['NORMAL'] = g.accessor(np.concatenate(N).astype(np.float32), FLOAT, 'VEC3', ARRAY_BUFFER)
        if s0.has_uv:
            attrs['TEXCOORD_0'] = g.accessor(np.concatenate(U), FLOAT, 'VEC2', ARRAY_BUFFER)
        attrs['COLOR_0'] = g.accessor(np.concatenate(C), UBYTE, 'VEC4', ARRAY_BUFFER, normalized=True)
        if skin:
            # rigid skinning: every vertex follows its piece's joint with weight 1
            jt = np.zeros((base, 4), np.uint8 if nj < 256 else np.uint16)
            jt[:, 0] = np.concatenate(J)
            wt = np.zeros((base, 4), np.float32)
            wt[:, 0] = 1.0
            attrs['JOINTS_0'] = g.accessor(jt, UBYTE if nj < 256 else USHORT, 'VEC4', ARRAY_BUFFER)
            attrs['WEIGHTS_0'] = g.accessor(wt, FLOAT, 'VEC4', ARRAY_BUFFER)
        idx = np.concatenate(I)
        idx = idx.astype(np.uint16) if base < 65536 else idx.astype(np.uint32)
        t = tex_of[id(s0)]
        return dict(attributes=attrs, mode=4,
                    indices=g.accessor(idx, USHORT if idx.dtype == np.uint16 else UINT, 'SCALAR',
                                       ELEMENT_ARRAY_BUFFER),
                    material=mat_of[(t.lu.path, t.off)] if t is not None else untextured())

    def add_node(name, prims, skin):
        gl['meshes'].append(dict(name=name, primitives=prims))
        node = dict(name=name, mesh=len(gl['meshes']) - 1)
        if skin:
            node['skin'] = 0
        gl['nodes'].append(node)
        # skinned meshes sit at the scene root (their own transform is ignored)
        (gl['scenes'][0]['nodes'] if skin else gl['nodes'][0]['children']).append(len(gl['nodes']) - 1)

    # a texture can't be mapped without UVs
    tex_of = {id(s): (t if s.has_uv else None) for s, t in zip(m.surfaces, surf_tex)}
    pieces = [s for s in m.surfaces if len(s.tris)]
    is_skinned = lambda s: skinned and 0 <= s.joint < nj
    nmesh = 0
    if merge:
        # one mesh for the whole model (plus one for pieces without a joint, if any),
        # with one primitive per material and vertex layout
        for skin, name in ((True, m.name), (False, m.name + '_static')):
            group = [s for s in pieces if is_skinned(s) == skin]
            if not group:
                continue
            by_mat = {}
            for s in group:
                t = tex_of[id(s)]
                key = ((t.lu.path, t.off) if t is not None else None, s.has_uv, s.has_nrm)
                by_mat.setdefault(key, []).append(s)
            add_node(name, [primitive(ps, skin) for ps in by_mat.values()], skin)
            nmesh += 1
    else:
        # one mesh per piece, named like the viewer's pieces
        for s in pieces:
            add_node(f'piece{s.index:03d}_joint{s.joint}', [primitive([s], is_skinned(s))], is_skinned(s))
            nmesh += 1
    if not gl['materials']:
        del gl['materials']
    nanim = _write_animations(g, gl, m, j0, anims) if skinned and anims else 0
    if not gl['meshes']:
        del gl['meshes']
    gl['bufferViews'], gl['accessors'] = g.views, g.accessors
    while len(g.bin) % 4:
        g.bin += b'\0'
    gl['buffers'] = [dict(byteLength=len(g.bin))]
    js = json.dumps(gl, separators=(',', ':')).encode()
    js += b' ' * ((-len(js)) % 4)
    path = os.path.join(out_dir, base + '.glb')
    with open(path, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(g.bin)))
        f.write(struct.pack('<II', len(js), 0x4E4F534A) + js)
        f.write(struct.pack('<II', len(g.bin), 0x004E4942) + bytes(g.bin))
    return path, dict(meshes=nmesh, textures=len(pngs), joints=nj if skinned else 0, animations=nanim)


# ----------------------------------------------------------------------------
# Wavefront OBJ (no skeleton)
# ----------------------------------------------------------------------------
def export_model_obj(index, obj, out_dir):
    """Writes <name>.obj, <name>.mtl and the PNGs of its textures into out_dir. Returns obj path."""
    m = index.load_model(obj)
    os.makedirs(out_dir, exist_ok=True)
    base = safe(m.name)
    surf_tex, pngs = _surface_textures(index, obj, m, out_dir)
    mtl_of = {}
    with open(os.path.join(out_dir, base + '.mtl'), 'w') as f:
        for t in surf_tex:
            k = (t.lu.path, t.off) if t else None
            if k in mtl_of:
                continue
            mtl_of[k] = os.path.splitext(pngs[k])[0] if t else 'untextured'
            f.write(f'newmtl {mtl_of[k]}\nKd 1 1 1\n' + (f'map_Kd {pngs[k]}\n' if t else '') + '\n')
    objp = os.path.join(out_dir, base + '.obj')
    with open(objp, 'w') as f:
        f.write(f'# {m.name} (from {obj.lu.path})\nmtllib {base}.mtl\n')
        vb = 1
        for s, t in zip(m.surfaces, surf_tex):
            f.write(f'o piece{s.index:03d}_joint{s.joint}\nusemtl {mtl_of[(t.lu.path, t.off) if t else None]}\n')
            f.write(''.join('v %.6f %.6f %.6f\n' % tuple(p) for p in s.pos))
            f.write(''.join('vn %.6f %.6f %.6f\n' % tuple(n) for n in s.nrm))
            # OBJ's v = 0 is the bottom row of the image, like the game's
            f.write(''.join('vt %.6f %.6f\n' % (u, v) for u, v in s.uv))
            for a, b, c in (s.tris + vb):
                f.write(f'f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n')
            vb += len(s.pos)
    return objp


def export_all(index, out_root, progress=None, textures=True, fmt='glb', skeleton=True, merge=True,
               animations=True):
    """Export every model (and optionally every texture) grouped by load unit."""
    models = [m for lu in index.lus for m in lu.models]
    all_anims, model_names = [], sorted({m.name for m in models})
    if animations and skeleton and fmt != 'obj':
        import anim as anim_mod
        for lu in index.lus:
            for o in lu.objects:
                if o.type == anim_mod.T_ANIM:
                    try:
                        all_anims.append(anim_mod.Anim(o))
                    except Exception:
                        pass
    texs = [t for lu in index.lus for t in lu.textures] if textures else []
    total = len(models) + len(texs)
    done, errors = 0, []
    for m in models:
        if progress and progress(done, total, f'{m.lu.name}/{m.name}') is False:
            return errors
        try:
            if index.load_model(m).surfaces:
                d = os.path.join(out_root, safe(m.lu.name), 'models', safe(m.name))
                if fmt == 'obj':
                    export_model_obj(index, m, d)
                else:
                    fits = []
                    if all_anims:
                        import anim as anim_mod
                        fits = [a for a, _s in anim_mod.matching_anims(index.load_model(m), m.name, all_anims, model_names)]
                    export_model(index, m, d, skeleton=skeleton, merge=merge, anims=fits)
        except Exception as e:
            errors.append(f'{m.lu.path} {m.name}: {e}')
        done += 1
    for t in texs:
        if progress and progress(done, total, f'{t.lu.name}/{t.name}') is False:
            return errors
        try:
            d = os.path.join(out_root, safe(t.lu.name), 'textures')
            os.makedirs(d, exist_ok=True)
            save_texture_png(index, t, os.path.join(d, safe(t.name) + '.png'))
        except Exception as e:
            errors.append(f'{t.lu.path} {t.name}: {e}')
        done += 1
    return errors
