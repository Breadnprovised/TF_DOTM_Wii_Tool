"""GameCube/Wii GX texture decoders -> RGBA numpy arrays (level 0 only).

DOTM stores its textures bottom row first (OpenGL convention: texture
coordinate v = 0 is the first row in memory), so decode() flips the result
to give an upright image (top row first), as it should look in a PNG."""
import numpy as np

BLOCK = {0: (8, 8, 4), 1: (8, 4, 8), 2: (8, 4, 8), 3: (4, 4, 16), 4: (4, 4, 16),
         5: (4, 4, 16), 6: (4, 4, 32), 14: (8, 8, 4)}  # bw, bh, bits/px
NAMES = {0: 'I4', 1: 'I8', 2: 'IA4', 3: 'IA8', 4: 'RGB565', 5: 'RGB5A3', 6: 'RGBA8', 14: 'CMPR'}

def tex_size(fmt, w, h):
    bw, bh, bpp = BLOCK[fmt]
    W = (w + bw - 1) // bw * bw; H = (h + bh - 1) // bh * bh
    return W * H * bpp // 8

def _565(v):
    v = v.astype(np.uint32)
    r = (v >> 11) & 31; g = (v >> 5) & 63; b = v & 31
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1).astype(np.uint8)

def _untile(px, w, h, bw, bh):
    """px: (nblocks, bh, bw, C) in block order -> (H, W, C) cropped."""
    W = (w + bw - 1) // bw; H = (h + bh - 1) // bh
    img = px.reshape(H, W, bh, bw, -1).transpose(0, 2, 1, 3, 4).reshape(H * bh, W * bw, -1)
    return img[:h, :w]

def decode(data, fmt, w, h):
    bw, bh, bpp = BLOCK[fmt]
    n = tex_size(fmt, w, h)
    b = np.frombuffer(data[:n], np.uint8)
    nb = ((w + bw - 1) // bw) * ((h + bh - 1) // bh)
    if fmt == 0:
        v = np.stack([b >> 4, b & 15], -1).reshape(nb, bh, bw) * 17
        rgba = np.stack([v, v, v, np.full_like(v, 255)], -1)
    elif fmt == 1:
        v = b.reshape(nb, bh, bw); rgba = np.stack([v, v, v, np.full_like(v, 255)], -1)
    elif fmt == 2:
        a = (b >> 4) * 17; i = (b & 15) * 17
        rgba = np.stack([i, i, i, a], -1).reshape(nb, bh, bw, 4)
    elif fmt == 3:
        p = b.reshape(-1, 2); rgba = np.stack([p[:, 1], p[:, 1], p[:, 1], p[:, 0]], -1).reshape(nb, bh, bw, 4)
    elif fmt == 4:
        v = b.view('>u2'); rgba = np.concatenate([_565(v), np.full(v.shape + (1,), 255, np.uint8)], -1).reshape(nb, bh, bw, 4)
    elif fmt == 5:
        v = b.view('>u2').astype(np.uint32); out = np.zeros(v.shape + (4,), np.uint8)
        op = (v & 0x8000) != 0
        r = (v >> 10) & 31; g = (v >> 5) & 31; bl = v & 31
        out[op, 0] = ((r << 3) | (r >> 2))[op]; out[op, 1] = ((g << 3) | (g >> 2))[op]
        out[op, 2] = ((bl << 3) | (bl >> 2))[op]; out[op, 3] = 255
        t = ~op; a = (v >> 12) & 7; r = (v >> 8) & 15; g = (v >> 4) & 15; bl = v & 15
        out[t, 0] = (r * 17)[t]; out[t, 1] = (g * 17)[t]; out[t, 2] = (bl * 17)[t]
        out[t, 3] = ((a << 5) | (a << 2) | (a >> 1))[t]
        rgba = out.reshape(nb, bh, bw, 4)
    elif fmt == 6:
        blk = b.reshape(nb, 2, 16, 2)  # [AR x16][GB x16]
        a = blk[:, 0, :, 0]; r = blk[:, 0, :, 1]; g = blk[:, 1, :, 0]; bl = blk[:, 1, :, 1]
        rgba = np.stack([r, g, bl, a], -1).reshape(nb, 4, 4, 4)
    elif fmt == 14:
        # 8x8 block = 2x2 DXT1 sub-blocks, each 8 bytes, big-endian colours, 2-bit idx MSB-first
        sb = b.reshape(-1, 8)
        c0 = sb[:, 0].astype(np.uint16) << 8 | sb[:, 1]; c1 = sb[:, 2].astype(np.uint16) << 8 | sb[:, 3]
        p0 = _565(c0).astype(np.int32); p1 = _565(c1).astype(np.int32)
        four = c0 > c1
        p2 = np.where(four[:, None], (2 * p0 + p1) // 3, (p0 + p1) // 2)
        p3 = np.where(four[:, None], (p0 + 2 * p1) // 3, 0)
        pal = np.stack([p0, p1, p2, p3], 1)                       # (n,4,3)
        alpha = np.full((len(sb), 4), 255, np.int32); alpha[~four, 3] = 0
        bits = sb[:, 4:8]
        idx = np.stack([(bits >> s) & 3 for s in (6, 4, 2, 0)], -1).reshape(-1, 16)  # (n,16)
        col = np.take_along_axis(pal, idx[:, :, None].repeat(3, 2), 1)
        al = np.take_along_axis(alpha, idx, 1)
        sub = np.concatenate([col, al[:, :, None]], -1).astype(np.uint8).reshape(-1, 2, 2, 4, 4, 4)
        rgba = sub.transpose(0, 1, 3, 2, 4, 5).reshape(nb, 8, 8, 4)
    else:
        raise ValueError(f'unsupported GX format {fmt}')
    return np.ascontiguousarray(_untile(rgba, w, h, bw, bh)[::-1])   # bottom-up -> upright
