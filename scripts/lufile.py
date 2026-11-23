"""Lazy reader for Chimera load-unit (.wii) files.

Only the header, the type tables and the small per-object headers are read when a file
is opened, so a whole game folder can be indexed quickly. Object payloads are read on
demand with LUFile.read_object(). See FORMAT.md for the layout.
"""
import os, struct

MAGIC = 0x03050548
NSLOTS = 110
T_TEXTURE, T_MODEL = 34, 91
GX_FORMATS = {0: 'I4', 1: 'I8', 2: 'IA4', 3: 'IA8', 4: 'RGB565', 5: 'RGB5A3', 6: 'RGBA8', 14: 'CMPR'}
TYPE_NAMES = {19: 'LU root', 23: 'FX', 34: 'Texture', 40: 'Proxy', 41: 'GameObjectDef', 42: 'Logic',
              43: 'Logic lib', 67: 'FX/sound', 69: 'FX', 79: 'Anim state', 89: 'FX event',
              90: 'FX event', 91: 'Model', 95: 'Matrix array', 99: 'List'}


class LUError(Exception):
    pass


def cstr(b):
    return b.split(b'\0', 1)[0].decode('latin-1')


class Obj:
    __slots__ = ('lu', 'type', 'id', 'off', 'size', 'name', 'tex')

    def __init__(self, lu, type_, id_, off, size):
        self.lu, self.type, self.id, self.off, self.size = lu, type_, id_, off, size
        self.name = ''
        self.tex = None   # dict(w, h, fmt, data_off) for textures

    @property
    def type_name(self):
        return TYPE_NAMES.get(self.type, f'type {self.type}')

    def read(self):
        return self.lu.read(self.off, self.size)

    def __repr__(self):
        return f'<Obj {self.type_name} {self.name!r} {self.id:08x} in {self.lu.name}>'


class LUFile:
    def __init__(self, path):
        self.path = path
        self.fsize = os.path.getsize(path)
        with open(path, 'rb') as f:
            head = f.read(0x40 + NSLOTS * 8)
            if len(head) < 0x48 or struct.unpack_from('>I', head, 0)[0] != MAGIC:
                raise LUError('not a Chimera load unit (bad magic)')
            self.name = cstr(head[0x10:0x1c]) or os.path.splitext(os.path.basename(path))[0]
            self.game = cstr(head[0x1c:0x28])
            raw = []
            for i in range(NSLOTS):
                o = 0x40 + i * 8
                if o + 8 > len(head):
                    break
                cnt, toff = struct.unpack_from('>II', head, o)
                if toff == 0xFFFFFFFF or not cnt or toff + cnt * 8 > self.fsize:
                    continue
                f.seek(toff)
                tab = f.read(cnt * 8)
                for k in range(cnt):
                    oid, ooff = struct.unpack_from('>II', tab, k * 8)
                    raw.append((i, oid, ooff))
            offs = sorted({o for _, _, o in raw if o < self.fsize})
            nxt = dict(zip(offs, offs[1:] + [self.fsize]))
            self.objects = []
            for t, oid, o in raw:
                if o >= self.fsize:
                    continue
                ob = Obj(self, t, oid, o, nxt[o] - o)
                f.seek(o)
                h = f.read(0xA0 if t == T_TEXTURE else 0x30)
                ob.name = cstr(h[0x10:0x30]) if len(h) >= 0x30 else ''
                if t == T_TEXTURE and len(h) >= 0xA0:
                    doff, w, hgt = struct.unpack_from('>III', h, 0x68)
                    fmt = struct.unpack_from('>I', h, 0x9C)[0]
                    if fmt in GX_FORMATS and 0 < w <= 4096 and 0 < hgt <= 4096 and doff < ob.size:
                        ob.tex = dict(w=w, h=hgt, fmt=fmt, data_off=doff)
                self.objects.append(ob)
        self.textures = sorted((o for o in self.objects if o.tex), key=lambda o: o.name.lower())
        self.models = sorted((o for o in self.objects if o.type == T_MODEL), key=lambda o: o.name.lower())

    def read(self, off, size):
        with open(self.path, 'rb') as f:
            f.seek(off)
            return f.read(size)

    def __repr__(self):
        return f'<LUFile {self.name} {self.path}>'
