"""Index of every load unit in a game folder (or a list of files) with a global
texture-reference table so models resolve textures stored in any other file."""
import os
from collections import defaultdict
from lufile import LUFile, LUError
from hashes import jenk
from gxtex import decode
from model import Model


def tex_ref(name):
    return 0x22000000 | (jenk(name.encode('latin-1')) & 0xFFFFFF)


def find_wii_files(root):
    out = []
    for dp, dn, fn in os.walk(root):
        dn.sort()
        for f in sorted(fn):
            if f.lower().endswith('.wii'):
                out.append(os.path.join(dp, f))
    return out


class GameIndex:
    def __init__(self, root=None):
        self.root = root
        self.lus = []
        self.errors = []            # (path, message)
        self.tex_by_ref = defaultdict(list)
        self._model_cache = {}
        self._tex_cache = {}

    @staticmethod
    def open_file(path):
        """Parse one file. Returns (LUFile, None) or (None, error message). Thread-safe."""
        try:
            return LUFile(path), None
        except (LUError, OSError, ValueError) as e:
            return None, str(e)
        except Exception as e:  # malformed tables etc.
            return None, f'{type(e).__name__}: {e}'

    def register(self, lu):
        if any(l.path == lu.path for l in self.lus):
            return
        self.lus.append(lu)
        for t in lu.textures:
            self.tex_by_ref[tex_ref(t.name)].append(t)

    def add_file(self, path):
        lu, err = self.open_file(path)
        if err:
            self.errors.append((path, err))
            return None
        self.register(lu)
        return lu

    def scan(self, paths, progress=None):
        """paths: folder(s) and/or .wii files. progress(i, n, path) is called per file."""
        files = []
        for p in paths:
            files += find_wii_files(p) if os.path.isdir(p) else [p]
        for i, f in enumerate(files):
            if progress:
                progress(i, len(files), f)
            self.add_file(f)
        return self

    def rel(self, lu):
        try:
            return os.path.relpath(lu.path, self.root) if self.root else os.path.basename(lu.path)
        except ValueError:
            return lu.path

    # ---- lookups ----
    @property
    def models(self):
        return [m for lu in self.lus for m in lu.models]

    @property
    def textures(self):
        return [t for lu in self.lus for t in lu.textures]

    def resolve_texture(self, ref, prefer_lu=None):
        cands = self.tex_by_ref.get(ref)
        if not cands:
            return None
        if prefer_lu is not None:
            for c in cands:
                if c.lu is prefer_lu:
                    return c
        return cands[0]

    def load_model(self, obj):
        key = (obj.lu.path, obj.off)
        if key not in self._model_cache:
            if len(self._model_cache) > 64:
                self._model_cache.clear()
            self._model_cache[key] = Model(obj.read(), obj.name)
        return self._model_cache[key]

    def texture_rgba(self, t):
        key = (t.lu.path, t.off)
        if key not in self._tex_cache:
            if len(self._tex_cache) > 256:
                self._tex_cache.clear()
            tx = t.tex
            data = t.lu.read(t.off + tx['data_off'], t.size - tx['data_off'])
            self._tex_cache[key] = decode(data, tx['fmt'], tx['w'], tx['h'])
        return self._tex_cache[key]

    def surface_texture(self, model_obj, surface):
        for r in surface.texrefs:
            t = self.resolve_texture(r, model_obj.lu)
            if t:
                return t
        return None
