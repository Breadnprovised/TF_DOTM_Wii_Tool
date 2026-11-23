#!/usr/bin/env python3
"""dotm_extract.py - batch extractor for Transformers: Dark of the Moon (Wii) load units.

    python scripts\\dotm_extract.py OUT_DIR <game folder | file.wii> [...]   # models -> .glb (skeleton, animations) + PNG, textures -> PNG
    python scripts\\dotm_extract.py --obj OUT_DIR <game folder | file.wii> [...]   # models as OBJ/MTL/PNG instead
    python scripts\\dotm_extract.py --list <game folder | file.wii> [...]    # inventory

.glb options (before OUT_DIR; the model viewer's Export Settings have the same two):
    --separate      every piece as its own mesh (default: pieces merged into one mesh)
    --no-skeleton   meshes only, in the bind pose (default: with bones and weights)
    --no-anims      leave out the animations (default: every animation that fits the model)

Textures are resolved across every file given, so passing the whole game folder
lets each model find its textures wherever they are stored.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gameindex import GameIndex
from export import export_all


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    fmt, merge, skeleton, anims = 'glb', True, True, True
    while argv and argv[0] in ('--obj', '--separate', '--no-skeleton', '--no-anims'):
        opt, argv = argv[0], argv[1:]
        if opt == '--obj':
            fmt = 'obj'
        elif opt == '--separate':
            merge = False
        elif opt == '--no-anims':
            anims = False
        else:
            skeleton = False
    if len(argv) < 2:
        print(__doc__)
        return 1
    listing = argv[0] == '--list'
    paths = argv[1:]
    root = paths[0] if len(paths) == 1 and os.path.isdir(paths[0]) else None
    idx = GameIndex(root).scan(paths, lambda i, n, p: print(f'\rindexing {i + 1}/{n}', end='', file=sys.stderr))
    print(file=sys.stderr)
    for p, e in idx.errors:
        print(f'skipped {p}: {e}', file=sys.stderr)
    if listing:
        for lu in idx.lus:
            print(f'== {idx.rel(lu)}  ({lu.name}): {len(lu.models)} models, {len(lu.textures)} textures, '
                  f'{len(lu.objects)} objects')
            for m in lu.models:
                print(f'   model   {m.name}')
            for t in lu.textures:
                print(f"   texture {t.name}  {t.tex['w']}x{t.tex['h']}")
        return 0
    out = argv[0]
    errors = export_all(idx, out, fmt=fmt, merge=merge, skeleton=skeleton, animations=anims, progress=lambda i, n, what: print(f'\r[{i + 1}/{n}] {what[:60]:60s}', end='', file=sys.stderr))
    print(f'\ndone: {len(idx.models)} models, {len(idx.textures)} textures from {len(idx.lus)} files -> {out}',
          file=sys.stderr)
    for e in errors:
        print('error:', e, file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
