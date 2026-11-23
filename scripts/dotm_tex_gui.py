#!/usr/bin/env python3
"""
dotm_tex_gui.py - texture browser for Transformers: Dark of the Moon (Wii)

Pick the game's folder ("transformers dark of the moon", or any folder with
.wii load units). Every .wii under it is indexed (headers only, so it is
quick) and every texture is listed; click one to preview it, then export
the selection or the whole list to PNG.

    python3 dotm_tex_gui.py [GAME_FOLDER]

Requires Python 3.8+ with tkinter, numpy and Pillow, and the other scripts
of this folder next to it.
"""

import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
from collections import OrderedDict
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from gameindex import GameIndex, find_wii_files
    from lufile import GX_FORMATS
    from gxtex import decode as gx_decode
    from export import safe
except ImportError:
    tk.Tk().withdraw()
    messagebox.showerror("Missing file",
                         "gameindex.py, lufile.py, gxtex.py and export.py must be in "
                         "the same folder as this script.")
    raise SystemExit(1)

from PIL import Image, ImageTk

APP_TITLE = "Dark of the Moon Texture Browser"
CONFIG = os.path.join(os.path.expanduser("~"), ".dotm_tex_gui.json")

COLUMNS = [  # id, heading, width, anchor
    ("file", "File", 200, "w"),
    ("name", "Name", 220, "w"),
    ("size", "Size", 90, "center"),
    ("fmt", "Format", 70, "center"),
    ("bytes", "Bytes", 80, "e"),
    ("id", "Object ID", 100, "w"),
    ("offset", "Offset", 90, "e"),
]
BACKGROUNDS = ["Checker", "Black", "White", "Grey", "Magenta"]
ZOOMS = ["Fit", "1x", "2x", "4x", "8x"]


class Entry:
    """A row in the list: one texture object plus where it came from."""
    __slots__ = ("obj", "rel", "base")

    def __init__(self, obj, rel, base):
        self.obj, self.rel, self.base = obj, rel, base

    @property
    def w(self):
        return self.obj.tex["w"]

    @property
    def h(self):
        return self.obj.tex["h"]

    @property
    def fmt(self):
        return GX_FORMATS[self.obj.tex["fmt"]]

    @property
    def data_size(self):
        return self.obj.size - self.obj.tex["data_off"]

    def values(self):
        o = self.obj
        return (self.rel, o.name, f"{self.w}x{self.h}", self.fmt,
                f"{self.data_size:,}", f"0x{o.id:08X}", f"0x{o.off:X}")

    def sort_key(self, col):
        o = self.obj
        return {
            "file": (self.rel.lower(), o.name.lower()), "name": (o.name.lower(), self.rel.lower()),
            "size": (self.w * self.h, self.w), "fmt": self.fmt,
            "bytes": self.data_size, "id": o.id, "offset": o.off,
        }[col]

    def image(self):
        tx = self.obj.tex
        data = self.obj.lu.read(self.obj.off + tx["data_off"], self.data_size)
        return Image.fromarray(gx_decode(data, tx["fmt"], tx["w"], tx["h"]), "RGBA")


def unique_names(objs):
    """PNG base names for the textures of one file (object ID added to repeats)."""
    seen = {}
    for o in objs:
        seen[o.name.lower()] = seen.get(o.name.lower(), 0) + 1
    return [safe(o.name) + (f"_{o.id:08x}" if seen[o.name.lower()] > 1 or not o.name else "")
            for o in objs]


class App(tk.Tk):
    def __init__(self, start_folder=None):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1280x780")
        self.minsize(900, 500)

        self.entries = []          # all textures found
        self.visible = []          # indices into entries, in display order
        self.root_dir = None
        self.sort_col, self.sort_rev = "file", False
        self.cache = OrderedDict() # entry id -> PIL image
        self.q = queue.Queue()
        self.worker = None
        self.stop_flag = threading.Event()
        self._filter_job = None
        self._photo = None
        self._bg_cache = {}

        self.cfg = self._load_cfg()
        self._build_ui()
        self.after(50, self._poll)
        folder = start_folder or None
        if folder:
            self.after(100, lambda: self.open_folder(folder))
        else:
            self._show_message("Open your game folder to begin\n(File ▸ Open game folder…)")
            # no folder on the command line: ask for it straight away
            self.after(150, self.ask_folder)

    # ---------------------------------------------------------------- UI --
    def _build_ui(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names() and sys.platform.startswith("linux"):
            style.theme_use("clam")

        menu = tk.Menu(self)
        fm = tk.Menu(menu, tearoff=0)
        fm.add_command(label="Open game folder…", accelerator="Ctrl+O",
                       command=self.ask_folder)
        fm.add_command(label="Rescan", accelerator="F5", command=self.rescan)
        fm.add_separator()
        fm.add_command(label="Export selected…", accelerator="Ctrl+E",
                       command=self.export_selected)
        fm.add_command(label="Export all listed…", command=self.export_listed)
        fm.add_command(label="Save preview as PNG…", accelerator="Ctrl+S",
                       command=self.save_preview)
        fm.add_separator()
        fm.add_command(label="Quit", command=self._on_close)
        menu.add_cascade(label="File", menu=fm)
        self.config(menu=menu)
        self.bind("<Control-o>", lambda e: self.ask_folder())
        self.bind("<F5>", lambda e: self.rescan())
        self.bind("<Control-e>", lambda e: self.export_selected())
        self.bind("<Control-s>", lambda e: self.save_preview())

        # toolbar
        bar = ttk.Frame(self, padding=(6, 6, 6, 2))
        bar.pack(fill="x")
        ttk.Button(bar, text="Open game folder…", command=self.ask_folder).pack(side="left")
        ttk.Button(bar, text="Rescan", command=self.rescan).pack(side="left", padx=(4, 12))
        ttk.Label(bar, text="Search:").pack(side="left")
        self.search = tk.StringVar()
        self.search.trace_add("write", lambda *a: self._schedule_filter())
        se = ttk.Entry(bar, textvariable=self.search, width=20)
        se.pack(side="left", padx=(4, 10))
        self.bind("<Control-f>", lambda e: se.focus_set())
        ttk.Label(bar, text="Format:").pack(side="left")
        self.fmt_filter = tk.StringVar(value="All")
        self.fmt_box = ttk.Combobox(bar, textvariable=self.fmt_filter, width=9,
                                    state="readonly", values=["All"])
        self.fmt_box.pack(side="left", padx=(4, 10))
        self.fmt_box.bind("<<ComboboxSelected>>", lambda e: self.apply_filter())
        self.dedupe = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Hide duplicates", variable=self.dedupe,
                        command=self.apply_filter).pack(side="left", padx=(0, 10))

        ttk.Button(bar, text="Export all listed…",
                   command=self.export_listed).pack(side="right")
        ttk.Button(bar, text="Export selected…",
                   command=self.export_selected).pack(side="right", padx=4)

        self.folder_lbl = ttk.Label(self, text="No folder open", padding=(8, 0),
                                    foreground="#666")
        self.folder_lbl.pack(fill="x")

        # main split
        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=6, pady=4)

        left = ttk.Frame(pane)
        self.tree = ttk.Treeview(left, columns=[c[0] for c in COLUMNS],
                                 show="headings", selectmode="extended")
        for cid, head, w, anchor in COLUMNS:
            self.tree.heading(cid, text=head, command=lambda c=cid: self.sort_by(c))
            self.tree.column(cid, width=w, anchor=anchor, stretch=(cid in ("file", "name")))
        vs = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        hs = ttk.Scrollbar(left, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self.update_preview())
        self.tree.bind("<Double-1>", lambda e: self.save_preview())
        self.tree.bind("<Button-3>", self._context_menu)
        self.tree.bind("<Control-a>", lambda e: (self.tree.selection_set(self.tree.get_children()), "break")[1])
        pane.add(left, weight=5)

        right = ttk.Frame(pane, padding=(6, 0, 0, 0))
        ctl = ttk.Frame(right)
        ctl.pack(fill="x")
        ttk.Label(ctl, text="Zoom:").pack(side="left")
        self.zoom = tk.StringVar(value="Fit")
        z = ttk.Combobox(ctl, textvariable=self.zoom, values=ZOOMS, width=4, state="readonly")
        z.pack(side="left", padx=(2, 8))
        z.bind("<<ComboboxSelected>>", lambda e: self.update_preview())
        ttk.Label(ctl, text="BG:").pack(side="left")
        self.bg = tk.StringVar(value=self.cfg.get("bg", "Checker"))
        b = ttk.Combobox(ctl, textvariable=self.bg, values=BACKGROUNDS, width=8, state="readonly")
        b.pack(side="left", padx=2)
        b.bind("<<ComboboxSelected>>", lambda e: self.update_preview())
        self.alpha_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctl, text="Alpha", variable=self.alpha_on,
                        command=self.update_preview).pack(side="left", padx=6)

        cf = ttk.Frame(right, relief="sunken", borderwidth=1)
        cf.pack(fill="both", expand=True, pady=6)
        self.canvas = tk.Canvas(cf, background="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._on_canvas_resize())
        self.info = ttk.Label(right, text="", justify="left", font=("TkFixedFont", 9))
        self.info.pack(fill="x")
        pane.add(right, weight=4)

        # status bar
        sb = ttk.Frame(self, padding=(6, 2))
        sb.pack(fill="x", side="bottom")
        self.status = ttk.Label(sb, text="Ready")
        self.status.pack(side="left", fill="x", expand=True)
        self.stop_btn = ttk.Button(sb, text="Stop", command=self.stop_flag.set,
                                   state="disabled")
        self.stop_btn.pack(side="right")
        self.prog = ttk.Progressbar(sb, length=240, mode="determinate")
        self.prog.pack(side="right", padx=6)

        self.menu_ctx = tk.Menu(self, tearoff=0)
        self.menu_ctx.add_command(label="Export selected…", command=self.export_selected)
        self.menu_ctx.add_command(label="Save preview as PNG…", command=self.save_preview)
        self.menu_ctx.add_separator()
        self.menu_ctx.add_command(label="Copy name", command=lambda: self._copy("name"))
        self.menu_ctx.add_command(label="Copy file path", command=lambda: self._copy("path"))
        self.menu_ctx.add_command(label="Show all textures from this file",
                                  command=self._filter_to_file)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------ config --
    def _load_cfg(self):
        try:
            with open(CONFIG) as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_cfg(self):
        self.cfg.update(bg=self.bg.get())
        try:
            with open(CONFIG, "w") as f:
                json.dump(self.cfg, f)
        except Exception:
            pass

    def _on_close(self):
        self.stop_flag.set()
        self._save_cfg()
        self.destroy()

    # ------------------------------------------------------------ scanning --
    def ask_folder(self):
        d = filedialog.askdirectory(title='Select the game folder ("transformers dark of the moon")',
                                    initialdir=self.cfg.get("last_folder") or os.getcwd())
        if d:
            self.open_folder(d)

    def rescan(self):
        if self.root_dir:
            self.open_folder(self.root_dir)

    def open_folder(self, folder):
        if self._busy():
            messagebox.showinfo(APP_TITLE, "Please wait for the current job, or press Stop.")
            return
        folder = os.path.abspath(folder)
        if os.path.isfile(folder):          # a .wii dropped on the launcher: use its folder
            folder = os.path.dirname(folder)
        if not os.path.isdir(folder):
            messagebox.showerror(APP_TITLE, f"Not a folder:\n{folder}")
            return
        self.root_dir = folder
        self.cfg["last_folder"] = folder
        self._save_cfg()
        self.folder_lbl.config(text=folder)
        self.title(f"{APP_TITLE} — {os.path.basename(folder) or folder}")
        self.entries.clear()
        self.cache.clear()
        self.tree.delete(*self.tree.get_children())
        self.visible = []
        self._show_message("Scanning…")
        self._start(self._scan_worker, folder)

    def _scan_worker(self, folder):
        files = find_wii_files(folder)
        total = len(files)
        self.q.put(("progress", 0, total, "Listing files…"))
        found_files = 0
        warnings = []
        last = 0
        for i, path in enumerate(files):
            if self.stop_flag.is_set():
                break
            rel = os.path.relpath(path, folder)
            lu, err = GameIndex.open_file(path)
            if err:
                warnings.append(f"{rel}: {err}")
            elif lu.textures:
                found_files += 1
                names = unique_names(lu.textures)
                self.q.put(("file", rel, [Entry(t, rel, names[k])
                                          for k, t in enumerate(lu.textures)]))
            now = time.monotonic()
            if now - last > 0.05 or i + 1 == total:
                last = now
                self.q.put(("progress", i + 1, total, f"Scanning {rel}"))
        self.q.put(("scan_done", total, found_files, warnings, self.stop_flag.is_set()))

    # ------------------------------------------------------------ worker plumbing --
    def _busy(self):
        return self.worker is not None and self.worker.is_alive()

    def _start(self, fn, *args):
        self.stop_flag.clear()
        self.stop_btn.config(state="normal")
        self.worker = threading.Thread(target=fn, args=args, daemon=True)
        self.worker.start()

    def _poll(self):
        added = False
        try:
            for _ in range(500):
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, i, n, text = msg
                    self.prog.config(maximum=max(n, 1), value=i)
                    self.status.config(text=f"{text}   ({i}/{n})")
                elif kind == "file":
                    self.entries.extend(msg[2])
                    added = True
                elif kind == "scan_done":
                    _, total, nfiles, warnings, stopped = msg
                    self.stop_btn.config(state="disabled")
                    self._refresh_formats()
                    self.apply_filter()
                    shown = (f"{len(self.visible):,} of " if len(self.visible) != len(self.entries) else "")
                    s = (f"{shown}{len(self.entries):,} textures shown, from {nfiles} file(s) "
                         f"— {total:,} .wii files scanned")
                    if stopped:
                        s += " (stopped)"
                    if warnings:
                        s += f" — {len(warnings)} skipped, see console"
                        for w in warnings:
                            print("skipped:", w, file=sys.stderr)
                    self.status.config(text=s)
                    if not self.entries:
                        self._show_message("No textures found in this folder.\n"
                                           "Pick the folder that contains the game's .wii files.")
                    elif not self.tree.selection():
                        kids = self.tree.get_children()
                        if kids:
                            self.tree.selection_set(kids[0])
                            self.tree.focus(kids[0])
                elif kind == "export_done":
                    _, n, fails, dest, stopped = msg
                    self.stop_btn.config(state="disabled")
                    text = f"Exported {n:,} texture(s) to {dest}"
                    if stopped:
                        text += " (stopped)"
                    if fails:
                        text += f" — {len(fails)} failed"
                        for f in fails:
                            print("export failed:", f, file=sys.stderr)
                    self.status.config(text=text)
                    if fails:
                        messagebox.showwarning(APP_TITLE, text + ":\n\n" + "\n".join(fails[:15]))
        except queue.Empty:
            pass
        if added and self._busy():
            self._refresh_formats()
            self.apply_filter(keep_scroll=True)
        self.after(50, self._poll)

    # ------------------------------------------------------------ list --
    def _refresh_formats(self):
        fmts = sorted({e.fmt for e in self.entries})
        self.fmt_box.config(values=["All"] + fmts)

    def _schedule_filter(self):
        if self._filter_job:
            self.after_cancel(self._filter_job)
        self._filter_job = self.after(200, self.apply_filter)

    def apply_filter(self, keep_scroll=False):
        self._filter_job = None
        terms = self.search.get().lower().split()
        fmt = self.fmt_filter.get()
        seen = set()
        vis = []
        for i, e in enumerate(self.entries):
            if fmt != "All" and e.fmt != fmt:
                continue
            if terms:
                hay = f"{e.rel.lower()} {e.obj.name.lower()} {e.w}x{e.h} {e.fmt.lower()} {e.obj.id:08x}"
                if not all(term in hay for term in terms):
                    continue
            if self.dedupe.get():
                key = (e.obj.name.lower(), e.w, e.h, e.fmt, e.data_size)
                if key in seen:
                    continue
                seen.add(key)
            vis.append(i)
        key = self.sort_col
        vis.sort(key=lambda i: self.entries[i].sort_key(key), reverse=self.sort_rev)

        if vis == self.visible:
            return
        sel = set(self.tree.selection())
        top = self.tree.yview()[0]
        self.visible = vis
        self.tree.delete(*self.tree.get_children())
        for i in vis:
            self.tree.insert("", "end", iid=str(i), values=self.entries[i].values())
        keep = [s for s in sel if self.tree.exists(s)]
        if keep:
            self.tree.selection_set(keep)
            if not keep_scroll:
                self.tree.see(keep[0])
        if keep_scroll:
            self.tree.yview_moveto(top)
        if not self._busy():
            n = len(self.entries)
            shown = f"{len(vis):,} of {n:,} textures shown" if len(vis) != n else f"{n:,} textures"
            self.status.config(text=shown)

    def sort_by(self, col):
        if self.sort_col == col:
            self.sort_rev = not self.sort_rev
        else:
            self.sort_col, self.sort_rev = col, False
        for cid, head, *_ in COLUMNS:
            arrow = (" ▼" if self.sort_rev else " ▲") if cid == col else ""
            self.tree.heading(cid, text=head + arrow)
        self.visible = None  # force rebuild
        self.apply_filter()

    def selected_entries(self):
        return [self.entries[int(i)] for i in self.tree.selection()]

    def _context_menu(self, ev):
        row = self.tree.identify_row(ev.y)
        if row and row not in self.tree.selection():
            self.tree.selection_set(row)
        if self.tree.selection():
            self.menu_ctx.tk_popup(ev.x_root, ev.y_root)

    def _copy(self, what):
        es = self.selected_entries()
        if not es:
            return
        if what == "name":
            text = "\n".join(e.obj.name for e in es)
        else:
            text = "\n".join(sorted({os.path.join(self.root_dir, e.rel) for e in es}))
        self.clipboard_clear()
        self.clipboard_append(text)

    def _filter_to_file(self):
        es = self.selected_entries()
        if es:
            self.fmt_filter.set("All")
            self.search.set(es[0].rel)
            self.apply_filter()

    # ------------------------------------------------------------ preview --
    def _show_message(self, text):
        self._message = text
        self.canvas.delete("all")
        self._photo = None
        w, h = max(self.canvas.winfo_width(), 200), max(self.canvas.winfo_height(), 200)
        self.canvas.create_text(w // 2, h // 2, text=text, fill="#ccc",
                                justify="center", font=("TkDefaultFont", 11))
        self.info.config(text="")

    def _on_canvas_resize(self):
        if self.tree.selection():
            self.update_preview()
        elif getattr(self, "_message", None):
            self._show_message(self._message)      # keep the message centred

    def _get_image(self, idx):
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        img = self.entries[idx].image()
        self.cache[idx] = img
        while len(self.cache) > 48:
            self.cache.popitem(last=False)
        return img

    def _background(self, w, h):
        kind = self.bg.get()
        key = (kind, w, h)
        if key in self._bg_cache:
            return self._bg_cache[key]
        if kind == "Checker":
            tile = Image.new("RGBA", (16, 16), (200, 200, 200, 255))
            tile.paste((150, 150, 150, 255), (0, 0, 8, 8))
            tile.paste((150, 150, 150, 255), (8, 8, 16, 16))
            img = Image.new("RGBA", (w, h))
            for y in range(0, h, 16):
                for x in range(0, w, 16):
                    img.paste(tile, (x, y))
        else:
            col = {"Black": (0, 0, 0), "White": (255, 255, 255),
                   "Grey": (128, 128, 128), "Magenta": (255, 0, 255)}[kind]
            img = Image.new("RGBA", (w, h), col + (255,))
        if len(self._bg_cache) > 8:
            self._bg_cache.clear()
        self._bg_cache[key] = img
        return img

    def update_preview(self):
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(self.tree.focus() if self.tree.focus() in sel else sel[0])
        e = self.entries[idx]
        o = e.obj
        info = (f"{e.rel}   {o.name}\n"
                f"{e.w}x{e.h}   {e.fmt} (GX format {o.tex['fmt']})   object 0x{o.id:08X}\n"
                f"{e.data_size:,} bytes incl. mips   object @0x{o.off:X}   "
                f"data @0x{o.off + o.tex['data_off']:X}")
        if len(sel) > 1:
            info += f"\n{len(sel)} textures selected"
        try:
            img = self._get_image(idx)
        except Exception as ex:
            self._show_message(f"Could not decode:\n{ex}")
            self.info.config(text=info)
            return
        if not self.alpha_on.get():
            img = img.convert("RGB").convert("RGBA")

        cw, ch = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        z = self.zoom.get()
        if z == "Fit":
            s = min((cw - 16) / img.width, (ch - 16) / img.height)
            s = min(s, 8.0)
            if s >= 1:
                s = max(1, int(s))  # whole-number upscale keeps pixels crisp
        else:
            s = float(z.rstrip("x"))
        w, h = max(1, round(img.width * s)), max(1, round(img.height * s))
        shown = img.resize((w, h), Image.NEAREST if s >= 1 else Image.LANCZOS) \
            if (w, h) != img.size else img
        # very large zooms: crop to the visible area to keep things quick
        if w > 4096 or h > 4096:
            shown = shown.crop((0, 0, min(w, cw), min(h, ch)))
            w, h = shown.size
        comp = self._background(w, h).copy()
        comp.alpha_composite(shown)
        self._photo = ImageTk.PhotoImage(comp)
        self.canvas.delete("all")
        self._message = None
        self.canvas.create_image(max(cw // 2, w // 2), max(ch // 2, h // 2),
                                 image=self._photo, anchor="center")
        self.canvas.config(scrollregion=(0, 0, max(w, cw), max(h, ch)))
        self.info.config(text=info + f"\nshowing {e.w}x{e.h} at {s:.2g}x")

    # ------------------------------------------------------------ export --
    def save_preview(self):
        sel = self.selected_entries()
        if not sel:
            return
        e = sel[0]
        stem = os.path.splitext(os.path.basename(e.rel))[0]
        default = f"{stem}_{e.base}.png"
        path = filedialog.asksaveasfilename(
            title="Save texture as PNG", defaultextension=".png", initialfile=default,
            initialdir=self.cfg.get("last_export") or os.getcwd(),
            filetypes=[("PNG image", "*.png")])
        if not path:
            return
        try:
            e.image().save(path)
        except Exception as ex:
            messagebox.showerror(APP_TITLE, f"Could not save:\n{ex}")
            return
        self.cfg["last_export"] = os.path.dirname(path)
        self.status.config(text=f"Saved {path}")

    def export_selected(self):
        self._export(self.selected_entries(), "selected")

    def export_listed(self):
        self._export([self.entries[i] for i in self.visible], "listed")

    def _export(self, entries, what):
        if self._busy():
            messagebox.showinfo(APP_TITLE, "Please wait for the current job, or press Stop.")
            return
        if not entries:
            messagebox.showinfo(APP_TITLE, f"No {what} textures to export.")
            return
        dest = filedialog.askdirectory(
            title=f"Export {len(entries):,} texture(s) to…",
            initialdir=self.cfg.get("last_export") or os.getcwd())
        if not dest:
            return
        self.cfg["last_export"] = dest
        self._save_cfg()
        self._start(self._export_worker, list(entries), dest)

    def _export_worker(self, entries, dest):
        done, fails, n = 0, [], len(entries)
        last = 0
        for e in entries:
            if self.stop_flag.is_set():
                break
            # one folder per .wii file, mirroring the game's folder layout
            out_dir = os.path.join(dest, os.path.splitext(e.rel)[0])
            try:
                os.makedirs(out_dir, exist_ok=True)
                e.image().save(os.path.join(out_dir, e.base + ".png"))
                done += 1
            except Exception as ex:
                fails.append(f"{e.rel} {e.obj.name}: {ex}")
            now = time.monotonic()
            if now - last > 0.05:
                last = now
                self.q.put(("progress", done + len(fails), n, f"Exporting {e.rel}"))
        self.q.put(("progress", n, n, "Export finished"))
        self.q.put(("export_done", done, fails, dest, self.stop_flag.is_set()))


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    App(folder).mainloop()


if __name__ == "__main__":
    main()
