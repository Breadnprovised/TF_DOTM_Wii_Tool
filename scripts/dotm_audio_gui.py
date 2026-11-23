#!/usr/bin/env python3
"""
dotm_audio_gui.py - audio browser for Transformers: Dark of the Moon (Wii)

Lists every sound in the game's .str (and .sbk) files - dialogue, music,
ambience and cutscene audio - sorted into categories or by file. Sounds can
be previewed and exported as standard 16-bit WAV files.

Usage:
    python dotm_audio_gui.py ["path to the game folder"]

Keys: Space = play / stop, Enter = play, Ctrl+E = export selected,
      click the waveform to play from that point.
"""

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageTk
except ImportError as ex:
    tk.Tk().withdraw()
    messagebox.showerror("Missing package", f"{ex}\n\nInstall with:\n  pip install numpy pillow")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dotm_audio as da  # noqa: E402

APP_TITLE = "Dark of the Moon Audio Browser"
CONFIG = os.path.join(os.path.expanduser("~"), ".dotm_audio_gui.json")
WAVE_W, WAVE_H = 640, 170
CACHE_BYTES = 300 * 1024 * 1024
CATEGORY_ORDER = {"Dialogue": 0, "Music": 1, "Ambience": 2, "Cutscenes": 3, "Sound effects": 4, "Other": 5}


def fmt_time(t):
    if t is None:
        return ""
    m, s = divmod(t, 60)
    return f"{int(m)}:{s:04.1f}"


# ------------------------------------------------------------- playback --
class Player:
    """Plays int16 PCM through the system: winsound on Windows, afplay on
    macOS, paplay/aplay/ffplay on Linux. Audio is handed over as a
    temporary WAV file."""

    def __init__(self):
        self.tmpdir = tempfile.mkdtemp(prefix="dotm_audio_")
        self.flip = 0
        self.proc = None
        self.playing = False
        self.start_wall = 0.0
        self.start_pos = 0.0
        self.length = 0.0
        self.loop = False
        self.whole_loop = False
        self.pcm = None
        self.rate = 32000
        self.backend = self._find_backend()

    @staticmethod
    def _find_backend():
        if os.name == "nt":
            try:
                import winsound  # noqa: F401  (availability check)
                return "winsound"
            except ImportError:
                return None
        import shutil
        cands = ["afplay"] if sys.platform == "darwin" else ["paplay", "aplay", "ffplay"]
        for c in cands:
            if shutil.which(c):
                return c
        return None

    def _write(self, pcm):
        self.flip ^= 1
        path = os.path.join(self.tmpdir, f"preview{self.flip}.wav")
        da.write_wav(path, pcm, self.rate)
        return path

    def play(self, pcm, rate, pos=0.0, loop=False):
        """Play from `pos` seconds."""
        self.stop()
        if self.backend is None:
            raise RuntimeError("No audio player found on this system")
        self.pcm, self.rate, self.loop = pcm, rate, loop
        self.length = len(pcm) / rate
        pos = max(0.0, min(pos, self.length))
        start = int(pos * rate)
        path = self._write(pcm[start:] if start else pcm)
        whole_loop = loop and start == 0
        if self.backend == "winsound":
            import winsound
            flags = winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT
            if whole_loop:
                flags |= winsound.SND_LOOP
            winsound.PlaySound(path, flags)
        else:
            cmd = {"afplay": ["afplay", path], "paplay": ["paplay", path],
                   "aplay": ["aplay", "-q", path],
                   "ffplay": ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path]}[self.backend]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.whole_loop = whole_loop
        self.playing = True
        self.start_wall = time.perf_counter()
        self.start_pos = pos

    def stop(self):
        if self.backend == "winsound":
            try:
                import winsound
                winsound.PlaySound(None, 0)
            except Exception:
                pass
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None
        self.playing = False

    def position(self):
        """Current position in seconds, or None when stopped. Also restarts
        looping sounds that were started part-way through."""
        if not self.playing:
            return None
        pos = self.start_pos + time.perf_counter() - self.start_wall
        if pos >= self.length:
            if self.loop:
                if self.backend == "winsound" and self.whole_loop:
                    return (pos % self.length) if self.length else 0.0
                self.play(self.pcm, self.rate, 0.0, True)     # loop from the start
                return 0.0
            self.playing = False
            self.proc = None
            return None
        return pos

    def close(self):
        self.stop()
        try:
            for f in os.listdir(self.tmpdir):
                os.remove(os.path.join(self.tmpdir, f))
            os.rmdir(self.tmpdir)
        except OSError:
            pass


# ------------------------------------------------------------------ app --
class App(tk.Tk):
    def __init__(self, start_folder=None):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1400x800")
        self.minsize(900, 500)
        self.root_dir = None
        self.sounds = []             # every Sound found
        self.dups = set()            # id() of sounds that are copies of another listed sound
        self.leaf = {}               # tree iid -> Sound
        self.cache = {}              # (path, data_off) -> pcm
        self.cache_order = []
        self.current = None          # Sound shown on the right
        self.current_pcm = None
        self.wave_img = None
        self.playhead = None
        self.q = queue.Queue()
        self.busy = False
        self.stop_flag = threading.Event()
        self.player = Player()
        try:
            self.cfg = json.load(open(CONFIG))
        except Exception:
            self.cfg = {}
        self._build_ui()
        self.after(50, self._poll)
        self.after(40, self._tick)
        self.protocol("WM_DELETE_WINDOW", self._close)
        if start_folder:
            self.after(100, lambda: self.open_folder(start_folder))
        else:
            self.status.config(text="Open the game folder to begin.")
            # no folder on the command line: ask for it straight away
            self.after(150, self.ask_folder)

    # ---------------------------------------------------------- layout --
    def _build_ui(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names() and sys.platform.startswith("linux"):
            style.theme_use("clam")
        menu = tk.Menu(self)
        fm = tk.Menu(menu, tearoff=0)
        fm.add_command(label="Open game folder...", accelerator="Ctrl+O", command=self.ask_folder)
        fm.add_command(label="Rescan", accelerator="F5", command=self.rescan)
        fm.add_separator()
        fm.add_command(label="Export selected...", accelerator="Ctrl+E", command=self.export)
        fm.add_command(label="Export all listed...", command=lambda: self.export(all_listed=True))
        fm.add_separator()
        fm.add_command(label="Exit", command=self._close)
        menu.add_cascade(label="File", menu=fm)
        self.config(menu=menu)

        top = ttk.Frame(self, padding=(6, 6, 6, 2))
        top.pack(fill="x")
        ttk.Button(top, text="Open game folder...", command=self.ask_folder).pack(side="left")
        ttk.Button(top, text="Rescan", command=self.rescan).pack(side="left", padx=(4, 0))
        ttk.Label(top, text="  Language:").pack(side="left")
        self.lang_var = tk.StringVar(value=self.cfg.get("language", "English"))
        self.lang_cb = ttk.Combobox(top, textvariable=self.lang_var, state="readonly", width=10,
                                    values=[self.lang_var.get()])
        self.lang_cb.pack(side="left")
        self.lang_cb.bind("<<ComboboxSelected>>", lambda e: self.rebuild())
        ttk.Label(top, text="  Sort by:").pack(side="left")
        self.group_var = tk.StringVar(value="Category")
        gcb = ttk.Combobox(top, textvariable=self.group_var, state="readonly", width=9,
                           values=["Category", "File"])
        gcb.pack(side="left")
        gcb.bind("<<ComboboxSelected>>", lambda e: self.rebuild())
        ttk.Label(top, text="  Search:").pack(side="left")
        self.search_var = tk.StringVar()
        se = ttk.Entry(top, textvariable=self.search_var, width=24)
        se.pack(side="left")
        self.bind("<Control-f>", lambda e: se.focus_set())
        self._search_job = None
        self.search_var.trace_add("write", lambda *a: self._search_later())
        self.dedupe = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Hide duplicates", variable=self.dedupe,
                        command=self.rebuild).pack(side="left", padx=(10, 0))
        ttk.Button(top, text="Export all listed...", command=lambda: self.export(all_listed=True)).pack(side="right")
        ttk.Button(top, text="Export selected...", command=self.export).pack(side="right", padx=4)
        self.path_lbl = ttk.Label(self, text="No folder open", foreground="#666", padding=(8, 0))
        self.path_lbl.pack(fill="x")

        pan = ttk.PanedWindow(self, orient="horizontal")
        pan.pack(fill="both", expand=True, padx=6, pady=4)
        left = ttk.Frame(pan)
        cols = [("dur", "Length", 60, "e"), ("fmt", "Format", 110, "w"),
                ("src", "Stored in", 220, "w")]
        self.tree = ttk.Treeview(left, columns=[c[0] for c in cols], selectmode="extended")
        self.tree.heading("#0", text="Sound")
        self.tree.column("#0", width=300, stretch=True)
        for cid, head, w, anc in cols:
            self.tree.heading(cid, text=head)
            self.tree.column(cid, width=w, anchor=anc, stretch=False)
        ys = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=ys.set)
        self.tree.pack(side="left", fill="both", expand=True)
        ys.pack(side="right", fill="y")
        self.tree.tag_configure("group", foreground="#224")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", self._on_double)
        self.tree.bind("<Return>", self._on_return)
        self.tree.bind("<Button-3>", self._context_menu)
        pan.add(left, weight=3)

        right = ttk.Frame(pan, padding=(8, 0, 0, 0))
        self.title_lbl = ttk.Label(right, text="", font=("TkDefaultFont", 12, "bold"), wraplength=WAVE_W)
        self.title_lbl.pack(anchor="w")
        self.info_lbl = ttk.Label(right, text="", justify="left", wraplength=WAVE_W)
        self.info_lbl.pack(anchor="w", pady=(2, 8))
        self.canvas = tk.Canvas(right, width=WAVE_W, height=WAVE_H, bg="#1e2228",
                                highlightthickness=0, cursor="hand2")
        self.canvas.pack(anchor="w")
        self.canvas.bind("<Button-1>", self._on_wave_click)
        ctl = ttk.Frame(right)
        ctl.pack(anchor="w", fill="x", pady=6)
        self.play_btn = ttk.Button(ctl, text="Play  (Space)", command=self.toggle_play, width=14)
        self.play_btn.pack(side="left")
        self.loop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctl, text="Loop", variable=self.loop_var).pack(side="left", padx=10)
        ttk.Button(ctl, text="Export this sound...", command=self.export_current).pack(side="left", padx=4)
        self.time_lbl = ttk.Label(ctl, text="", font=("TkFixedFont", 10))
        self.time_lbl.pack(side="right")
        ttk.Label(right, text="Click the waveform to play from that point.",
                  foreground="#777").pack(anchor="w")
        pan.add(right, weight=2)
        self.pan = pan
        self.after(200, lambda: pan.sashpos(0, max(600, self.winfo_width() - WAVE_W - 40)))

        sb = ttk.Frame(self, padding=(6, 2))
        sb.pack(fill="x", side="bottom")
        self.status = ttk.Label(sb, text="")
        self.status.pack(side="left", fill="x", expand=True)
        self.stop_btn = ttk.Button(sb, text="Stop", command=self.stop_flag.set, state="disabled")
        self.stop_btn.pack(side="right")
        self.prog = ttk.Progressbar(sb, length=220, mode="determinate")
        self.prog.pack(side="right", padx=6)

        self.menu_ctx = tk.Menu(self, tearoff=0)
        self.menu_ctx.add_command(label="Play", command=self.play)
        self.menu_ctx.add_command(label="Export selected...", command=self.export)
        self.menu_ctx.add_separator()
        self.menu_ctx.add_command(label="Copy name", command=lambda: self._copy("name"))
        self.menu_ctx.add_command(label="Copy file path", command=lambda: self._copy("path"))

        self.bind_all("<space>", self._space)
        self.bind_all("<Control-e>", lambda e: self.export())
        self.bind("<Control-o>", lambda e: self.ask_folder())
        self.bind("<F5>", lambda e: self.rescan())
        self._draw_empty_wave()

    def _save_cfg(self):
        self.cfg["language"] = self.lang_var.get()
        try:
            json.dump(self.cfg, open(CONFIG, "w"))
        except Exception:
            pass

    # ----------------------------------------------------------- folder --
    def ask_folder(self):
        d = filedialog.askdirectory(title='Select the game folder ("transformers dark of the moon")',
                                    initialdir=self.cfg.get("last_folder") or os.getcwd())
        if d:
            self.open_folder(d)

    def rescan(self):
        if self.root_dir:
            self.open_folder(self.root_dir)

    def open_folder(self, folder):
        if self.busy:
            messagebox.showinfo(APP_TITLE, "Please wait for the current job, or press Stop.")
            return
        folder = os.path.abspath(folder)
        if os.path.isfile(folder):
            folder = os.path.dirname(folder)
        if not os.path.isdir(folder):
            messagebox.showerror(APP_TITLE, f"Not a folder:\n{folder}")
            return
        self.player.stop()
        self.root_dir = folder
        self.cfg["last_folder"] = folder
        self._save_cfg()
        self.path_lbl.config(text=folder)
        self.title(f"{APP_TITLE} — {os.path.basename(folder) or folder}")
        self.sounds = []
        self.cache.clear()
        self.cache_order.clear()
        self.current = self.current_pcm = None
        self.title_lbl.config(text="")
        self.info_lbl.config(text="")
        self._draw_empty_wave()
        self.tree.delete(*self.tree.get_children())
        self.leaf.clear()
        self.busy = True
        self.stop_flag.clear()
        self.stop_btn.config(state="normal")

        def work():
            files = da.find_audio_files(folder)
            found, warns = [], []
            for i, path in enumerate(files):
                if self.stop_flag.is_set():
                    break
                rel = os.path.relpath(path, folder)
                try:
                    found += da.scan_file(path, folder)
                except (OSError, da.AudioError) as ex:
                    warns.append(f"{rel}: {ex}")
                self.q.put(("progress", i + 1, len(files), f"Reading {rel}"))
            self.q.put(("scanned", found, warns, len(files), self.stop_flag.is_set()))
        threading.Thread(target=work, daemon=True).start()

    def _scanned(self, found, warns, nfiles, stopped):
        self.busy = False
        self.stop_btn.config(state="disabled")
        self.prog.configure(value=0)
        self.sounds = found
        # copies: same name, length and data size as a sound listed before them
        seen, self.dups = set(), set()
        for s in sorted(found, key=lambda s: (len(s.rel), s.rel.lower())):
            k = (s.name.lower(), s.samples, s.data_size, s.channels)
            if k in seen:
                self.dups.add(id(s))
            seen.add(k)
        langs = sorted({s.lang for s in found if s.lang})
        names = [da.LANGS.get(l, l) for l in langs] + (["All"] if len(langs) > 1 else [])
        self.lang_cb.configure(values=names or ["-"])
        if self.lang_var.get() not in names:
            self.lang_var.set(names[0] if names else "-")
        for w in warns:
            print("skipped:", w, file=sys.stderr)
        self.rebuild()
        if not found:
            self._draw_empty_wave("No sounds found in this folder.\n"
                                  "Pick the game folder (with the .str files).")
        txt = self.status.cget("text") + f" — {nfiles:,} audio file(s) scanned"
        if warns:
            txt += f", {len(warns)} skipped (see console)"
        if stopped:
            txt += " (stopped)"
        self.status.config(text=txt)

    # ------------------------------------------------------------- tree --
    def _search_later(self):
        if self._search_job:
            self.after_cancel(self._search_job)
        self._search_job = self.after(250, self.rebuild)

    def _visible_sounds(self):
        lang_name = self.lang_var.get()
        code = next((k for k, v in da.LANGS.items() if v == lang_name), lang_name)
        terms = self.search_var.get().lower().split()
        out = []
        for s in self.sounds:
            if s.lang and lang_name != "All" and s.lang != code:
                continue
            if self.dedupe.get() and id(s) in self.dups:
                continue
            if terms:
                hay = f"{s.name} {' '.join(s.category)} {s.rel}".lower()
                if not all(t in hay for t in terms):
                    continue
            out.append(s)
        return out

    def rebuild(self):
        self._search_job = None
        self._save_cfg()
        sel_sound = self.current
        self.tree.delete(*self.tree.get_children())
        self.leaf.clear()
        sounds = self._visible_sounds()
        by_file = self.group_var.get() == "File"
        groups = {}
        # by file: drop a top folder every file shares (e.g. trnsw3ds)
        rels = [s.rel.replace("\\", "/").split("/") for s in sounds]
        skip = 1 if rels and all(len(r) > 2 and r[0] == rels[0][0] for r in rels) else 0
        for s, r in zip(sounds, rels):
            path = tuple(r[skip:]) if by_file else s.category
            groups.setdefault(path, []).append(s)
        searching = bool(self.search_var.get().strip())
        nodes = {}

        def node(path):
            if path in nodes:
                return nodes[path]
            parent = node(path[:-1]) if len(path) > 1 else ""
            iid = "g\x1f" + "\x1f".join(path)
            self.tree.insert(parent, "end", iid=iid, text=path[-1], tags=("group",),
                             open=searching and len(sounds) < 400)
            nodes[path] = iid
            return iid

        count = 0
        select_iid = None
        key = (lambda p: [x.lower() for x in p]) if by_file else \
            (lambda p: (CATEGORY_ORDER.get(p[0], 50), [x.lower() for x in p]))
        for path in sorted(groups, key=key):
            g = node(path)
            for s in sorted(groups[path], key=lambda s: s.name.lower()):
                iid = f"s\x1f{s.path}\x1f{s.data_off}"
                self.tree.insert(g, "end", iid=iid, text=s.name, values=self._row_values(s))
                self.leaf[iid] = s
                count += 1
                if s is sel_sound and select_iid is None:
                    select_iid = iid
        for path, iid in nodes.items():
            self.tree.item(iid, text=f"{path[-1]}  ({self._count_leaves(iid)})")
        if select_iid:
            self.tree.see(select_iid)
            self.tree.selection_set(select_iid)
        lang = self.lang_var.get()
        hidden = sum(1 for s in self.sounds if id(s) in self.dups) if self.dedupe.get() else 0
        self.status.config(text=f"{count:,} sounds listed"
                           + (f" ({lang} + not localized)" if lang not in ("All", "-") else "")
                           + (f", {hidden:,} duplicates hidden" if hidden else ""))

    def _count_leaves(self, iid):
        n = 0
        for c in self.tree.get_children(iid):
            n += 1 if c in self.leaf else self._count_leaves(c)
        return n

    @staticmethod
    def _describe(s):
        ch = "mono" if s.channels == 1 else ("stereo" if s.channels == 2 else f"{s.channels} ch")
        khz = s.rate // 1000 if s.rate % 1000 == 0 else round(s.rate / 1000, 1)
        return f"{khz} kHz {ch}"

    def _row_values(self, s):
        return (fmt_time(s.duration), self._describe(s), s.rel.replace("\\", "/"))

    def _on_double(self, ev):
        """Double-click plays only when it lands on a sound; on a category row
        it just opens / closes it (Treeview's own behaviour)."""
        if self.tree.identify_region(ev.x, ev.y) in ("tree", "cell") and \
                self.tree.identify_row(ev.y) in self.leaf:
            self.play()

    def _on_return(self, ev):
        if self.tree.focus() in self.leaf:
            self.play()

    def _context_menu(self, ev):
        row = self.tree.identify_row(ev.y)
        if row and row not in self.tree.selection():
            self.tree.selection_set(row)
        if self.tree.selection():
            self.menu_ctx.tk_popup(ev.x_root, ev.y_root)

    def _copy(self, what):
        ss = [s for s, _r in self._sounds_under(self.tree.selection())]
        if not ss:
            return
        text = "\n".join(s.name for s in ss) if what == "name" else \
            "\n".join(sorted({s.path for s in ss}))
        self.clipboard_clear()
        self.clipboard_append(text)

    # ------------------------------------------------------- selection --
    def _selected_sound(self):
        for iid in self.tree.selection():
            if iid in self.leaf:
                return self.leaf[iid]
        return None

    def _on_select(self, e=None):
        s = self._selected_sound()
        if s is None or s is self.current:
            return
        self.player.stop()
        self.current = s
        self.current_pcm = None
        self.title_lbl.config(text=s.name)
        lang = da.LANGS.get(s.lang, s.lang) if s.lang else "not localized"
        self.info_lbl.config(text=f"Category: {' / '.join(s.category)}\n"
                                  f"Language: {lang}\n"
                                  f"File: {s.path}\n"
                                  f"Bank @ 0x{s.bank_off:X}, data @ 0x{s.data_off:X} ({s.data_size:,} bytes)\n"
                                  f"Audio: {s.codec}, {s.rate:,} Hz {self._describe(s).split(' ', 2)[-1]}, "
                                  f"{fmt_time(s.duration)}")
        self._draw_empty_wave("Loading...")
        self.time_lbl.config(text="")
        self._load(s, autoplay=False)

    def _load(self, s, autoplay):
        key = (s.path, s.data_off)
        if key in self.cache:
            self._loaded(s, self.cache[key], autoplay=autoplay)
            return

        def work():
            try:
                self.q.put(("loaded", s, s.load(), autoplay))
            except Exception as ex:
                self.q.put(("load_err", s, ex))
        threading.Thread(target=work, daemon=True).start()

    def _loaded(self, s, pcm, autoplay=False):
        key = (s.path, s.data_off)
        if key not in self.cache:
            self.cache[key] = pcm
            self.cache_order.append(key)
            while sum(self.cache[k].nbytes for k in self.cache_order) > CACHE_BYTES and len(self.cache_order) > 1:
                self.cache.pop(self.cache_order.pop(0), None)
        if s is not self.current:
            return
        self.current_pcm = pcm
        self._draw_wave(pcm)
        self.time_lbl.config(text=f"0:00.0 / {fmt_time(s.duration)}")
        if autoplay:
            self.play()

    # ---------------------------------------------------------- waveform --
    def _draw_empty_wave(self, text=""):
        self.canvas.delete("all")
        self.playhead = None
        if text:
            self.canvas.create_text(WAVE_W // 2, WAVE_H // 2, text=text, fill="#9aa", justify="center")

    def _draw_wave(self, pcm):
        mono = pcm.astype(np.float32)
        mono = mono[:, 0] if mono.shape[1] == 1 else mono.mean(axis=1)
        n = len(mono)
        img = Image.new("RGB", (WAVE_W, WAVE_H), (30, 34, 40))
        d = ImageDraw.Draw(img)
        mid = WAVE_H // 2
        d.line([(0, mid), (WAVE_W, mid)], fill=(60, 66, 76))
        if n:
            edges = np.linspace(0, n, WAVE_W + 1).astype(int)
            for x in range(WAVE_W):
                seg = mono[edges[x]:max(edges[x + 1], edges[x] + 1)]
                lo, hi = float(seg.min()), float(seg.max())
                d.line([(x, mid - hi / 32768 * (mid - 4)), (x, mid - lo / 32768 * (mid - 4))],
                       fill=(110, 170, 235))
        self.wave_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.wave_img, anchor="nw")
        self.playhead = self.canvas.create_line(0, 0, 0, WAVE_H, fill="#ffcc44", width=2, state="hidden")

    def _on_wave_click(self, e):
        if self.current_pcm is None:
            return
        dur = len(self.current_pcm) / self.current.rate
        self.play(pos=max(0.0, min(1.0, e.x / WAVE_W)) * dur)

    # ---------------------------------------------------------- playback --
    def _space(self, e):
        if isinstance(e.widget, (tk.Entry, ttk.Entry)):
            return
        self.toggle_play()
        return "break"

    def toggle_play(self):
        if self.player.playing:
            self.player.stop()
        else:
            self.play()

    def play(self, pos=0.0):
        s = self._selected_sound() or self.current
        if s is None:
            return
        if s is not self.current:
            self._on_select()
        if self.current_pcm is None:
            self._load(s, autoplay=True)
            return
        try:
            self.player.play(self.current_pcm, s.rate, pos, self.loop_var.get())
        except Exception as ex:
            messagebox.showerror(APP_TITLE, f"Could not play the sound:\n{ex}")

    def _tick(self):
        pos = self.player.position()
        self.play_btn.config(text="Stop  (Space)" if pos is not None else "Play  (Space)")
        if self.current_pcm is not None and self.playhead is not None:
            dur = len(self.current_pcm) / self.current.rate
            if pos is None:
                self.canvas.itemconfigure(self.playhead, state="hidden")
            else:
                x = pos / dur * WAVE_W if dur else 0
                self.canvas.coords(self.playhead, x, 0, x, WAVE_H)
                self.canvas.itemconfigure(self.playhead, state="normal")
                self.time_lbl.config(text=f"{fmt_time(pos)} / {fmt_time(dur)}")
        self.after(40, self._tick)

    # ------------------------------------------------------------ export --
    def _sounds_under(self, iids):
        out, seen, used = [], set(), set()

        def walk(iid):
            if iid in self.leaf:
                s = self.leaf[iid]
                if id(s) not in seen:
                    seen.add(id(s))
                    rel = self._export_path(iid, s)
                    if rel.lower() in used:            # different sounds with the same name
                        rel = rel[:-4] + f"_{s.index}_{s.data_off:x}.wav"
                    used.add(rel.lower())
                    out.append((s, rel))
            else:
                for c in self.tree.get_children(iid):
                    walk(c)
        for i in iids:
            walk(i)
        return out

    def _export_path(self, iid, s):
        """Mirrors the list: category (or file) folders, then <name>.wav."""
        parts = []
        p = self.tree.parent(iid)
        while p:
            parts.insert(0, p.split("\x1f")[-1])
            p = self.tree.parent(p)
        return os.path.join(*[da.safe(x) for x in parts], da.safe(s.name) + ".wav")

    def export_current(self):
        if self.current is None:
            return
        s = self.current
        path = filedialog.asksaveasfilename(defaultextension=".wav", initialfile=da.safe(s.name) + ".wav",
                                            initialdir=self.cfg.get("last_export") or os.getcwd(),
                                            filetypes=[("WAV audio", "*.wav")])
        if path:
            self.cfg["last_export"] = os.path.dirname(path)
            self._run_export([(s, os.path.basename(path))], os.path.dirname(path))

    def export(self, all_listed=False):
        if self.busy or not self.sounds:
            return
        iids = self.tree.get_children("") if all_listed else self.tree.selection()
        items = self._sounds_under(iids)
        if not items:
            messagebox.showinfo(APP_TITLE, "Select sounds or folders in the list first." if not all_listed
                                else "Nothing is listed.")
            return
        d = filedialog.askdirectory(title=f"Export {len(items):,} sound(s) to...",
                                    initialdir=self.cfg.get("last_export") or os.getcwd())
        if d:
            self.cfg["last_export"] = d
            self._save_cfg()
            self._run_export(items, d)

    def _run_export(self, items, out_dir):
        self.busy = True
        self.stop_flag.clear()
        self.stop_btn.config(state="normal")
        self.prog.configure(maximum=len(items), value=0)

        def work():
            fails, done = [], 0
            for i, (s, rel) in enumerate(items):
                if self.stop_flag.is_set():
                    break
                dst = os.path.join(out_dir, rel)
                try:
                    key = (s.path, s.data_off)
                    pcm = self.cache.get(key)
                    if pcm is None:
                        pcm = s.load()
                    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                    da.write_wav(dst, pcm, s.rate)
                    done += 1
                except Exception as ex:
                    fails.append(f"{s.name}: {ex}")
                self.q.put(("progress", i + 1, len(items), "Exporting"))
            self.q.put(("export_done", done, fails, out_dir, self.stop_flag.is_set()))
        threading.Thread(target=work, daemon=True).start()

    # -------------------------------------------------------------- queue --
    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, i, n, text = msg
                    self.prog.configure(maximum=max(n, 1), value=i)
                    self.status.config(text=f"{text}   ({i:,}/{n:,})")
                elif kind == "scanned":
                    self._scanned(*msg[1:])
                elif kind == "loaded":
                    self._loaded(msg[1], msg[2], msg[3])
                elif kind == "load_err":
                    if msg[1] is self.current:
                        self._draw_empty_wave(f"Cannot play: {msg[2]}")
                elif kind == "export_done":
                    self.busy = False
                    self.stop_btn.config(state="disabled")
                    done, fails, out_dir, stopped = msg[1:]
                    self.prog.configure(value=0)
                    self.status.config(text=f"Exported {done:,} sound(s) to {out_dir}"
                                       + (" (stopped)" if stopped else "")
                                       + (f" - {len(fails)} failed" if fails else ""))
                    if fails:
                        for f in fails:
                            print("export failed:", f, file=sys.stderr)
                        messagebox.showwarning(APP_TITLE, f"{len(fails)} sound(s) could not be exported:\n\n"
                                               + "\n".join(fails[:15]) + ("\n..." if len(fails) > 15 else ""))
        except queue.Empty:
            pass
        self.after(50, self._poll)

    def _close(self):
        self.stop_flag.set()
        self._save_cfg()
        self.player.close()
        self.destroy()


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    App(folder).mainloop()


if __name__ == "__main__":
    main()
