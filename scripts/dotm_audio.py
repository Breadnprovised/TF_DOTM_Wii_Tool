#!/usr/bin/env python3
"""
dotm_audio.py - audio reader / converter for Transformers: Dark of the Moon (Wii)

The game's audio is stored as FMOD sound banks (FSB4, Nintendo DSP ADPCM):
  <load unit>/<load unit>.str   music, ambience and in-game cutscene audio
  <load unit>/en.str, fr.str    dialogue, one file per language
  <load unit>/<load unit>.sbk   sound effects (read the same way, if they are FSB4)
A .str file is a run of FSB4 banks, one sound each; each sound's name is
stored in its bank.

    python dotm_audio.py "D:\\Wii\\transformers dark of the moon" [OUT_DIR]

converts every sound to a standard 16-bit WAV file in OUT_DIR (default:
converted_audio next to the launchers), keeping the game's folder layout.
"""

import os
import re
import struct
import sys

import numpy as np

AUDIO_EXTS = (".str", ".sbk")
LANGS = {"en": "English", "fr": "French", "de": "German", "es": "Spanish", "it": "Italian"}
FSOUND_GCADPCM = 0x02000000


class AudioError(Exception):
    pass


class Sound:
    """One sound inside an FSB4 bank."""
    __slots__ = ("path", "rel", "bank_off", "index", "name", "samples", "rate", "channels",
                 "data_off", "data_size", "mode", "coefs", "hist", "loop", "lang", "lu")

    @property
    def duration(self):
        return self.samples / self.rate if self.rate else 0.0

    @property
    def stem(self):
        """Name without the language suffix (vo_..._03_en -> vo_..._03)."""
        n = self.name
        if self.lang and n.lower().endswith("_" + self.lang):
            n = n[:-len(self.lang) - 1]
        return n

    @property
    def category(self):
        return categorise(self.stem)

    @property
    def codec(self):
        return "DSP ADPCM" if self.mode & FSOUND_GCADPCM else f"mode 0x{self.mode:x}"

    def read(self):
        with open(self.path, "rb") as f:
            f.seek(self.data_off)
            return f.read(self.data_size)

    def load(self):
        """-> int16 PCM array (samples, channels)."""
        return decode(self, self.read())


def parse_fsb4(f, off, path):
    """Sounds of the FSB4 bank at `off` in open file `f`, and the bank's total size."""
    f.seek(off)
    h = f.read(0x30)
    if len(h) < 0x30 or h[:4] != b"FSB4":
        raise AudioError(f"no FSB4 bank at 0x{off:x}")
    nsamp, shsize, dsize, ver, fmode = struct.unpack_from("<5I", h, 4)
    if nsamp == 0 or nsamp > 100000 or shsize > 64 * 1024 * 1024:
        raise AudioError(f"bad FSB4 header at 0x{off:x}")
    sh = f.read(shsize)
    out = []
    p = 0
    data = off + 0x30 + shsize
    for k in range(nsamp):
        if p + 0x50 > len(sh):
            raise AudioError(f"truncated FSB4 sample header at 0x{off:x}")
        size = struct.unpack_from("<H", sh, p)[0]
        s = Sound()
        s.path, s.bank_off, s.index = path, off, k
        s.name = sh[p + 2:p + 0x20].split(b"\0")[0].decode("latin-1")
        s.samples, clen, ls, le, s.mode, s.rate = struct.unpack_from("<6I", sh, p + 0x20)
        s.channels = struct.unpack_from("<H", sh, p + 0x3E)[0] or 1
        s.loop = (ls, le)
        s.coefs, s.hist = [], []
        for c in range(s.channels):           # DSP coefficients follow the 0x50-byte header
            q = p + 0x50 + c * 0x2E
            if q + 0x2E <= p + size:
                s.coefs.append(struct.unpack_from(">16h", sh, q))
                s.hist.append(struct.unpack_from(">hh", sh, q + 0x24))
        s.data_off, s.data_size = data, clen
        # sounds in a bank follow each other, each padded to 32 bytes
        data += (clen + 31) // 32 * 32
        out.append(s)
        p += size
    return out, 0x30 + shsize + dsize


def scan_file(path, root=None):
    """All sounds in a .str / .sbk file (FSB4 banks one after another;
    other data between banks is skipped by searching for the next one)."""
    rel = os.path.relpath(path, root) if root else os.path.basename(path)
    base = os.path.splitext(os.path.basename(path))[0].lower()
    lang = base if base in LANGS else ""
    lu = os.path.basename(os.path.dirname(path))
    sounds, size = [], os.path.getsize(path)
    with open(path, "rb") as f:
        off = 0
        while off + 0x30 <= size:
            f.seek(off)
            if f.read(4) != b"FSB4":
                # find the next bank (32-byte aligned)
                f.seek(off)
                chunk = f.read(1 << 20)
                i = chunk.find(b"FSB4", 4)
                while i >= 0 and (off + i) % 32:
                    i = chunk.find(b"FSB4", i + 1)
                if i < 0:
                    if len(chunk) < (1 << 20):
                        break
                    off += len(chunk) - 4
                    continue
                off += i
                continue
            got, total = parse_fsb4(f, off, path)
            for s in got:
                s.rel, s.lang, s.lu = rel, lang, lu
            sounds += got
            off += (total + 31) // 32 * 32
    if not sounds and size:
        raise AudioError("no FSB4 sound banks in this file")
    return sounds


def find_audio_files(root):
    out = []
    for d, dirs, files in os.walk(root):
        dirs.sort()
        for n in sorted(files):
            if n.lower().endswith(AUDIO_EXTS):
                out.append(os.path.join(d, n))
    return out


# --------------------------------------------------------------- naming --
# speaker codes in dialogue names; only the ones that are certain are named,
# the others are listed by their code
SPEAKERS = {"opt": "Optimus Prime", "mgt": "Megatron", "sdw": "Soundwave", "lkd": "Lockdown",
            "mir": "Mirage", "whj": "Wheeljack", "bbb": "Bumblebee"}
FX_GROUPS = (("Interface", ("ui", "hud")),
             ("Characters", ("all", "mgt", "opt", "decpt", "bbb", "irn", "mir", "sdw", "lkd")),
             ("Weapons & pickups", ("turret", "hero", "genericmissile", "dfa", "dfw", "ammo",
                                    "power", "collectible")))


def speaker(code):
    return f"{SPEAKERS[code]} ({code})" if code in SPEAKERS else code


def categorise(name):
    """Category path for a sound name, from its prefix."""
    n = name.lower()
    m = re.match(r"vo_mis_m(\d+)_([a-z0-9]+)_", n)
    if m:
        return ("Dialogue", "Missions", f"Mission {m.group(1)}", speaker(m.group(2)))
    m = re.match(r"vo_bark_([a-z0-9]+)_", n)
    if m:
        return ("Dialogue", "Combat barks", speaker(m.group(1)))
    if n.startswith("vo_"):
        return ("Dialogue", "Other")
    if n.startswith("mu_"):
        return ("Music", "Menus") if n.startswith("mu_menu") else ("Music",)
    if n.startswith("am_"):
        return ("Ambience",)
    if n.startswith("igc_"):
        return ("Cutscenes",)
    if n.startswith("fx_"):
        word = n.split("_")[1] if "_" in n else ""
        for group, words in FX_GROUPS:
            if word in words or (word == "ammo" and "pickup" in n):
                return ("Sound effects", group)
        return ("Sound effects", "Props & level")
    return ("Other",)


# --------------------------------------------------------------- decode --
def decode_dsp(frames, coefs, nsamples, yn1=0, yn2=0):
    """Nintendo DSP ADPCM: 8-byte frames, 1 header byte (predictor index in
    the high nibble, scale exponent in the low one) + 14 4-bit samples."""
    nframes = len(frames) // 8
    a = np.frombuffer(frames[:nframes * 8], np.uint8).reshape(nframes, 8)
    head = a[:, 0]
    nib = np.empty((nframes, 14), np.int32)
    nib[:, 0::2] = a[:, 1:] >> 4
    nib[:, 1::2] = a[:, 1:] & 15
    nib = np.where(nib >= 8, nib - 16, nib)
    scaled = (nib << (head & 15)[:, None].astype(np.int32)) << 11     # sample * scale, in 1/2048
    pred = np.minimum(head >> 4, 7)
    c1 = np.array(coefs[0::2], np.int64)[pred].tolist()
    c2 = np.array(coefs[1::2], np.int64)[pred].tolist()
    sc = scaled.tolist()
    out = [0] * (nframes * 14)
    k = 0
    for fi in range(nframes):
        a1, a2, row = c1[fi], c2[fi], sc[fi]
        for s in row:
            v = (s + a1 * yn1 + a2 * yn2 + 1024) >> 11
            v = -32768 if v < -32768 else (32767 if v > 32767 else v)
            out[k] = v
            k += 1
            yn2, yn1 = yn1, v
    return np.array(out[:nsamples], np.int16)


def decode(s, data):
    """PCM (samples, channels). FSB4 interleaves the channels of DSP ADPCM
    sounds every 2 bytes (verified: every frame header of both channels of
    the game's stereo music is valid only with that layout)."""
    if not s.mode & FSOUND_GCADPCM:
        raise AudioError(f"unsupported FMOD format (mode 0x{s.mode:x})")
    nch = s.channels
    if len(s.coefs) < nch:
        raise AudioError("missing DSP coefficients")
    if nch == 1:
        chans = [data]
    else:
        n = len(data) // (2 * nch) * (2 * nch)
        a = np.frombuffer(data[:n], np.uint8).reshape(-1, nch, 2)
        chans = [a[:, c].tobytes() for c in range(nch)]
    pcm = [decode_dsp(chans[c], s.coefs[c], s.samples, *s.hist[c]) for c in range(nch)]
    n = min(len(p) for p in pcm)
    return np.stack([p[:n] for p in pcm], axis=1)


def write_wav(path, pcm, rate):
    """Standard little-endian 16-bit PCM WAV."""
    nch = pcm.shape[1]
    body = pcm.astype("<i2").tobytes()
    chunks = b"fmt " + struct.pack("<IHHIIHH", 16, 1, nch, rate, rate * nch * 2, nch * 2, 16)
    chunks += b"data" + struct.pack("<I", len(body)) + body
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)


def safe(name):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .") or "_"


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    root = argv[0]
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.dirname(here) if os.path.basename(here).lower() == "scripts" else here
    out = argv[1] if len(argv) > 1 else os.path.join(base, "converted_audio")
    files = find_audio_files(root)
    done = fails = 0
    for path in files:
        try:
            sounds = scan_file(path, root)
        except (OSError, AudioError) as ex:
            print(f"skipped {os.path.relpath(path, root)}: {ex}", file=sys.stderr)
            continue
        d = os.path.join(out, os.path.splitext(os.path.relpath(path, root))[0])
        os.makedirs(d, exist_ok=True)
        used = set()
        for s in sounds:
            name = safe(s.name)
            if name.lower() in used:
                name += f"_{s.index}"
            used.add(name.lower())
            try:
                write_wav(os.path.join(d, name + ".wav"), s.load(), s.rate)
                done += 1
            except Exception as ex:
                print(f"failed {s.rel} {s.name}: {ex}", file=sys.stderr)
                fails += 1
        print(f"{os.path.relpath(path, root)}: {len(sounds)} sound(s)")
    print(f"converted {done} sound(s) to {out}" + (f", {fails} failed" if fails else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
