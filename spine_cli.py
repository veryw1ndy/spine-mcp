"""spine_cli — thin wrappers around the licensed Spine 4.3 command-line tool.

The Spine editor ships a headless CLI that (on an activated license) can pack
atlases, import a runtime skeleton json into an editable .spine project, export a
project back to runtime, and print project info. We drive it via subprocess.
"""
from __future__ import annotations
import os, subprocess

# Local setup: the native Spine.app binary must never be invoked from this server.
SPINE_BIN = os.environ.get("SPINE_BIN", os.path.expanduser("~/spine-probe/spine-cli"))
if SPINE_BIN.startswith("/Applications/Spine.app"):
    raise RuntimeError("The native Spine.app binary must not be used")
_NOISE = ("Spine Launcher", "Esoteric Software", "Mac OS X", "Starting:",
          "Spine 4.3", "Licensed to:")


def available() -> bool:
    return os.path.exists(SPINE_BIN)


def _run(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    # Longer default timeout: every wrapper run starts Spine under Wine (~15 s).
    return subprocess.run([SPINE_BIN, *args], capture_output=True, text=True, timeout=timeout)


def _clean(out: str) -> str:
    return "\n".join(l for l in out.splitlines() if not any(l.startswith(p) for p in _NOISE)).strip()


def version() -> str:
    if not available():
        return "Spine CLI not found at " + SPINE_BIN
    out = _run(["--version"]).stdout
    for line in out.splitlines():
        if "Professional" in line or "Trial" in line or "Essential" in line:
            return line.strip().removeprefix("Starting:").strip()
    return out.strip().splitlines()[-1] if out.strip() else "n/a"


def info(project_or_data: str) -> str:
    """Print bones/slots/animations of a .spine project or skeleton .json."""
    return _clean(_run(["-i", project_or_data]).stdout)


def pack_atlas(images_dir: str, out_dir: str, name: str) -> dict:
    """Pack a folder of PNGs into <name>.atlas + <name>.png using Spine's packer.

    Pages up to 4096: the 2048 default produced TWO pages for large units, and a
    game export that copies or renames only the first one 404s on the second —
    every Spine animation in the game silently degrading to a static sprite.

    Passed with --set rather than written to disk. The old version dropped a
    pack.json into the caller's own images folder and removed it in a finally,
    so a process killed in between left litter in a directory this tool does not
    own. The CLI takes these as flags, so there is no file to leave behind and
    an existing pack.json of the caller's is neither read nor overwritten."""
    os.makedirs(out_dir, exist_ok=True)
    r = _run(["-i", images_dir, "-o", out_dir, "-n", name, "-p", name,
              "--set", "maxWidth=4096", "--set", "maxHeight=4096", "--set", "pot=false"])
    atlas = os.path.join(out_dir, f"{name}.atlas")
    ok = os.path.exists(atlas)
    pages = 0
    if ok:
        with open(atlas) as f:
            pages = sum(1 for l in f if l.strip().endswith(".png"))
    return {"ok": ok, "atlas": atlas, "pages": pages,
            "png": os.path.join(out_dir, f"{name}.png"), "log": _clean(r.stdout)}


def make_project(runtime_json: str, out_spine: str) -> dict:
    """Import a runtime skeleton json into an EDITABLE .spine project (-r)."""
    os.makedirs(os.path.dirname(out_spine) or ".", exist_ok=True)
    r = _run(["-i", runtime_json, "-o", out_spine, "-r"])
    return {"ok": os.path.exists(out_spine), "project": out_spine, "log": _clean(r.stdout)}


def export_project(project: str, out_dir: str, fmt: str = "json+pack") -> dict:
    """Export a .spine project back to runtime (json/binary [+pack])."""
    os.makedirs(out_dir, exist_ok=True)
    r = _run(["-i", project, "-o", out_dir, "-e", fmt])
    return {"ok": r.returncode == 0, "out_dir": out_dir, "log": _clean(r.stdout)}
