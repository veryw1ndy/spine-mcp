#!/usr/bin/env python
"""Spine MCP server — make production Spine 2D animations from cut-up characters.

Like ampersante/spine2d-animation-mcp, but it ships REAL runtime rigs that play in
spine-pixi/pixi games AND editable .spine projects (via the licensed Spine 4.3
CLI), with hand-crafted motion instead of canned templates.

Pipeline:  cut parts (PSD or PhotoshopToSpine export)  ->  rig_and_animate
           ->  runtime json + atlas + editable .spine  ->  preview / wire into game

Tools:
  spine_doctor          check Spine CLI + deps
  inspect_source        list parts / detected head-states before rigging
  rig_and_animate       PSD or export folder -> rig + anims + atlas (+ .spine)
  pack_atlas            Spine-CLI atlas pack of an images folder
  make_project          runtime json -> editable .spine project
  export_project        .spine project -> runtime json + atlas
  project_info          bones/slots/anims of a .spine or json
  preview               render a keyframe-montage PNG of a rig
  batch                 rig every subfolder of a roster directory
"""
from __future__ import annotations
import os, glob, json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # importable from any cwd
from mcp.server.fastmcp import FastMCP

import spine_rig
import spine_cli
import spine_preview

mcp = FastMCP("spine")


@mcp.tool()
def spine_doctor() -> dict:
    """Report whether the Spine CLI and Python deps are available."""
    deps = {}
    for m in ("PIL", "psd_tools"):
        try:
            __import__(m); deps[m] = True
        except Exception:
            deps[m] = False
    return {"spine_cli": spine_cli.available(), "spine_bin": spine_cli.SPINE_BIN,
            "spine_version": spine_cli.version() if spine_cli.available() else None,
            "deps": deps}


@mcp.tool()
def inspect_source(source: str) -> dict:
    """List the parts and detected head-state families of a .psd or a
    PhotoshopToSpine export folder, WITHOUT building anything. Use this first to
    confirm what will be rigged."""
    if source.lower().endswith(".psd"):
        import tempfile
        parts, draw, _ = spine_rig._read_psd(source, tempfile.mkdtemp())
    else:
        parts, draw, _ = spine_rig._read_photoshop_export(source)
    fam = {}
    for n in draw:
        base = n
        for s in spine_rig.SUFFIX:
            if n.lower().endswith(s):
                base = n[: -len(s)]
        fam.setdefault(base, []).append(n)
    families = {b: v for b, v in fam.items() if len(v) > 1}
    return {"parts": draw, "count": len(draw), "head_state_families": families}


@mcp.tool()
def rig_and_animate(source: str, out_dir: str, name: str = "", kind: str = "symbol",
                    anims: list[str] | None = None, make_editable: bool = True) -> dict:
    """Build a rigged + animated Spine skeleton from a cut-up character.

    source        path to a .psd OR a PhotoshopToSpine export folder
    out_dir       where to write <name>.json/.atlas/.png (e.g. a game's
                  static/assets/spine/<name>/)
    name          skeleton name (defaults to the source basename)
    kind          "symbol" or "mascot" (reserved; both rig the same way now)
                  Layers named as limbs (arm_l, forearm_r, hand_l, thigh_r,
                  shin_l, foot_r, 左大臂, 右小腿 ...) get bone chains with joints
                  found from the art; see spine_rig.LIMB_WORDS. The summary
                  lists them under "limbs" and any skipped layers under "warnings".
    anims         subset of ["idle","win","blink","pop"] (default all applicable)
    make_editable also emit an editable <name>.spine next to the source (Spine CLI)

    Returns a summary incl. file paths and an editable-project path."""
    res = spine_rig.build_rig(source, out_dir, name or None, kind, anims)
    if make_editable and spine_cli.available():
        src_dir = source if os.path.isdir(source) else os.path.dirname(source)
        proj = os.path.join(src_dir, f"{res['name']}.spine")
        res["editable_project"] = spine_cli.make_project(res["files"]["json"], proj)
    return res


@mcp.tool()
def pack_atlas(images_dir: str, out_dir: str, name: str) -> dict:
    """Pack a folder of PNGs into <name>.atlas + <name>.png with the Spine packer."""
    return spine_cli.pack_atlas(images_dir, out_dir, name)


@mcp.tool()
def make_project(runtime_json: str, out_spine: str) -> dict:
    """Import a runtime skeleton json into an EDITABLE .spine project."""
    return spine_cli.make_project(runtime_json, out_spine)


@mcp.tool()
def export_project(project: str, out_dir: str, fmt: str = "json+pack") -> dict:
    """Export a .spine project to runtime files (fmt: json|binary, +pack for atlas)."""
    return spine_cli.export_project(project, out_dir, fmt)


@mcp.tool()
def project_info(project_or_json: str) -> str:
    """Print bones/slots/animations of a .spine project or skeleton .json."""
    return spine_cli.info(project_or_json)


@mcp.tool()
def preview(rig_dir: str, images_dir: str = "", out_png: str = "", maxpx: int = 200) -> dict:
    """Render a keyframe-montage PNG of a built rig (idle/win/blink/pop poses).

    rig_dir      folder holding <name>.json (the out_dir from rig_and_animate)
    images_dir   folder with the part PNGs (defaults to the source export images)
    out_png      output path (defaults to <rig_dir>/_preview.png)"""
    rig_json = sorted(glob.glob(f"{rig_dir}/*.json"))[0]
    name = os.path.splitext(os.path.basename(rig_json))[0]
    if not images_dir:
        # try the atlas-sibling images, else fall back to the rig dir itself
        for cand in (os.path.join(rig_dir, "images"), rig_dir):
            if glob.glob(os.path.join(cand, "*.png")):
                images_dir = cand; break
    out = out_png or os.path.join(rig_dir, "_preview.png")
    path = spine_preview.montage(rig_json, images_dir, out, maxpx)
    return {"preview": path, "name": name}


@mcp.tool()
def batch(roster_dir: str, out_root: str, kind: str = "symbol",
          make_editable: bool = True) -> dict:
    """Rig every PhotoshopToSpine export subfolder under roster_dir into
    out_root/<name>/. Returns per-character summaries."""
    results, errors = [], []
    for d in sorted(glob.glob(f"{roster_dir}/*/")):
        d = d.rstrip("/")
        if not (glob.glob(f"{d}/*.json") and os.path.isdir(f"{d}/images")):
            continue
        nm = os.path.basename(d)
        try:
            results.append(rig_and_animate(d, os.path.join(out_root, nm), nm, kind,
                                           None, make_editable))
        except Exception as e:
            errors.append({"name": nm, "error": str(e)})
    return {"rigged": [r["name"] for r in results], "count": len(results),
            "errors": errors, "results": results}


def main() -> None:
    """Console entry point (pyproject [project.scripts]).

    Named rather than left as a bare __main__ block so the package can be
    launched as  from anywhere — which is what Provide a command to run with `uvx <command>`.

See `uvx --help` for more information.
    and the mozg plugin both do. A module-path launch only ever worked for
    whoever had the checkout.
    """
    mcp.run()


if __name__ == "__main__":
    main()
