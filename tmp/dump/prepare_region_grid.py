#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, shutil
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import numpy as np

DIRS={"north":(0,-1),"east":(1,0),"south":(0,1),"west":(-1,0)}

def load_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def _json_safe(v):
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {k: _json_safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(val) for val in v]
    return v

def save_json(p,d):
    Path(p).write_text(
        json.dumps(_json_safe(d), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def resize_h(im,h):
    if im.height==h: return im.copy()
    return im.resize((max(1,round(im.width*h/im.height)),h),Image.LANCZOS)

def discover(jobs):
    found={}
    for d in jobs.iterdir():
        if not d.is_dir(): continue
        rj=d/"region.json"
        if not rj.exists(): continue
        try:
            m=load_json(rj)
            p=Path(m["image"])
            if not p.is_absolute(): p=d/p
            if p.exists():
                found[(int(m["x"]),int(m["y"]))]={"id":m.get("id",d.name),"image":p.resolve(),"job":d.resolve()}
        except Exception: pass
    return found

def neighbors(regions,x,y):
    out={}
    for s,(dx,dy) in DIRS.items():
        r=regions.get((x+dx,y+dy))
        if r: out[s]=r
    return out

def board(neigh,out):
    W,H=1260,780
    b=Image.new("RGB",(W,H),(18,17,15)); d=ImageDraw.Draw(b)
    boxes={"north":(420,0,840,260),"west":(0,260,420,520),"east":(840,260,1260,520),"south":(420,520,840,780)}
    d.rectangle((420,260,840,520),outline=(160,130,80),width=3)
    d.text((545,380),"NEW REGION",fill=(220,205,170))
    for s,r in neigh.items():
        im=Image.open(r["image"]).convert("RGB")
        t=ImageOps.contain(im,(420,260),Image.LANCZOS)
        x0,y0,x1,y1=boxes[s]
        b.paste(t,(x0+(420-t.width)//2,y0+(260-t.height)//2))
        d.text((x0+8,y0+8),f"{s.upper()} {r['id']}",fill=(245,235,210))
    b.save(out)

def prompt(primary,neigh,target_w,target_h,overlap,context):
    secondary=[f"- {s.upper()}: {r['id']}" for s,r in neigh.items() if s!=primary]
    return f"""Create one continuous regional map expansion for The Frontier.

TARGET REGION GEOMETRY
- target region width: {target_w}px
- target region height: {target_h}px
- primary neighbor: {primary.upper()}
- shared overlap: {overlap}px
- source context: {context}px

The primary generation/stitch workflow is the same proven SHARED OVERLAP method used for the existing WEST chain.
Preserve its style, camera, object scale, forest density, rivers, roads, mountains, lighting and atmospheric depth.

OTHER EXISTING NEIGHBORS
{chr(10).join(secondary) if secondary else '- none'}

Use context_board.png and neighbor reference files to continue geography toward all existing neighbors.
The new region must match the width/height implied by the surrounding grid.
Do not create a visible seam.

STYLE
- high oblique 3/4 aerial regional map
- realistic painterly medieval/frontier strategy map
- northern conifer forests, rocky ridges, rivers, roads, meadows, sparse frontier traces

STRICT
- no UI, text, labels, borders or markers
- no hard panel division
- no mirrored copy
- keep one continuous handcrafted world
"""

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--jobs",default="jobs")
    ap.add_argument("--x",type=int,required=True)
    ap.add_argument("--y",type=int,required=True)
    ap.add_argument("--primary",choices=DIRS,required=True)
    ap.add_argument("--job",required=True)
    ap.add_argument("--id",default=None)
    ap.add_argument("--overlap",type=int,default=256)
    ap.add_argument("--context",type=int,default=700)
    args=ap.parse_args()

    jobs=Path(args.jobs).resolve()
    regs=discover(jobs)
    neigh=neighbors(regs,args.x,args.y)
    if args.primary not in neigh:
        raise RuntimeError(f"Primary {args.primary} missing. Available: {list(neigh)}")

    # Geometry from surrounding grid:
    # width should match NORTH/SOUTH column when available;
    # height should match WEST/EAST row when available.
    if "north" in neigh:
        target_w=Image.open(neigh["north"]["image"]).width
    elif "south" in neigh:
        target_w=Image.open(neigh["south"]["image"]).width
    else:
        target_w=Image.open(neigh[args.primary]["image"]).width

    if "west" in neigh:
        target_h=Image.open(neigh["west"]["image"]).height
    elif "east" in neigh:
        target_h=Image.open(neigh["east"]["image"]).height
    else:
        target_h=Image.open(neigh[args.primary]["image"]).height

    primary=neigh[args.primary]
    src=Image.open(primary["image"]).convert("RGB")

    overlap=min(args.overlap, max(32, (target_w if args.primary in ("west","east") else target_h)//3))
    context=args.context

    if args.primary in ("west","east"):
        # region tile width = core + overlap = target_w
        core=max(32,target_w-overlap)
        src=resize_h(src,target_h)
        a=np.array(src); h,w,_=a.shape
        context=min(context,w)
        total=core+overlap+context
        canvas=np.zeros((h,total,3),np.uint8)
        mask=np.zeros((h,total),np.uint8)
        if args.primary=="west":
            canvas[:,:context]=a[:,w-context:]
            mask[:,context:]=255
            direction="right"
        else:
            canvas[:,core+overlap:]=a[:,:context]
            mask[:,:core+overlap]=255
            direction="left"
    else:
        # region tile height = core + overlap = target_h
        core=max(32,target_h-overlap)
        # primary width must match target column width
        src=src.resize((target_w, max(1,round(src.height*target_w/src.width))),Image.LANCZOS)
        a=np.array(src); h,w,_=a.shape
        context=min(context,h)
        total=core+overlap+context
        canvas=np.zeros((total,w,3),np.uint8)
        mask=np.zeros((total,w),np.uint8)
        if args.primary=="north":
            canvas[:context]=a[h-context:]
            mask[context:]=255
            direction="down"
        else:
            canvas[core+overlap:]=a[:context]
            mask[:core+overlap]=255
            direction="up"

    job=Path(args.job)
    inp=job/"input"; refs=inp/"neighbors"; prev=job/"preview"
    refs.mkdir(parents=True,exist_ok=True); prev.mkdir(parents=True,exist_ok=True)
    Image.fromarray(canvas).save(inp/"generation_canvas.png")
    Image.fromarray(mask).save(inp/"generation_mask.png")
    src.save(inp/"source_region_native.png")
    shutil.copy2(primary["image"],inp/"source_region.png")
    for s,r in neigh.items():
        if s!=args.primary: shutil.copy2(r["image"],refs/f"{s}.png")
    board(neigh,inp/"context_board.png")
    (inp/"prompt.txt").write_text(prompt(args.primary,neigh,target_w,target_h,overlap,context),encoding="utf-8")

    meta={
        "version":3,"method":"grid_shared_overlap",
        "region_id":args.id or job.name,
        "target":{"x":args.x,"y":args.y},
        "primary_side":args.primary,"primary_neighbor":primary,
        "secondary_neighbors":{s:r for s,r in neigh.items() if s!=args.primary},
        "direction":direction,
        "target_region_size":[target_w,target_h],
        "expected_generation_size":[canvas.shape[1],canvas.shape[0]],
        "core":core,"overlap":overlap,"context":context,
        "source_native_size":[src.width,src.height]
    }
    save_json(job/"grid_job.json",meta)
    Image.fromarray(canvas).save(prev/"prepare_overview.png")
    print(f"target={args.x},{args.y} size={target_w}x{target_h}")
    print(f"neighbors={','.join(neigh)}")
    print(f"canvas={canvas.shape[1]}x{canvas.shape[0]}")

if __name__=="__main__":
    main()
