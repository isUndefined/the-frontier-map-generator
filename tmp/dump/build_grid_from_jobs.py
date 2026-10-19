#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, re, shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

WEST_RE = re.compile(r"^west_(\d+)(?:_h\d+)?$", re.I)

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def save_json(p: Path, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def west_num(name: str):
    m = WEST_RE.fullmatch(name)
    return int(m.group(1)) if m else None

def find_west_job(jobs: Path, n: int):
    candidates=[]
    for d in jobs.iterdir():
        if d.is_dir() and west_num(d.name)==n:
            candidates.append(d)
    # Prefer the non-h-suffixed folder, then shortest name.
    candidates.sort(key=lambda p: ("_h" in p.name.lower(), len(p.name)))
    return candidates[0] if candidates else None

def find_final(job: Path):
    for p in [job/"stitched"/"final_native.png", job/"final_native.png"]:
        if p.exists():
            return p
    return None

def find_center_source(west1: Path):
    # CENTER means the ORIGINAL STARTING MAP.
    for p in [
        west1/"input"/"source_region_native.png",
        west1/"input"/"source_region.png",
    ]:
        if p.exists():
            return p
    return None

def resize_h(im: Image.Image, h: int):
    if im.height == h:
        return im.copy()
    return im.resize((max(1, round(im.width*h/im.height)), h), Image.LANCZOS)

def extract_west_from_cumulative(current_final: Path, previous_world: Path, out: Path):
    """
    Legacy WEST_N final_native is cumulative:
      [ WEST_N ][ PREVIOUS WORLD ]
    We derive WEST_N by subtracting the previous cumulative world's width
    after scaling previous_world to the exact current height.
    """
    cur = Image.open(current_final).convert("RGB")
    prev = Image.open(previous_world).convert("RGB")
    prev_at_h = resize_h(prev, cur.height)

    region_w = cur.width - prev_at_h.width
    if region_w <= 16:
        raise RuntimeError(
            f"Cannot derive WEST tile: current={cur.size}, previous@height={prev_at_h.size}, "
            f"calculated width={region_w}"
        )

    region = cur.crop((0, 0, region_w, cur.height))
    ensure_dir(out.parent)
    region.save(out)
    return out

def discover_new_grid_regions(jobs: Path):
    found={}
    for d in jobs.iterdir():
        if not d.is_dir() or d.name.startswith("_") or west_num(d.name):
            continue
        rj=d/"region.json"
        if not rj.exists():
            continue
        try:
            m=load_json(rj)
            x,y=int(m["x"]),int(m["y"])
            ip=Path(m["image"])
            if not ip.is_absolute():
                ip=d/ip
            if ip.exists():
                found[(x,y)]={"id":m.get("id",d.name),"image":ip.resolve(),"job":d.resolve()}
        except Exception:
            pass
    return found

def make_preview(regions, out):
    """Clean preview: no frames, no labels, no coordinates."""
    if not regions:
        return

    xs=sorted({x for x,y in regions})
    ys=sorted({y for x,y in regions})

    col_w={}
    for x in xs:
        top=regions.get((x,0))
        if top:
            im=Image.open(top["image"])
            col_w[x]=im.width
        else:
            vals=[Image.open(r["image"]).width for (rx,ry),r in regions.items() if rx==x]
            col_w[x]=max(vals) if vals else 1

    row_h={}
    for y in ys:
        vals=[Image.open(r["image"]).height for (rx,ry),r in regions.items() if ry==y]
        row_h[y]=max(vals) if vals else 1

    raw_w=sum(col_w[x] for x in xs)
    scale=min(1.0, 1500/max(1,raw_w))
    draw_col={x:max(80,round(col_w[x]*scale)) for x in xs}
    draw_row={y:max(70,round(row_h[y]*scale)) for y in ys}

    W=sum(draw_col.values())
    H=sum(draw_row.values())
    board=Image.new("RGB",(W,H),(18,17,15))

    x_pos={}
    xx=0
    for x in xs:
        x_pos[x]=xx
        xx += draw_col[x]

    y_pos={}
    yy=0
    for y in ys:
        y_pos[y]=yy
        yy += draw_row[y]

    for (x,y),r in regions.items():
        box_w,box_h=draw_col[x],draw_row[y]
        im=Image.open(r["image"]).convert("RGB")
        # Preserve full region; fit inside its grid cell without decorations.
        thumb=ImageOps.contain(im,(box_w,box_h),Image.LANCZOS)
        slot=Image.new("RGB",(box_w,box_h),(18,17,15))
        slot.paste(thumb,((box_w-thumb.width)//2,(box_h-thumb.height)//2))
        board.paste(slot,(x_pos[x],y_pos[y]))

    board.save(out)


def make_final_world(regions, out):
    """
    Native-resolution world mosaic.
    No borders, labels, coordinates or preview decorations.
    Missing grid cells remain transparent.
    """
    if not regions:
        return

    xs=sorted({x for x,y in regions})
    ys=sorted({y for x,y in regions})

    # Column width is defined primarily by row 0 (the horizontal backbone).
    col_w={}
    for x in xs:
        top=regions.get((x,0))
        if top:
            col_w[x]=Image.open(top["image"]).width
        else:
            vals=[Image.open(r["image"]).width for (rx,ry),r in regions.items() if rx==x]
            col_w[x]=max(vals) if vals else 1

    # Each row may have its own height.
    row_h={}
    for y in ys:
        vals=[Image.open(r["image"]).height for (rx,ry),r in regions.items() if ry==y]
        row_h[y]=max(vals) if vals else 1

    x_pos={}
    xx=0
    for x in xs:
        x_pos[x]=xx
        xx+=col_w[x]

    y_pos={}
    yy=0
    for y in ys:
        y_pos[y]=yy
        yy+=row_h[y]

    W=sum(col_w.values())
    H=sum(row_h.values())
    world=Image.new("RGBA",(W,H),(0,0,0,0))

    for (x,y),r in regions.items():
        im=Image.open(r["image"]).convert("RGBA")
        cw,ch=col_w[x],row_h[y]

        # Grid invariant: generated region should already match its column width.
        # If legacy data differs slightly, center it without rescaling.
        px=x_pos[x]+max(0,(cw-im.width)//2)
        py=y_pos[y]+max(0,(ch-im.height)//2)
        crop=im.crop((0,0,min(im.width,cw),min(im.height,ch)))
        world.alpha_composite(crop,(px,py))

    world.save(out)


def _find_latest_west_final(jobs_dir: Path):
    """Return the last WEST cumulative final_native.png."""
    candidates=[]
    for d in jobs_dir.iterdir():
        if not d.is_dir():
            continue
        n=west_num(d.name)
        if n is None:
            continue
        final=find_final(d)
        if final:
            candidates.append((n,d,final))
    if not candidates:
        return None
    candidates.sort(key=lambda t:t[0])
    return candidates[-1]


def _top_x_positions_from_cumulative(jobs_dir: Path, latest_height: int):
    """
    Compute exact top-row X positions using cumulative final widths.
    This preserves the already-good old WEST chain instead of reconstructing it.
    """
    west=[]
    for d in jobs_dir.iterdir():
        if not d.is_dir():
            continue
        n=west_num(d.name)
        if n is None:
            continue
        final=find_final(d)
        if not final:
            continue
        im=Image.open(final)
        scaled_w=round(im.width * latest_height / im.height)
        west.append((n,d,scaled_w))
    west.sort(key=lambda t:t[0])
    if not west:
        return {}

    latest_w=west[-1][2]
    pos={}
    for n,d,cum_w in west:
        # cumulative WEST_n begins at this x in the latest cumulative world
        pos[d.name]=latest_w-cum_w
        pos[f"west_{n:02d}"]=latest_w-cum_w

    # Center is the rightmost original starting map.
    first_job=west[0][1]
    center=find_center_source(first_job)
    if center:
        ci=Image.open(center)
        center_w=round(ci.width*latest_height/ci.height)
        pos["center"]=latest_w-center_w
    return pos


def build_seam_preserving_world(jobs_dir: Path, regions, out: Path):
    """
    v3.4 compositor.

    Rules:
    1. The entire top row is taken unchanged from the latest cumulative WEST final_native.png.
       It is NEVER overwritten by a vertical SOUTH patch.
    2. For a SOUTH region stitched with primary=NORTH, only the extension beginning
       at the north/south shared-overlap is composited.
    3. Inside that overlap a vertical alpha ramp preserves the original top row at
       the beginning of the overlap.
    4. Near left/right column edges the overlap is protected when a top-row neighbor
       exists there. This prevents the vertical patch from reintroducing a seam
       between WEST-04 and WEST-03.
    """
    latest=_find_latest_west_final(jobs_dir)
    if latest is None:
        make_final_world(regions,out)
        return

    _,latest_job,latest_final=latest
    top=Image.open(latest_final).convert("RGBA")
    top_h=top.height
    xpos=_top_x_positions_from_cumulative(jobs_dir,top_h)

    patches=[]
    max_h=top.height
    max_w=top.width

    for job_dir in jobs_dir.iterdir():
        if not job_dir.is_dir():
            continue

        gj=job_dir/"grid_job.json"
        fn=job_dir/"stitched"/"final_native.png"
        sm=job_dir/"stitched"/"stitch_meta.json"
        if not (gj.exists() and fn.exists() and sm.exists()):
            continue

        try:
            m=load_json(gj)
            stitch=load_json(sm)
            side=m.get("primary_side")
            target=m.get("target",{})
            x=int(target.get("x",0))
            y=int(target.get("y",0))

            if side!="north" or y<=0:
                continue

            primary=m.get("primary_neighbor",{})
            primary_id=primary.get("id","")
            px=xpos.get(primary_id)

            if px is None:
                top_region=regions.get((x,0))
                if top_region:
                    px=xpos.get(top_region["id"])
            if px is None:
                continue

            patch=Image.open(fn).convert("RGBA")

            actual_core=int(
                stitch.get("actual_core")
                or stitch.get("actual_new_core_height")
                or 0
            )
            actual_overlap=int(
                stitch.get("actual_overlap")
                or stitch.get("actual_overlap_height")
                or 0
            )
            if actual_core<=0 or actual_overlap<=0:
                continue

            # final_native = [NORTH SOURCE without its bottom overlap]
            #                + [MERGED OVERLAP]
            #                + [NEW SOUTH CORE]
            # Therefore source_h = final_h - actual_core.
            source_h=patch.height-actual_core
            if source_h<=actual_overlap:
                continue

            extension_start=source_h-actual_overlap
            extension=patch.crop((0,extension_start,patch.width,patch.height))

            # Align overlap to the bottom edge of the unchanged top row.
            py=top_h-actual_overlap

            west_neighbor_exists=(x-1,0) in regions
            east_neighbor_exists=(x+1,0) in regions

            patches.append({
                "x":px,
                "y":py,
                "image":extension,
                "overlap":actual_overlap,
                "west_guard":west_neighbor_exists,
                "east_guard":east_neighbor_exists,
                "name":job_dir.name,
            })

            max_w=max(max_w,px+extension.width)
            max_h=max(max_h,py+extension.height)

        except Exception:
            continue

    world=Image.new("RGBA",(max_w,max_h),(0,0,0,0))
    world.alpha_composite(top,(0,0))

    for item in patches:
        patch=item["image"].copy()
        overlap=min(item["overlap"],patch.height)
        pw,ph=patch.size

        # Full opacity below the shared overlap.
        mask=Image.new("L",(pw,ph),255)
        mpx=mask.load()

        # In overlap: top starts at 0 alpha and reaches 255 at its bottom.
        # This guarantees the already-good WEST top row remains exact.
        guard=max(12,min(48,pw//10))

        for yy in range(overlap):
            vy=yy/max(1,overlap-1)  # 0 -> 1
            for xx in range(pw):
                edge=1.0

                # Preserve column side edge where it touches another existing top column.
                if item["west_guard"] and xx<guard:
                    edge=min(edge,xx/max(1,guard-1))
                if item["east_guard"] and xx>=pw-guard:
                    edge=min(edge,(pw-1-xx)/max(1,guard-1))

                a=int(max(0,min(255,round(255*vy*edge))))
                mpx[xx,yy]=a

        patch.putalpha(mask)
        world.alpha_composite(patch,(item["x"],item["y"]))

    world.save(out)


def main():
    ap=argparse.ArgumentParser(description="Rebuild grid from existing jobs using cumulative final_native files.")
    ap.add_argument("--jobs",default="jobs")
    ap.add_argument("--force",action="store_true")
    args=ap.parse_args()

    jobs=Path(args.jobs).resolve()
    grid=ensure_dir(jobs/"_grid")
    regions={}

    # Find all existing WEST jobs regardless of the starting number.
    # Example: if the user began with west_02, then west_02 is physically the
    # first region left of CENTER, west_03 the second, etc.
    west_jobs={}
    for d in jobs.iterdir():
        if not d.is_dir():
            continue
        n=west_num(d.name)
        if n is not None:
            # Prefer non-h-suffixed folder when duplicates exist.
            current=west_jobs.get(n)
            if current is None or ("_h" in current.name.lower() and "_h" not in d.name.lower()):
                west_jobs[n]=d

    if not west_jobs:
        raise RuntimeError("No WEST jobs found (expected folders like west_02, west_03, ...).")

    west_numbers=sorted(west_jobs)
    first_num=west_numbers[0]
    first_west=west_jobs[first_num]

    # CENTER = original starting map stored as the source of the FIRST west expansion,
    # regardless of whether that first job is named west_01, west_02, west_05, etc.
    center_src=find_center_source(first_west)
    if not center_src:
        raise RuntimeError(
            f"Starting CENTER map not found in {first_west.name}/input/"
            "source_region_native.png or source_region.png"
        )

    center_dir=ensure_dir(jobs/"center")
    center_out=center_dir/"region.png"
    if args.force or not center_out.exists():
        shutil.copy2(center_src,center_out)
    save_json(center_dir/"region.json",{
        "version":3,"id":"center","x":0,"y":0,"image":"region.png",
        "source":f"original starting map from {first_west.name}"
    })
    regions[(0,0)]={"id":"center","image":center_out.resolve(),"job":center_dir.resolve()}

    # Physical grid positions are based on expansion ORDER, not the literal numeric suffix.
    # If the chain starts at west_02:
    #   west_02 -> x=-1
    #   west_03 -> x=-2
    #   west_04 -> x=-3
    previous_world=center_src
    for order,n in enumerate(west_numbers, start=1):
        job=west_jobs[n]
        final=find_final(job)
        if not final:
            print(f"[SKIP] {job.name}: final_native.png missing")
            continue

        cache=ensure_dir(job/".region_cache")
        region=cache/"region.png"
        if args.force or not region.exists():
            extract_west_from_cumulative(final, previous_world, region)

        x=-order
        rid=job.name
        save_json(job/"region.json",{
            "version":3,
            "id":rid,
            "x":x,
            "y":0,
            "image":str(region.relative_to(job)),
            "source":"difference of cumulative final_native images",
            "legacy_sequence_number":n,
            "physical_order_from_center":order
        })
        regions[(x,0)]={"id":rid,"image":region.resolve(),"job":job.resolve()}
        previous_world=final
        print(f"[OK] {rid:20s} -> ({x},0) {region} {Image.open(region).size}")

    # Keep already generated lower/new grid regions.
    regions.update(discover_new_grid_regions(jobs))

    save_json(grid/"grid.json",{
        "version":2,
        "center_definition":"original starting map",
        "regions":[
            {"id":r["id"],"x":x,"y":y,"image":str(r["image"]),"job":str(r["job"])}
            for (x,y),r in sorted(regions.items(), key=lambda kv:(kv[0][1],kv[0][0]))
        ]
    })
    build_seam_preserving_world(jobs,regions,grid/"world_final.png")
    wf=Image.open(grid/"world_final.png").convert("RGBA")
    pv=ImageOps.contain(wf,(1500,1000),Image.LANCZOS)
    preview_bg=Image.new("RGBA",pv.size,(18,17,15,255))
    preview_bg.alpha_composite(pv,(0,0))
    preview_bg.convert("RGB").save(grid/"grid_preview.png")

    print(f"Grid: {grid/'grid.json'}")
    print(f"Preview: {grid/'grid_preview.png'}")
    print(f"Final world: {grid/'world_final.png'}")

if __name__=="__main__":
    main()
