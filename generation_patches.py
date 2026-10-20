"""Native local outpainting passes. No resampling, registration or pixel shifts."""
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tile_grid import discover, ordered_regions, open_tile, load_json, save_json


def plan_passes(layout, side, context=700, transition=192):
    w,h,ov = (layout[k] for k in ('width','height','overlap'))
    horizontal = side in ('east','west')
    axis,cross = (w,h) if horizontal else (h,w)
    along,across = 1536,1024
    # A complete original overlap AND broad neighbor context in the first pass.
    depth = max(transition,min(context,along-ov-1))
    step = along-ov-depth
    if step < 1:
        raise ValueError('Overlap/context leaves no editable pixels in a pass')
    stride = max(1,across-transition)
    starts = list(range(0,max(1,cross-across+1),stride))
    if cross > across and starts[-1] != cross-across:
        starts.append(cross-across)
    intervals=[]
    if side in ('east','south'):
        end=axis-ov
        while end>0:
            start=max(0,end-step)
            intervals.append((start,end,start,start+along))
            end=start
    else:
        start=ov
        while start<axis:
            end=min(axis,start+step)
            intervals.append((start,end,end-along,end))
            start=end
    result=[]
    for start,end,lo,hi in intervals:
        for c in starts:
            if cross <= across:
                clo=-(across-cross)//2
            else:
                clo=c
            chi=clo+across
            if horizontal:
                edit=[start,max(0,clo),end,min(cross,chi)]
                rect=[lo,clo,hi,chi]
            else:
                edit=[max(0,clo),start,min(cross,chi),end]
                rect=[clo,lo,chi,hi]
            result.append({'tile_edit_rect':edit,'tile_canvas_rect':rect})
    return result


def initialize(job, meta):
    inp=job/'input'
    with Image.open(inp/'neighbor_lock.png') as im: im.save(inp/'accepted_tile.png')
    with Image.open(inp/'neighbor_mask.png') as im: im.save(inp/'remaining_mask.png')
    state={'version':1,'current':0,'accepted_passes':[],
           'passes':plan_passes(meta['layout'],meta['primary_side'],meta.get('requested_context',700))}
    save_json(inp/'patch_state.json',state)
    write_handoff(job,meta,state)


def compose_patch(job, meta, state):
    jobs=Path(meta['jobs_root'])
    # Portable after moving the jobs directory or running inside Docker.
    if (job.parent/'layout.json').is_file(): jobs=job.parent
    w,h,ov=(meta['layout'][k] for k in ('width','height','overlap'))
    tx,ty=meta['target']['x']*(w-ov),meta['target']['y']*(h-ov)
    spec=state['passes'][state['current']]
    a,b,c,d=spec['tile_canvas_rect']
    ox,oy=tx+a,ty+b
    canvas=np.zeros((d-b,c-a,3),np.uint8)
    known=np.zeros(canvas.shape[:2],bool)
    sources=[]
    for source in ordered_regions(discover(jobs)):
        sx,sy=int(source['x']),int(source['y'])
        if (sx,sy)==(meta['target']['x'],meta['target']['y']):continue
        wx,wy=sx*(w-ov),sy*(h-ov)
        x0,y0,x1,y1=max(ox,wx),max(oy,wy),min(tx+c,wx+w),min(ty+d,wy+h)
        if x0>=x1 or y0>=y1:continue
        image=np.asarray(open_tile(source['image'],meta['layout']))
        canvas[y0-oy:y1-oy,x0-ox:x1-ox]=image[y0-wy:y1-wy,x0-wx:x1-wx]
        known[y0-oy:y1-oy,x0-ox:x1-ox]=True
        sources.append({'id':source['id'],'grid':[sx,sy],
                        'source_crop':[x0-wx,y0-wy,x1-wx,y1-wy],
                        'canvas_rect':[x0-ox,y0-oy,x1-ox,y1-oy],
                        'world_rect':[x0,y0,x1,y1]})
    target=np.asarray(Image.open(job/'input/accepted_tile.png').convert('RGB'))
    remaining=np.asarray(Image.open(job/'input/remaining_mask.png').convert('L'))
    x0,y0,x1,y1=max(0,a),max(0,b),min(w,c),min(h,d)
    fixed=remaining[y0:y1,x0:x1]==0
    patch=canvas[y0-b:y1-b,x0-a:x1-a]
    patch[fixed]=target[y0:y1,x0:x1][fixed]
    known[y0-b:y1-b,x0-a:x1-a]|=fixed
    mask=np.zeros(canvas.shape[:2],np.uint8)
    l,t,r,s=spec['tile_edit_rect']
    mask[t-b:s-b,l-a:r-a]=remaining[t:s,l:r]
    labels=np.zeros(mask.shape,np.uint8)
    labels[known]=1
    labels[mask!=0]=3
    distance=cv2.distanceTransform((~known).astype(np.uint8),cv2.DIST_C,3)
    labels[(mask!=0)&(distance<=meta.get('transition',192))]=2
    zones={'version':3,'rect_format':'xyxy_half_open','mode':'native_local_outpaint',
           'pass':state['current']+1,'pass_count':len(state['passes']),
           'canvas_size':[c-a,d-b],'canvas_world_origin':[ox,oy],
           'target':meta['target'],'tile_world_rect':[tx,ty,tx+w,ty+h],
           'tile_canvas_rect':[a,b,c,d],'tile_edit_rect':spec['tile_edit_rect'],
           'overlap':ov,'transition':meta.get('transition',192),
           'reference_context_px':meta.get('requested_context',700),
           'sources':sources,'accepted_passes':state['accepted_passes'],
           'neighbors':{s:{'id':r['id'],'grid':[r['x'],r['y']]}
                        for s,r in {**meta['neighbors'],**meta['diagonal_neighbors']}.items()},
           'zone_values':{'locked_padding':0,'locked_context':1,'transition':2,'free':3},
           'pixel_counts':{'known':int(known.sum()),'editable':int(np.count_nonzero(mask)),
                           'transition':int(np.count_nonzero(labels==2)),
                           'free':int(np.count_nonzero(labels==3))}}
    return canvas,mask,labels,zones


def write_handoff(job,meta,state):
    inp=job/'input'
    canvas,mask,labels,zones=compose_patch(job,meta,state)
    if not np.any(mask):raise ValueError('Planned pass has no new pixels')
    for name,data in [('generation_canvas.png',canvas),('generation_mask.png',mask),('generation_zones.png',labels)]:
        Image.fromarray(data).save(inp/name)
    save_json(inp/'zones.json',zones)
    overlay=(canvas.astype(float)*.55+np.array([(35,35,35),(210,60,60),(230,190,45),(55,190,90)])[labels]*.45).astype('uint8')
    Image.fromarray(overlay).save(inp/'prepare_debug_overlay.png')
    cw,ch=zones['canvas_size']
    lines=[
        'EDIT the FIRST attached image. The SECOND image is a mask reference, never output the mask.',
        f'Output EXACTLY {cw}x{ch} PNG. Keep the identical canvas frame, native pixel scale and camera.',
        f'Local outpaint pass {zones["pass"]}/{zones["pass_count"]} for The Frontier tile {meta["target"]}.',
        'Fill ONLY WHITE mask pixels. Preserve all BLACK mask pixels, including plain black padding.',
        f'The first {meta.get("transition",192)}px of white territory next to existing terrain is a transition zone; the rest is free new territory.',
        'This is a small continuation of the same landscape, not a new independent scenic composition.',
        'Start at the existing terrain edge. Continue the LOCAL slope, shore, valley, water and vegetation at every contact point.',
        'Continue a lake as the SAME lake with a coherent shoreline; continue a valley at the SAME elevation.',
        'Do not replace water or a valley with an abrupt wall of mountains, cliffs, forest or a straight ridge.',
        'Never turn the rectangular edit boundary into a geological barrier or a visible separator line.',
        'Do not repeat unique landmarks, lakes, castles or waterfalls from the reference into new territory.',
        'Match the visible terrain density, lighting, perspective and high oblique aerial painterly style.',
        'Keep the map continuous at both diagonal corner contacts.',
        'No text, labels, border, interface or divider line. Return the FULL canvas, without cropping or resizing.',
        f'Canvas world origin: {zones["canvas_world_origin"]}. All rectangles use XYXY half-open pixels.',
        f'New target-territory rectangle: {zones["tile_edit_rect"]}; canvas relative to tile: {zones["tile_canvas_rect"]}.',
        'Only canvas, mask and this prompt are needed. The pipeline restores exact existing world pixels.',
    ]
    (inp/'prompt.txt').write_text('\n'.join(lines)+'\n')
    meta.update(generation_tile_crop=None, expected_generation_size=zones['canvas_size'],generation_world_origin=zones['canvas_world_origin'],
                generation_patch=zones,workflow='chatgpt_native_patches_v1')
    save_json(job/'grid_job.json',meta)


def apply_patch_result(job,generated,require_complete=False):
    meta=load_json(job/'grid_job.json')
    state=load_json(job/'input/patch_state.json')
    # Rebuild the expected handoff from immutable sources/state; no output may
    # invent source geometry, restore missing anchors or move accepted pixels.
    _,expected_mask,_,_=compose_patch(job,meta,state)
    mask=np.asarray(Image.open(job/'input/generation_mask.png').convert('L'))
    if not np.array_equal(mask,expected_mask):raise ValueError('Pass mask differs from planned editable terrain')
    target=np.array(Image.open(job/'input/accepted_tile.png').convert('RGB'))
    remaining=np.array(Image.open(job/'input/remaining_mask.png').convert('L'))
    a,b,c,d=state['passes'][state['current']]['tile_canvas_rect']
    w,h=meta['target_region_size']
    if generated.shape[:2]!=(d-b,c-a):raise ValueError('Native pass dimensions differ; no resizing permitted')
    x0,y0,x1,y1=max(0,a),max(0,b),min(w,c),min(h,d)
    editable=mask[y0-b:y1-b,x0-a:x1-a]!=0
    target[y0:y1,x0:x1][editable]=generated[y0-b:y1-b,x0-a:x1-a][editable]
    remaining[y0:y1,x0:x1][editable]=0
    if require_complete and np.any(remaining):
        raise ValueError('Tile is incomplete. Run advance-job with this PNG, then generate the next canvas/mask/prompt before finalize')
    return target,remaining,state
