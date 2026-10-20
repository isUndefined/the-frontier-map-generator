"""Native world-space context; all rectangles are half-open [x0,y0,x1,y1]."""
import cv2
import numpy as np
from PIL import Image, ImageDraw
from tile_grid import discover, ordered_regions, open_tile, save_json


def build_generation_zones(jobs, x, y, layout, transition=192):
    if transition < 0:
        raise ValueError('Transition must be nonnegative')
    w, h, ov = (int(layout[k]) for k in ('width', 'height', 'overlap'))
    # Bounded local halo, never an accumulated panorama. Keep native pixels.
    margin = transition
    cw, ch = w + 2 * margin, h + 2 * margin
    px, py = x * (w - ov), y * (h - ov)
    ox, oy = px - margin, py - margin
    canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
    known = np.zeros((ch, cw), dtype=bool)
    intersections = []
    manifest = {}
    names = {(-1,-1): 'northwest', (0,-1): 'north', (1,-1): 'northeast',
             (-1,0): 'west', (1,0): 'east',
             (-1,1): 'southwest', (0,1): 'south', (1,1): 'southeast'}
    for source in ordered_regions(discover(jobs)):
        sx, sy = int(source['x']), int(source['y'])
        if (sx, sy) == (x, y):
            continue
        wx, wy = sx * (w - ov), sy * (h - ov)
        a, b = max(ox, wx), max(oy, wy)
        c, d = min(ox + cw, wx + w), min(oy + ch, wy + h)
        if a >= c or b >= d:
            continue
        crop = [a-wx, b-wy, c-wx, d-wy]
        rect = [a-ox, b-oy, c-ox, d-oy]
        data = np.asarray(open_tile(source['image'], layout))
        canvas[rect[1]:rect[3], rect[0]:rect[2]] = data[crop[1]:crop[3], crop[0]:crop[2]]
        known[rect[1]:rect[3], rect[0]:rect[2]] = True
        ta, tb, tc, td = max(px, wx), max(py, wy), min(px+w, wx+w), min(py+h, wy+h)
        item = {'id': source['id'], 'grid': [sx, sy], 'source_crop': crop,
                'canvas_rect': rect, 'world_rect': [a,b,c,d],
                'lock_rect': [ta-ox,tb-oy,tc-ox,td-oy] if ta < tc and tb < td else None,
                'overlap_source_crop': [ta-wx,tb-wy,tc-wx,td-wy] if ta < tc and tb < td else None}
        intersections.append(item)
        side = names.get((sx-x, sy-y))
        if side:
            manifest[side] = {'id': source['id'], 'grid': [sx,sy]}
    inside = np.zeros((ch, cw), dtype=bool)
    inside[margin:margin+h, margin:margin+w] = True
    editable = inside & ~known
    # Chebyshev distance gives exact 192 px rectangular bands around real locks,
    # including diagonal-only constraints. No synthetic neighbor geography.
    distance = cv2.distanceTransform((~known).astype(np.uint8), cv2.DIST_C, 3)
    transition_pixels = editable & (distance <= transition)
    labels = np.zeros((ch, cw), dtype=np.uint8)  # locked surrounding world/padding
    labels[editable] = 3  # free
    labels[transition_pixels] = 2
    labels[known] = 1  # exact known world
    mask = np.where(editable, 255, 0).astype(np.uint8)
    zones = {'version': 2, 'rect_format': 'xyxy_half_open',
             'coordinate_system': 'world_pixels_y_down', 'target': {'x':x,'y':y},
             'tile_width':w, 'tile_height':h, 'overlap':ov, 'transition':transition,
             'canvas_size':[cw,ch], 'canvas_world_origin':[ox,oy],
             'tile_world_rect':[px,py,px+w,py+h],
             'tile_crop':[margin,margin,margin+w,margin+h], 'neighbors':manifest,
             'paint_order':[i['id'] for i in intersections], 'sources':intersections,
             'zone_map':'generation_zones.png',
             'zone_values':{'locked_padding':0,'lock':1,'transition':2,'free':3},
             'pixel_counts':{name:int(np.count_nonzero(labels == value))
                             for name,value in [('locked_padding',0),('lock',1),('transition',2),('free',3)]}}
    return canvas, mask, labels, zones


def save_generation_artifacts(inp, jobs, x, y, layout, transition=192):
    canvas, mask, labels, zones = build_generation_zones(jobs, x, y, layout, transition)
    Image.fromarray(canvas).save(inp / 'generation_canvas.png')
    Image.fromarray(mask).save(inp / 'generation_mask.png')
    Image.fromarray(labels).save(inp / 'generation_zones.png')
    save_json(inp / 'zones.json', zones)
    save_json(inp / 'neighbor_map.json', zones['neighbors'])
    colors = np.array([(48,48,48),(235,70,70),(245,195,45),(60,205,100)], dtype=np.float32)
    overlay = (canvas.astype(np.float32) * .45 + colors[labels] * .55).astype(np.uint8)
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    a,b,c,d = zones['tile_crop']
    draw.rectangle((a,b,c-1,d-1), outline='white', width=2)
    draw.text((8,8), 'RED lock / YELLOW transition / GREEN free / GRAY locked padding', fill='white')
    image.save(inp / 'prepare_debug_overlay.png')
    return zones
