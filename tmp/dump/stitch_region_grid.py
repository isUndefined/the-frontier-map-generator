#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image


def load_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save_json(p,d): Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def cost(a,b): return np.abs(a.astype(np.float32)-b.astype(np.float32)).sum(axis=2)
def min_v(c):
    h,w=c.shape; dp=c.copy(); par=np.zeros((h,w),np.int32); big=np.float32(1e20); xs=np.arange(w)
    for y in range(1,h):
        prev=dp[y-1]; l=np.empty_like(prev); r=np.empty_like(prev); l[0]=big; l[1:]=prev[:-1]; r[-1]=big; r[:-1]=prev[1:]; st=np.stack([l,prev,r]); idx=np.argmin(st,axis=0); par[y]=np.clip(xs+idx-1,0,w-1); dp[y]=c[y]+st[idx,xs]
    s=np.zeros(h,np.int32); s[-1]=int(np.argmin(dp[-1]));
    for y in range(h-1,0,-1): s[y-1]=par[y,s[y]]
    return s

def merge_lr(l,r,feather):
    s=min_v(cost(l,r)); h,w,_=l.shape; out=np.empty_like(l)
    for y,x in enumerate(s):
        x=int(x); a=max(0,x-feather); b=min(w,x+feather+1); out[y,:a]=l[y,:a]; out[y,b:]=r[y,b:]
        if b>a:
            al=np.linspace(0,1,b-a,dtype=np.float32)[:,None]; out[y,a:b]=np.clip(l[y,a:b]*(1-al)+r[y,a:b]*al,0,255).astype(np.uint8)
    return out,s

def merge_tb(t,b,feather):
    # transpose spatial axes -> reuse vertical seam
    mt,s=merge_lr(np.transpose(t,(1,0,2)),np.transpose(b,(1,0,2)),feather)
    return np.transpose(mt,(1,0,2)),s

def edge_score(a,b,side):
    A=np.array(Image.open(a).convert('RGB')); B=np.array(Image.open(b).convert('RGB'))
    if side in ('north','south'):
        w=min(A.shape[1],B.shape[1]); A=np.array(Image.fromarray(A).resize((w,round(A.shape[0]*w/A.shape[1])),Image.LANCZOS)); B=np.array(Image.fromarray(B).resize((w,round(B.shape[0]*w/B.shape[1])),Image.LANCZOS)); band=min(32,A.shape[0],B.shape[0]); x=A[:band] if side=='north' else A[-band:]; y=B[-band:] if side=='north' else B[:band]
    else:
        h=min(A.shape[0],B.shape[0]); A=np.array(Image.fromarray(A).resize((round(A.shape[1]*h/A.shape[0]),h),Image.LANCZOS)); B=np.array(Image.fromarray(B).resize((round(B.shape[1]*h/B.shape[0]),h),Image.LANCZOS)); band=min(32,A.shape[1],B.shape[1]); x=A[:,:band] if side=='west' else A[:,-band:]; y=B[:,-band:] if side=='west' else B[:,:band]
    return float(np.mean(np.abs(x.astype(np.float32)-y.astype(np.float32))))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--job',required=True); p.add_argument('--generated',required=True); p.add_argument('--feather',type=int,default=4); args=p.parse_args()
    job=Path(args.job); m=load_json(job/'grid_job.json'); gen=np.array(Image.open(args.generated).convert('RGB')); src_img=Image.open(job/m['files']['source_native']).convert('RGB'); side=m['primary_side']; core=m['core']; ov=m['overlap']; ctx=m['context']; ew,eh=m['expected_generation_size']; gw,gh=gen.shape[1],gen.shape[0]
    sx,sy=gw/ew,gh/eh
    if side in ('west','east'):
        corea=max(1,round(core*sx)); ova=max(8,round(ov*sx)); ctxa=gw-corea-ova; src=src_img.resize((max(1,round(src_img.width*gh/src_img.height)),gh),Image.LANCZOS); S=np.array(src)
        if side=='east':
            gcore=gen[:,:corea]; gov=gen[:,corea:corea+ova]; sov=S[:,:ova]; tail=S[:,ova:]; mov,seam=merge_lr(gov,sov,args.feather); final=np.concatenate([gcore,mov,tail],axis=1); region=np.concatenate([gcore,mov],axis=1)
        else:
            gov=gen[:,ctxa:ctxa+ova]; gcore=gen[:,ctxa+ova:]; head=S[:,:S.shape[1]-ova]; sov=S[:,S.shape[1]-ova:]; mov,seam=merge_lr(sov,gov,args.feather); final=np.concatenate([head,mov,gcore],axis=1); region=np.concatenate([mov,gcore],axis=1)
    else:
        corea=max(1,round(core*sy)); ova=max(8,round(ov*sy)); ctxa=gh-corea-ova; src=src_img.resize((gw,max(1,round(src_img.height*gw/src_img.width))),Image.LANCZOS); S=np.array(src)
        if side=='south':
            gcore=gen[:corea]; gov=gen[corea:corea+ova]; sov=S[:ova]; tail=S[ova:]; mov,seam=merge_tb(gov,sov,args.feather); final=np.concatenate([gcore,mov,tail],axis=0); region=np.concatenate([gcore,mov],axis=0)
        else:
            gov=gen[ctxa:ctxa+ova]; gcore=gen[ctxa+ova:]; head=S[:S.shape[0]-ova]; sov=S[S.shape[0]-ova:]; mov,seam=merge_tb(sov,gov,args.feather); final=np.concatenate([head,mov,gcore],axis=0); region=np.concatenate([mov,gcore],axis=0)
    out=job/'stitched'; out.mkdir(exist_ok=True); Image.fromarray(final).save(out/'final_native.png'); Image.fromarray(region).save(out/'region_with_primary_overlap.png')
    validation={}
    for s,r in m.get('secondary_neighbors',{}).items():
        try: validation[s]={'neighbor':r['id'],'mean_edge_difference':edge_score(out/'region_with_primary_overlap.png',r['image'],s)}
        except Exception as e: validation[s]={'neighbor':r['id'],'error':str(e)}
    meta={'generated_size':[gw,gh],'expected_size':[ew,eh],'scale_x':sx,'scale_y':sy,'primary_side':side,'actual_core':corea,'actual_overlap':ova,'actual_context':ctxa,'final_size':[final.shape[1],final.shape[0]],'region_size':[region.shape[1],region.shape[0]],'secondary_validation':validation}
    save_json(out/'stitch_meta.json',meta)

    target=m.get('target',{})
    region_meta={
        'version':1,
        'id':m.get('region_id',job.name),
        'x':int(target.get('x',0)),
        'y':int(target.get('y',0)),
        'image':'stitched/region_with_primary_overlap.png',
        'primary_side':side,
        'primary_neighbor':m.get('primary_neighbor',{}).get('id'),
        'secondary_neighbors':{s:r.get('id') for s,r in m.get('secondary_neighbors',{}).items()},
    }
    save_json(job/'region.json',region_meta)

    print(out/'final_native.png')
    print(out/'region_with_primary_overlap.png')
    print(job/'region.json')
    print(json.dumps(validation,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
