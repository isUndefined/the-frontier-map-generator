"""Self-service native generation checks and retry handoffs, without transforms."""
from pathlib import Path
import numpy as np
from PIL import Image
from tile_grid import load_json, save_json


def save_diagnostics(job, report, generated=None, reference=None, fixed=None):
    job=Path(job)
    out=job/'draft'
    out.mkdir(exist_ok=True)
    report=dict(report)
    report['report']=str(out/'generation_check.json')
    report['instructions']=str(out/'generation_check.txt')
    report['retry_prompt']=str(out/'retry_prompt.txt') if report['status']=='rejected' else None
    save_json(out/'generation_check.json',report)
    alignment=report.get('alignment',{})
    rows=[f"Статус: {report['status']}",report.get('message',''),
          f"Проход: {report.get('pass','?')}/{report.get('pass_count','?')}"]
    for key in ('anchor_mae_before','matches','inliers','inlier_fraction',
                'landmark_displacement_p90_px','max_fitted_drift_px'):
        if key in alignment:rows.append(f'{key}: {alignment[key]}')
    if alignment.get('failed_checks'):
        rows.append('Не пройдены проверки: '+', '.join(alignment['failed_checks']))
    rows.extend(['Порог проверки геометрии: 2 px. Проверка ничего не сдвигает и не масштабирует.',
                 report.get('next_step','')])
    (out/'generation_check.txt').write_text('\n'.join(rows)+'\n')
    retry=out/'retry_prompt.txt'
    if report['status']=='rejected':
        original=(job/'input/prompt.txt').read_text()
        # A failed old request must not reproduce the old conflicting text.
        # Derive retry instructions from the actual current image/mask, while
        # leaving all input files and accepted progress untouched.
        try:
            from handoff_prompt import render_prompt
            with Image.open(job/'input/generation_mask.png') as im:
                retry_mask=np.array(im.convert('L'))
            with Image.open(job/'input/generation_zones.png') as im:
                retry_labels=np.array(im.convert('L'))
            original=render_prompt(load_json(job/'grid_job.json'),
                                   load_json(job/'input/zones.json'),retry_mask,retry_labels)
        except (ValueError,FileNotFoundError):
            # Invalid handoffs already have a distinct rejection and cannot
            # be advanced; keep their original text for diagnosis only.
            pass
        retry.write_text(original+'\nRETRY OF THIS SAME PASS; THE EDIT AREA HAS NOT CHANGED.\n'+
            'Use the original currently attached generation_canvas.png and generation_mask.png again.\n'+
            'Do not use the previous failed generated.png as a reference.\n'+
            'The prior attempt failed: '+report.get('message','Generation checks rejected the result')+'\n'+
            'Keep every existing feature in the locked rectangles at the exact original position and scale.\n'+
            'Do not crop or expand the frame, remove padding, resize, shift or recompose the existing map.\n')
    else:
        retry.unlink(missing_ok=True)
    if generated is not None and reference is not None and generated.shape==reference.shape:
        difference=np.abs(generated.astype(np.float32)-reference.astype(np.float32)).mean(axis=2)
        heat=np.zeros_like(reference)
        heat[...,0]=np.clip(difference*4,0,255).astype(np.uint8)
        heat[~fixed]=0
        Image.fromarray(heat).save(out/'locked_context_difference.png')
        restored=generated.copy()
        with Image.open(job/'input/generation_mask.png') as im:
            preserve=np.asarray(im.convert('L'))==0
        restored[preserve]=reference[preserve]
        Image.fromarray(restored).save(out/'locked_context_preview.png')
    return report


def anchor_failure(job,generated,reference,fixed,alignment):
    meta=load_json(job/'grid_job.json')
    zone=meta.get('generation_patch',{})
    return save_diagnostics(job,{
        'format':'frontier_generation_check_v1','status':'rejected','issues':['weak_alignment'],
        'pass':zone.get('pass'),'pass_count':zone.get('pass_count'),
        'alignment':alignment,'message':alignment['reason'],
        'next_step':'Повторите тот же проход: приложите исходные generation_canvas.png, generation_mask.png и текст draft/retry_prompt.txt. Не меняйте размер и не запускайте rebuild-job.',
    },generated,reference,fixed)


def check_job(jobs,job,generated=None,strict=False):
    from generation_patches import apply_patch_result
    from stitch_region_grid import prepare_alignment
    from seam_diagnostics import analyze_seam
    from terrain_quality import audit_tile
    jobs=Path(jobs).resolve()
    folder=jobs/job
    if folder.parent!=jobs or not (folder/'grid_job.json').is_file():
        raise ValueError('No prepared job')
    meta=load_json(folder/'grid_job.json')
    if meta.get('prepare_version')!=3:
        raise ValueError('check-job requires a native-patch job')
    zones=load_json(folder/'input/zones.json')
    image=Path(generated).resolve() if generated else folder/'generated.png'
    expected=tuple(meta['expected_generation_size'])
    with Image.open(image) as im:
        actual=im.size
        result=np.array(im.convert('RGB'))
    base={'format':'frontier_generation_check_v1','job':job,
          'pass':zones['pass'],'pass_count':zones['pass_count'],
          'expected_size':list(expected),'actual_size':list(actual),'strict':bool(strict)}
    if actual!=expected:
        return save_diagnostics(folder,{**base,'status':'rejected','issues':['wrong_dimensions'],
            'message':f'Generated image is {actual}, expected {expected}. No resize or upscaling permitted.',
            'next_step':'Сохраните исходный PNG результата, не скриншот. Если размер неверен, повторите этот проход с draft/retry_prompt.txt; не увеличивайте изображение вручную.'})
    _,alignment=prepare_alignment(folder,result,expected,actual,meta,draft=True)
    issues=[]
    if alignment['status']!='accepted':issues.append('weak_alignment')
    try:
        target,remaining,state=apply_patch_result(folder,result)
    except ValueError as exc:
        return save_diagnostics(folder,{**base,'status':'rejected','alignment':alignment,
            'issues':['invalid_handoff'],'message':str(exc),
            'next_step':'Проверьте, что canvas, mask и prompt относятся к одному текущему проходу. Не расширяйте маску вручную.'})
    w,h=meta['target_region_size']
    left,top,right,bottom=state['passes'][state['current']]['tile_edit_rect']
    side=meta['primary_side']
    band={'east':w-right,'west':left,'north':top,'south':h-bottom}[side]
    seam=analyze_seam(target,side,band)
    if strict and seam['status']=='rejected':issues.append('seam_mismatch')
    complete=not bool(np.any(remaining))
    out=folder/'draft'
    out.mkdir(exist_ok=True)
    candidate=out/'pass_preview.png'
    Image.fromarray(target).save(candidate)
    if complete and not issues:
        quality=audit_tile(jobs,job,candidate,stitch_metadata={'alignment':alignment})
        issues.extend(quality['issues'])
    if issues:
        if 'weak_alignment' in issues:
            message=alignment['reason']
        elif 'seam_mismatch' in issues:
            message='The new terrain has a strong discontinuity at the join'
        else:message='Generation QA rejected: '+', '.join(issues)
        next_step='Повторите тот же проход: исходные generation_canvas.png, generation_mask.png и draft/retry_prompt.txt. Прогресс и опубликованный мир сохранены.'
    elif complete:
        message='Current pass is accepted; the tile is complete'
        next_step='Откройте preview. Затем finalize --draft --strict; после проверки finalize --replace --strict для опубликованного тайла.'
    else:
        message='Current pass is accepted; generate the next handoff after advance-job'
        next_step=f'Выполните advance-job --job {job} --strict. Затем отправьте новые canvas, mask и prompt для следующего прохода.'
    with Image.open(folder/'input/generation_canvas.png') as im:reference=np.array(im.convert('RGB'))
    with Image.open(folder/'input/generation_zones.png') as im:fixed=np.array(im)==1
    return save_diagnostics(folder,{**base,'status':'rejected' if issues else 'passed',
        'issues':issues,'alignment':alignment,'seam_quality':seam,'tile_complete':complete,
        'preview':str(candidate),'message':message,'next_step':next_step},result,reference,fixed)
