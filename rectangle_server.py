import io
import json
import threading
import time
import uuid
import zipfile
from pathlib import Path

from runtime_config import ROOT
from match_settings import SETTINGS
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont,ImageOps
from fastapi import FastAPI,UploadFile,Form,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from rectangle_engine import RectangleMatcher
from rectangle_geometry import validate_buttons,map_rectangle
from matching_visuals import matching_visuals

PANEL=ROOT/'rectangle_panel'
RESULTS=PANEL/'results'
RESULTS.mkdir(exist_ok=True)
REFERENCE=Image.open(PANEL/'reference.png').convert('RGB')
SEED=json.loads((PANEL/'template.json').read_text(encoding='utf-8'))
ENGINE=RectangleMatcher(REFERENCE)
TEMPLATE_LOCK=threading.Lock()
app=FastAPI(title='RoMa 矩形标定与映射')
app.mount('/results',StaticFiles(directory=RESULTS),name='results')


@app.get('/')
def page():
    return FileResponse(PANEL/'index.html')


@app.get('/reference.png')
def reference():
    return FileResponse(PANEL/'reference.png')


@app.get('/api/template')
def template():
    with TEMPLATE_LOCK:
        return json.loads((PANEL/'template.json').read_text(encoding='utf-8'))


@app.post('/api/template')
def save_template(payload:dict):
    try:
        buttons=validate_buttons(payload['buttons'],SEED['buttons'],REFERENCE.size)
        if payload.get('sha256') != SEED['sha256']:
            raise ValueError('基准照片不一致，请刷新页面。')
    except (ValueError,KeyError,TypeError) as error:
        raise HTTPException(400,str(error)) from error
    with TEMPLATE_LOCK:
        current=template_unlocked()
        backup=PANEL/'template_history'
        backup.mkdir(exist_ok=True)
        (backup/(uuid.uuid4().hex+'.json')).write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
        current.update(buttons=buttons,calibrated=True)
        temp=PANEL/'template.tmp'
        temp.write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
        temp.replace(PANEL/'template.json')
    return current


def template_unlocked():
    return json.loads((PANEL/'template.json').read_text(encoding='utf-8'))


@app.get('/api/status')
def status():
    return {'state':ENGINE.state,'variant':ENGINE.variant,'device':ENGINE.device,
            'reference_cached':len(ENGINE.features),'load_seconds':round(ENGINE.load_seconds,2)}


@app.get('/api/settings')
def settings():
    return SETTINGS


def perform_mapping(current,calibration,variant,confidence,max_matches,ransac_px,cache):
    started=time.perf_counter()
    buttons=validate_buttons(calibration['buttons'],SEED['buttons'],REFERENCE.size)
    with ENGINE.lock:
        warp,certainty,timing=ENGINE.match(current,variant,cache)
        valid=torch.isfinite(warp).all(dim=-1)&(certainty>=confidence)
        matches=warp[valid]
        if len(matches)<8:
            raise ValueError('有效匹配不足，请重新拍摄或适度降低置信度。')
        if len(matches)>max_matches:
            ids=torch.linspace(0,len(matches)-1,max_matches,device=matches.device).long()
            matches=matches[ids]
        a,b=ENGINE.model.to_pixel_coordinates(matches,REFERENCE.height,REFERENCE.width,current.height,current.width)
        a,b=a.cpu().numpy(),b.cpu().numpy()
    fit_start=time.perf_counter()
    matrix,inliers=cv2.findHomography(a,b,cv2.USAC_MAGSAC,ransac_px,maxIters=10000,confidence=.999)
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        raise ValueError('无法估计有效面板映射。')
    inliers=inliers.ravel().astype(bool)
    if inliers.sum()<8 or inliers.mean()<.2:
        raise ValueError('几何一致性不足，本次不输出框。')
    projected=cv2.perspectiveTransform(a[inliers,None,:],matrix)[:,0]
    residual=np.linalg.norm(projected-b[inliers],axis=1)
    folder=RESULTS/uuid.uuid4().hex
    folder.mkdir()
    drawn=current.copy()
    draw=ImageDraw.Draw(drawn)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',max(16,current.width//60))
    output=[]
    for button in buttons:
        mapped,mask=map_rectangle(button['rect_xyxy'],matrix,current.size)
        mapped.update(name=button['name'],reference_rect_xyxy=button['rect_xyxy'])
        mask_name=button['name']+'_mask.png'
        Image.fromarray(mask*255).save(folder/mask_name,compress_level=1)
        mapped['mask_file']=mask_name
        mapped['mask_url']=f'/results/{folder.name}/{mask_name}'
        mapped['requires_visibility_check']=True
        output.append(mapped)
        polygon=[tuple(p) for p in mapped['polygon_xy']]
        draw.line(polygon+[polygon[0]],fill='#00d8ff',width=3)
        if mapped['bbox_xyxy']:
            draw.rectangle(mapped['bbox_xyxy'],outline='#ffad32',width=2)
            draw.text(polygon[0],button['name'],font=font,fill='#ff3030')
    fit_seconds=time.perf_counter()-fit_start
    drawn.thumbnail((1400,1400))
    drawn.save(folder/'preview.png',compress_level=1)
    current.save(folder/'current.png',compress_level=1)
    connections,overlay=matching_visuals(REFERENCE,current,a,b,inliers,matrix)
    connections.save(folder/'matches.png',compress_level=1)
    overlay.save(folder/'overlay.png',compress_level=1)
    report={'schema_version':1,'image_width':current.width,'image_height':current.height,
            'reference_sha256':calibration['sha256'],'variant':variant,'homography':matrix.tolist(),
            'bbox_convention':'x1,y1 inclusive; x2,y2 exclusive; depth[y1:y2,x1:x2]',
            'mask_convention':'0 excluded; 255 included; original current-image resolution',
            'buttons':output,'matches':len(a),'inliers':int(inliers.sum()),
            'median_fit_residual_px':float(np.median(residual)),
            'parameters':{'confidence':confidence,'max_matches':max_matches,'ransac_px':ransac_px,
                          'coarse_resolution':560,'fine_resolution':864},
            'timing':{**timing,'geometry_and_masks_seconds':fit_seconds},
            'note':'图像面积不是物理面积；框内不代表可见；深度须与本实时原图对齐且过滤无效值。'}
    report['timing']['total_seconds']=time.perf_counter()-started
    (folder/'mapping.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    archive=folder/'results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_STORED) as bundle:
        for path in folder.iterdir():
            if path!=archive: bundle.write(path,path.name)
    return {**report,'preview_url':f'/results/{folder.name}/preview.png',
            'matches_url':f'/results/{folder.name}/matches.png',
            'overlay_url':f'/results/{folder.name}/overlay.png',
            'download_url':f'/results/{folder.name}/results.zip','json_url':f'/results/{folder.name}/mapping.json'}


@app.post('/api/match')
def match(image:UploadFile,variant:str=Form(SETTINGS['variant']),confidence:float=Form(SETTINGS['confidence']),
          max_matches:int=Form(SETTINGS['max_matches']),ransac_px:float=Form(SETTINGS['ransac_px']),cache:bool=Form(SETTINGS['cache'])):
    if variant not in ('indoor','outdoor') or not (.1<=confidence<=.95 and 500<=max_matches<=10000 and .5<=ransac_px<=10):
        raise HTTPException(400,'参数超出允许范围')
    try:
        current=ImageOps.exif_transpose(Image.open(image.file)).convert('RGB')
        calibration=template()
        return perform_mapping(current,calibration,variant,confidence,max_matches,ransac_px,cache)
    except ValueError as error:
        raise HTTPException(400,str(error)) from error
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        raise HTTPException(503,'显存不足，请关闭旧版匹配进程或其他 GPU 程序后重试。') from error


if __name__=='__main__':
    import httpx
    import webbrowser
    import uvicorn
    try:
        existing=httpx.get('http://127.0.0.1:7861/api/status',timeout=2,trust_env=False)
        if existing.status_code==200 and 'reference_cached' in existing.json():
            webbrowser.open('http://127.0.0.1:7861')
            raise SystemExit(0)
    except httpx.RequestError:
        pass
    threading.Thread(target=ENGINE.preload,daemon=True).start()
    threading.Timer(2,lambda:webbrowser.open('http://127.0.0.1:7861')).start()
    uvicorn.run(app,host='127.0.0.1',port=7861)
