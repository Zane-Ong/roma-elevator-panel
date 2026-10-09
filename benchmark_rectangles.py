import json
import statistics
import time
from pathlib import Path

from runtime_config import ROOT
import cv2
import numpy as np
import torch
from PIL import Image
from rectangle_engine import RectangleMatcher
from rectangle_geometry import map_rectangle

reference=Image.open(ROOT/'rectangle_panel/reference.png').convert('RGB')
w,h=reference.size
corners=np.float32([[0,0],[w,0],[w,h],[0,h]])
target=np.float32([[.06*w,.04*h],[.96*w,.08*h],[.93*w,.96*h],[.1*w,.9*h]])
truth=cv2.getPerspectiveTransform(corners,target)
current=Image.fromarray(cv2.warpPerspective(np.array(reference),truth,(w,h)))
engine=RectangleMatcher(reference)
start=time.perf_counter();engine.load('indoor');load=time.perf_counter()-start
records=[]
matrices=[]
for mode in ('official','cached'):
    times=[]
    for iteration in range(3):
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            if mode=='official':
                engine.cache_enabled=False
                warp,certainty=engine.model.match(reference,current,device='cuda')
            else:
                warp,certainty,_=engine.match(current,'indoor',True)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            matches=warp[torch.isfinite(warp).all(dim=-1)&(certainty>=.5)]
            ids=torch.linspace(0,len(matches)-1,min(5000,len(matches)),device='cuda').long()
            a,b=engine.model.to_pixel_coordinates(matches[ids],h,w,h,w)
            a,b=a.cpu().numpy(),b.cpu().numpy()
        matrix,mask=cv2.findHomography(a,b,cv2.USAC_MAGSAC,3,maxIters=10000,confidence=.999)
        assert matrix is not None
        points=np.float32([[400,1008],[537,1115],[674,1221],[537,1429]])
        actual=cv2.perspectiveTransform(points[:,None],matrix)[:,0]
        expected=cv2.perspectiveTransform(points[:,None],truth)[:,0]
        errors=np.linalg.norm(actual-expected,axis=1)
        assert errors.max()<2, errors
        print(mode,iteration,round(elapsed,3),'max_error',round(float(errors.max()),3),flush=True)
        times.append(elapsed)
        matrices.append(matrix)
        del warp,certainty,matches
    records.append({'mode':mode,'seconds':times,'warm_median_seconds':statistics.median(times[1:]),
                    'max_transfer_error_px':float(errors.max())})
baseline,optimized=records
report={'load_seconds':load,'gpu':torch.cuda.get_device_name(),'resolution':[560,864],
        'test':'fixed user reference with known synthetic perspective; unchanged full model and precision',
        'results':records,'warm_speedup':baseline['warm_median_seconds']/optimized['warm_median_seconds'],
        'cache_tensors_MB':sum(v.numel()*v.element_size() for f in engine.features.values() for v in f.values())/1024**2}
(ROOT/'rectangle_panel/performance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
