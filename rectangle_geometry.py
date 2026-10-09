import cv2
import numpy as np


def validate_buttons(buttons, template, size):
    if len(buttons) != len(template) or [b['name'] for b in buttons] != [b['name'] for b in template]:
        raise ValueError('必须保留固定的 11 个按钮名称与顺序。')
    result = []
    for button, source in zip(buttons, template):
        box = button.get('rect_xyxy')
        if box is None:
            raise ValueError(f"请先框选 {button['name']}")
        box = np.asarray(box, dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError('矩形必须包含四个有限数值。')
        x1,y1,x2,y2 = box
        if not (0 <= x1 < x2 <= size[0] and 0 <= y1 < y2 <= size[1]):
            raise ValueError('矩形必须位于原图范围内且宽高为正。')
        result.append({**source, 'rect_xyxy': box.tolist()})
    return result


def map_rectangle(box, matrix, size):
    x1,y1,x2,y2 = box
    corners = np.float64([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
    homogeneous = np.c_[corners, np.ones(4)] @ np.asarray(matrix).T
    denom = homogeneous[:,2]
    if not np.isfinite(homogeneous).all() or np.any(abs(denom)<1e-8) or not (np.all(denom>0) or np.all(denom<0)):
        raise ValueError('矩形跨越投影奇点，本次映射无效。')
    polygon = (homogeneous[:,:2]/denom[:,None]).astype(np.float32)
    center = np.asarray(matrix) @ np.array([(x1+x2)/2,(y1+y2)/2,1])
    center = center[:2]/center[2]
    width,height = size
    boundary = np.float32([[0,0],[width,0],[width,height],[0,height]])
    area, clipped = cv2.intersectConvexConvex(polygon, boundary)
    mask = np.zeros((height,width), dtype=np.uint8)
    bbox = None
    if clipped is not None and area>0:
        points = clipped.reshape(-1,2)
        lower = np.floor(points.min(axis=0)).astype(int)
        upper = np.ceil(points.max(axis=0)).astype(int)
        bx1,by1 = np.maximum(lower,[0,0]); bx2,by2 = np.minimum(upper,[width,height])
        bbox = [int(bx1),int(by1),int(bx2),int(by2)]
        yy,xx = np.mgrid[by1:by2,bx1:bx2]
        # Test pixel centers against each edge; bbox uses x2/y2 exclusive slicing.
        signs = []
        for start,end in zip(polygon,np.roll(polygon,-1,axis=0)):
            signs.append((end[0]-start[0])*(yy+.5-start[1])-(end[1]-start[1])*(xx+.5-start[0]))
        values = np.stack(signs)
        mask[by1:by2,bx1:bx2] = (np.all(values>=-1e-5,axis=0) | np.all(values<=1e-5,axis=0)).astype(np.uint8)
    fully_inside = bool(np.all(polygon>=[0,0]) and np.all(polygon<=[width,height]))
    item = {'polygon_xy': polygon.tolist(), 'center_xy': center.tolist(),
            'bbox_xyxy': bbox, 'polygon_area_px': float(cv2.contourArea(polygon)),
            'visible_polygon_area_px': float(area), 'mask_area_px': int(mask.sum()),
            'bbox_area_px': (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) if bbox else 0,
            'fully_inside_image': fully_inside, 'intersects_image': bool(mask.any())}
    return item,mask
