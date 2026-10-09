import gc
import json
import time
import zipfile
from datetime import datetime

from runtime_config import ROOT
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from romatch import roma_indoor, roma_outdoor

torch.set_float32_matmul_precision("highest")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_variant = None


def get_model(variant):
    global _model, _variant
    backbone = ROOT / "models/hub/checkpoints/dinov2_vitl14_pretrain.pth"
    if not backbone.exists() or backbone.stat().st_size != 1217586395:
        raise ValueError("模型权重尚未准备完成，请等待下载完成后再匹配。")
    if _variant != variant:
        _model = None
        _variant = None
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        factory = roma_indoor if variant == "indoor" else roma_outdoor
        _model = factory(device=DEVICE, use_custom_corr=False).eval()
        _variant = variant
    return _model


def parse_points(text, size):
    points = []
    for line in text.splitlines():
        if not line.strip():
            continue
        name, x, y = line.replace("，", ",").rsplit(",", 2)
        x, y = float(x), float(y)
        if not name.strip() or not (0 <= x < size[0] and 0 <= y < size[1]):
            raise ValueError("按钮名称不能为空，坐标需要在参考图片内。")
        points.append({"name": name.strip(), "reference_xy": [x, y]})
    return points


def label_points(image, points, key):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    font_path = "C:/Windows/Fonts/msyh.ttc"
    font = ImageFont.truetype(font_path, max(16, image.width // 65))
    radius = max(5, image.width // 180)
    for point in points:
        x, y = point[key]
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline="red", width=3)
        draw.text((x+radius+3, y), point["name"], fill="red", font=font)
    return result


def run_match(reference, current, variant="outdoor", point_text="", confidence=0.5):
    if reference is None or current is None:
        raise ValueError("请先上传参考照片和实时照片。")
    reference = ImageOps.exif_transpose(reference).convert("RGB")
    current = ImageOps.exif_transpose(current).convert("RGB")
    points = parse_points(point_text, reference.size)
    started = time.perf_counter()
    model = get_model(variant)
    torch.manual_seed(0)
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        warp, certainty = model.match(reference, current, device=DEVICE)
        # Keep the original confidences: model.sample() thresholds them internally.
        valid = torch.isfinite(warp).all(dim=-1) & (certainty >= confidence)
        selected = warp[valid]
        scores = certainty[valid]
        if len(selected) < 8:
            raise ValueError("高置信度匹配不足，建议保留面板文字和边缘，重新拍摄。")
        if len(selected) > 5000:
            indexes = torch.linspace(0, len(selected)-1, 5000, device=DEVICE).long()
            selected, scores = selected[indexes], scores[indexes]
        a, b = model.to_pixel_coordinates(selected, reference.height, reference.width,
                                         current.height, current.width)
        a, b, scores = a.cpu().numpy(), b.cpu().numpy(), scores.cpu().numpy()
    matrix, mask = cv2.findHomography(a, b, cv2.USAC_MAGSAC, 3.0,
                                     maxIters=10000, confidence=0.999)
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        raise ValueError("无法建立可靠的面板配准，请换一组照片。")
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < 8 or inliers.mean() < 0.2:
        raise ValueError("几何一致的匹配太少，本次不输出按钮坐标。")
    projected = cv2.perspectiveTransform(a[inliers].reshape(-1, 1, 2), matrix)[:, 0]
    residual = np.linalg.norm(projected - b[inliers], axis=1)
    mapped = []
    for point in points:
        xy = cv2.perspectiveTransform(np.float32([[point["reference_xy"]]]), matrix)[0, 0]
        if not np.isfinite(xy).all():
            continue
        mapped.append({**point, "current_xy": xy.tolist(),
                       "inside_image": bool(0 <= xy[0] < current.width and 0 <= xy[1] < current.height)})
    ref_array, cur_array = np.array(reference), np.array(current)
    aligned = cv2.warpPerspective(ref_array, matrix, current.size)
    coverage = cv2.warpPerspective(np.ones(ref_array.shape[:2], np.uint8), matrix, current.size) > 0
    overlay = cur_array.copy()
    overlay[coverage] = (0.5 * aligned[coverage] + 0.5 * cur_array[coverage]).astype(np.uint8)
    ids = np.flatnonzero(inliers)
    ids = ids[np.linspace(0, len(ids)-1, min(100, len(ids))).astype(int)]
    connections = cv2.drawMatches(ref_array, [cv2.KeyPoint(*map(float, xy), 2) for xy in a],
                                 cur_array, [cv2.KeyPoint(*map(float, xy), 2) for xy in b],
                                 [cv2.DMatch(int(i), int(i), 0) for i in ids], None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    marked = label_points(current, [p for p in mapped if p["inside_image"]], "current_xy")
    elapsed = time.perf_counter() - started
    report = {"model": variant, "device": DEVICE, "seconds": round(elapsed, 2),
              "peak_gpu_MB": round(torch.cuda.max_memory_allocated()/1024**2, 1) if DEVICE == "cuda" else None,
              "matches": len(a), "inliers": int(inliers.sum()),
              "median_fit_residual_px": float(np.median(residual)),
              "homography": matrix.tolist(), "buttons": mapped,
              "note": "拟合残差不是按钮真实误差；映射点需要核验可见性和按钮身份。"}
    folder = ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir()
    Image.fromarray(connections).save(folder / "matches.png")
    Image.fromarray(overlay).save(folder / "overlay.png")
    marked.save(folder / "buttons.png")
    (folder / "coordinates.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = folder / "results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in folder.iterdir():
            if path != archive:
                bundle.write(path, path.name)
    summary = (f"{variant} · {DEVICE} · {elapsed:.1f} 秒\n"
               f"几何一致匹配：{inliers.sum()} / {len(a)}；拟合残差中位数：{np.median(residual):.2f} 像素。\n"
               "该残差不代表按钮真实定位误差。红点为参考标注的映射位置，请核对按钮身份和遮挡。")
    return Image.fromarray(connections), Image.fromarray(overlay), marked, summary, str(archive)
