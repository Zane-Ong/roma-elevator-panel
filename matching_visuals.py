import cv2
import numpy as np
from PIL import Image


def matching_visuals(reference, current, a, b, inliers, matrix):
    ref = np.array(reference)
    live = np.array(current)
    warped = cv2.warpPerspective(ref, matrix, current.size)
    coverage = cv2.warpPerspective(np.ones(ref.shape[:2], np.uint8), matrix, current.size) > 0
    overlay = live.copy()
    overlay[coverage] = (0.5 * warped[coverage] + 0.5 * live[coverage]).astype(np.uint8)
    # Scale visualizations only; matching and exported coordinates keep original units.
    height = min(720, reference.height, current.height)
    ref_scale, live_scale = height/reference.height, height/current.height
    ref_small = cv2.resize(ref, (round(reference.width*ref_scale), height))
    live_small = cv2.resize(live, (round(current.width*live_scale), height))
    indexes = np.flatnonzero(inliers)
    indexes = indexes[np.linspace(0, len(indexes)-1, min(100, len(indexes))).astype(int)]
    keys_a = [cv2.KeyPoint(float(x*ref_scale), float(y*ref_scale), 2) for x,y in a[indexes]]
    keys_b = [cv2.KeyPoint(float(x*live_scale), float(y*live_scale), 2) for x,y in b[indexes]]
    connections = cv2.drawMatches(ref_small, keys_a, live_small, keys_b,
                                 [cv2.DMatch(i,i,0) for i in range(len(indexes))], None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    overlay = Image.fromarray(overlay)
    overlay.thumbnail((1400,1400))
    return Image.fromarray(connections), overlay
