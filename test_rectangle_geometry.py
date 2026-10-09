import unittest
import numpy as np
from rectangle_geometry import validate_buttons, map_rectangle


class RectangleGeometryTests(unittest.TestCase):
    def test_identity_has_half_open_yolo_box_and_exact_mask_area(self):
        item, mask = map_rectangle([10, 20, 30, 40], np.eye(3), (100, 80))
        self.assertEqual(item['bbox_xyxy'], [10, 20, 30, 40])
        self.assertEqual(item['mask_area_px'], 400)
        self.assertEqual(mask[20:40, 10:30].sum(), 400)
        self.assertEqual(item['center_xy'], [20, 30])

    def test_perspective_maps_four_corners_and_center_separately(self):
        matrix = np.array([[1, .3, 5], [.1, 1, 8], [.002, 0, 1]])
        item, mask = map_rectangle([10, 20, 30, 40], matrix, (100, 80))
        self.assertEqual(len(item['polygon_xy']), 4)
        self.assertLess(item['polygon_area_px'], item['bbox_area_px'])
        p = matrix @ np.array([20, 30, 1])
        np.testing.assert_allclose(item['center_xy'], p[:2]/p[2], atol=1e-5)

    def test_partial_and_outside_boxes_report_clipping(self):
        item, mask = map_rectangle([-10, 0, 10, 20], np.eye(3), (100, 80))
        self.assertFalse(item['fully_inside_image'])
        self.assertEqual(item['bbox_xyxy'], [0, 0, 10, 20])
        self.assertEqual(item['mask_area_px'], 200)
        item, mask = map_rectangle([110, 0, 130, 20], np.eye(3), (100, 80))
        self.assertFalse(item['intersects_image'])
        self.assertEqual(mask.sum(), 0)

    def test_horizon_and_invalid_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            map_rectangle([0, 0, 20, 20], np.array([[1,0,0],[0,1,0],[.1,0,-1]]), (100,80))
        template = [{'name':'key_1','reference_xy':[20,30]}]
        for box in ([30,20,10,40], [10,20,float('nan'),40], [0,0,101,80]):
            with self.assertRaises(ValueError):
                validate_buttons([{'name':'key_1','rect_xyxy':box}], template, (100,80))


if __name__ == '__main__':
    unittest.main()
