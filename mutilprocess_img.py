import math
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from matplotlib.backends.backend_agg import FigureCanvasAgg
import PIL.Image as Image
from cnocr import CnOcr

def change_plt_to_np(plt_image):
    canvas = FigureCanvasAgg(plt.gcf())
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
    plt_image.canvas.draw()
    w, h = plt_image.canvas.get_width_height()
    buf = np.fromstring(plt_image.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = np.asarray(image)
    rgb_image = image[:, :, :3]
    return rgb_image

def s_to_d(s):
    return int(math.pow(4 * s / math.pi, 0.5) * 100) / 100

def d_to_s(d):
    return int(d * d * 0.25 * math.pi * 100) / 100


class ImgProcessing:
    def __init__(self):
        self.is_auto_scale = True
        self.fig_length = 10
        self.manual_distance = None
        self.manual_unit = None
        self.unit = None
        self.distance = None
        self.px_length = None
        self.min_slider = None
        self.max_slider = None
        self.masks = None
        self.img = None
        self.result_image_analysis = None
        self.result_image = None
        self.complete = False
        self.max_d = 0
        self.max_area = 0
        self.points_per_side = 48
        self.pred_iou_thresh = 0.4
        self.stability_score_thresh = 0.5
        self.crop_n_layers = 1
        self.crop_n_points_downscale_factor = 5
        self.min_mask_region_area = 50
        self.result_csv = list()
        self.is_auto_auto_scale_info = False
        self.boxes_xyxy = None
        self.points_xy = None
        self.point_box_radius = 15
        self.label_mask = None
        self.original_shape = None
        self.label_mask = None
        self.original_shape = None
        self.original_img = None        

    def set_boxes(self, boxes_xyxy):
        self.boxes_xyxy = boxes_xyxy if boxes_xyxy and len(boxes_xyxy) > 0 else None      
    
    def set_points(self, roi_points):
        if roi_points and len(roi_points) > 0:
            self.points_xy = np.array(roi_points, dtype=np.float32)
        else:
            self.points_xy = None

    def _update_point_box_radius_from_boxes(self):
        if not self.boxes_xyxy:
            return

        radii = []
        for (x1, y1, x2, y2) in self.boxes_xyxy:
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if w > 0 and h > 0:
                radii.append(min(w, h) / 2.0)

        if radii:
            median_r = np.median(radii)
            self.point_box_radius = max(1, int(round(median_r)))

    def points_to_boxes(self, radius=None):
        if self.points_xy is None:
            return None

        if radius is None:
            self._update_point_box_radius_from_boxes()
            radius = self.point_box_radius

        boxes = []
        for (x, y) in self.points_xy:
            x1 = int(x - radius)
            y1 = int(y - radius)
            x2 = int(x + radius)
            y2 = int(y + radius)
            boxes.append([x1, y1, x2, y2])
        return boxes
    
    def set_config(self,
                   points_per_side,
                   pred_iou_thresh,
                   stability_score_thresh,
                   crop_n_layers,
                   crop_n_points_downscale_factor,
                   min_mask_region_area,
                   manual_distance,
                   manual_unit,
                   open_auto_scale_info,
                   px_length_input
                   ):
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.manual_distance = manual_distance
        self.manual_unit = manual_unit
        self.is_auto_auto_scale_info = open_auto_scale_info
        self.is_auto_scale = True
        self.manual_px_length = px_length_input          

    def px_to_real(self, px):
        
        scale = self.distance / self.px_length
        return px * scale

    def real_to_px(self, real):
        scale = self.px_length / self.distance
        return real * scale
    
    def show_anns(self, anns):
        plt.close('all')

        if not anns:
            print("No masks to show.")
            if self.original_img is not None:
                self.result_image = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
            elif self.img is not None:
                self.result_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            else:
                self.result_image = None
            self.result_image_analysis = None
            self.particle_records = []
            self.label_mask = None
            return

        valid_anns = []
        for ann in anns:
            seg = ann.get("segmentation")
            if seg is None:
                continue
            seg = seg.astype(bool)
            area = int(seg.sum())
            if area <= 0:
                continue
            valid_anns.append({"segmentation": seg, "area": area})

        if len(valid_anns) == 0:
            print("No valid masks with non-zero area.")
            if self.original_img is not None:
                self.result_image = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
            elif self.img is not None:
                self.result_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            else:
                self.result_image = None
            self.result_image_analysis = None
            self.particle_records = []
            self.label_mask = None
            return

        valid_anns = sorted(valid_anns, key=lambda x: x["area"], reverse=True)
        areas = [a["area"] for a in valid_anns]

        if self.max_d == 0:
            largest_d_px = s_to_d(areas[0])
            self.max_d = self.px_to_real(largest_d_px)

            q1 = np.percentile(areas, 25)
            q3 = np.percentile(areas, 75)
            iqr = q3 - q1
            lower_bound = max(0, q1 - 1.5 * iqr)
            upper_bound = q3 + 1.5 * iqr
        else:
            lower_bound = d_to_s(self.min_slider)
            upper_bound = d_to_s(self.max_slider)

        filtered_anns = [a for a in valid_anns if lower_bound <= a["area"] <= upper_bound]

        if len(filtered_anns) == 0:
            print("No particles found within the specified range.")
            if self.original_img is not None:
                self.result_image = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
            elif self.img is not None:
                self.result_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            else:
                self.result_image = None
            self.result_image_analysis = None
            self.particle_records = []
            self.label_mask = None
            return

        H, W = filtered_anns[0]["segmentation"].shape
        label_mask = np.zeros((H, W), dtype=np.uint16)

        particle_ds = []
        self.particle_records = []
        particle_id = 0

        for ann in filtered_anns:
            seg = ann["segmentation"]
            area = ann["area"]

            particle_id += 1

            ys, xs = np.where(seg)
            if xs.size == 0:
                continue

            center_x = int(xs.mean() / self.scale_factor)
            center_y = int(ys.mean() / self.scale_factor)
            diameter = self.px_to_real(s_to_d(area))

            particle_ds.append(diameter)

            label_mask[seg] = particle_id

            self.particle_records.append({
                "id": particle_id,
                "diameter": diameter,
                "center_x": center_x,
                "center_y": center_y
            })

        if self.original_img is not None:
            base_rgb = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
        else:
            base_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        mask_full = label_mask
        if self.original_shape and label_mask.shape[:2] != self.original_shape:
            h, w = self.original_shape
            mask_full = cv2.resize(label_mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint16)

        overlay_rgb = base_rgb.copy()
        mask_bool = mask_full > 0
        if np.any(mask_bool):
            alpha = 0.5
            overlay_color = np.array([0, 0, 255], dtype=np.float32)
            overlay_float = overlay_rgb.astype(np.float32)
            overlay_float[mask_bool] = (
                (1.0 - alpha) * overlay_float[mask_bool] + alpha * overlay_color
            )
            overlay_rgb = overlay_float.astype(np.uint8)

        self.result_image = overlay_rgb

        fig = plt.figure()
        counts, bins, patches = plt.hist(
            particle_ds,
            bins=self.fig_length,
            edgecolor="black"
        )
        plt.xlabel("diameter/" + self.unit)
        plt.ylabel("number")
        plt.title("distribution")

        for count, patch in zip(counts, patches):
            height = patch.get_height()
            if height > 0:
                plt.annotate(
                    f"{int(count)}",
                    xy=(patch.get_x() + patch.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom"
                )

        self.result_image_analysis = change_plt_to_np(fig)
        plt.close(fig)

        self.result_csv = particle_ds
        self.label_mask = label_mask

    def show_img(self):
        image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(5, 5))
        plt.imshow(image)
        plt.axis('off')

        pre1_picture = self.pretreatment(image)

        sam_checkpoint = "./model/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        torch.cuda.empty_cache()

        pre_img = pre1_picture
        H, W = pre_img.shape[:2]

        has_boxes = self.boxes_xyxy is not None and len(self.boxes_xyxy) > 0
        has_points = self.points_xy is not None and len(self.points_xy) > 0

        if not has_boxes and not has_points:
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=self.points_per_side,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
                crop_n_layers=self.crop_n_layers,
                crop_n_points_downscale_factor=self.crop_n_points_downscale_factor,
                min_mask_region_area=self.min_mask_region_area,
            )
            self.masks = mask_generator.generate(pre_img)

        else:
            all_boxes = []

            if has_boxes:
                all_boxes.extend(self.boxes_xyxy)

            if has_points:
                point_boxes = self.points_to_boxes()
                all_boxes.extend(point_boxes)

            predictor = SamPredictor(sam)
            predictor.set_image(pre_img[:, :, ::-1] if pre_img.shape[2] == 3 else pre_img)

            scaled = []
            for (x1, y1, x2, y2) in all_boxes:
                sx1 = int(round(x1 * self.scale_factor))
                sy1 = int(round(y1 * self.scale_factor))
                sx2 = int(round(x2 * self.scale_factor))
                sy2 = int(round(y2 * self.scale_factor))

                sy1c = max(0, min(H, sy1))
                sy2c = max(0, min(H, sy2))
                sx1c = max(0, min(W, sx1))
                sx2c = max(0, min(W, sx2))

                if sx2c > sx1c and sy2c > sy1c:
                    scaled.append([sx1c, sy1c, sx2c, sy2c])

            if len(scaled) == 0:
                mask_generator = SamAutomaticMaskGenerator(
                    model=sam,
                    points_per_side=self.points_per_side,
                    pred_iou_thresh=self.pred_iou_thresh,
                    stability_score_thresh=self.stability_score_thresh,
                    crop_n_layers=self.crop_n_layers,
                    crop_n_points_downscale_factor=self.crop_n_points_downscale_factor,
                    min_mask_region_area=self.min_mask_region_area,
                )
                self.masks = mask_generator.generate(pre_img)
            else:
                boxes_t = torch.tensor(scaled, device=device, dtype=torch.float32)
                boxes_t = predictor.transform.apply_boxes_torch(boxes_t, pre_img.shape[:2])
                masks_t, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=boxes_t,
                    multimask_output=False
                )

                masks_np = masks_t.squeeze(1).detach().cpu().numpy().astype(np.uint8)

                anns = []
                for m in masks_np:
                    area = int(m.sum())
                    anns.append({"segmentation": m.astype(bool), "area": area})

                fake_bg = {
                    "segmentation": np.zeros((H, W), dtype=bool),
                    "area": H * W + 1
                }
                self.masks = [fake_bg] + anns

        self.show_anns(self.masks)
        self.complete = True
        torch.cuda.empty_cache()
        return self.result_image, self.result_image_analysis


    def set_img(self, img):
        h, w = img.shape[:2]
        self.original_shape = (h, w)
        self.original_img = img.copy()
        scale = 1024 / w
        self.scale_factor = 1.0
        if self.is_auto_scale and w > 1024:
            self.img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
            self.scale_factor = 1024 / w
        else:
            self.img = img

    def get_label_mask_original_size(self):
        if self.label_mask is None:
            return None
        if self.original_shape and self.label_mask.shape[:2] != self.original_shape:
            h, w = self.original_shape
            return cv2.resize(self.label_mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint16)
        return self.label_mask

    def get_binary_mask_original_size(self):
        mask = self.get_label_mask_original_size()
        if mask is None:
            return None
        return (mask > 0).astype(np.uint8) * 255

    def redraw(self, fig_min_slider, fig_max_slider, fig_length):
        self.max_slider = self.real_to_px(fig_max_slider)
        self.min_slider = self.real_to_px(fig_min_slider)
        self.fig_length = fig_length
        self.show_anns(self.masks)
        return self.result_image, self.result_image_analysis

    def pretreatment(self, image):
        img = cv2.medianBlur(image, 5)
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        pre1_picture = image

        if self.is_auto_auto_scale_info:
            h_img, w_img = gray.shape[:2]

            
            v = np.median(gray)
            lower = int(max(0, 0.66 * v))
            upper = int(min(255, 1.33 * v))
            edges = cv2.Canny(gray, lower, upper)

            
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180.0,
                threshold=50,
                minLineLength=int(0.1 * w_img),
                maxLineGap=10
            )

            if lines is None:
                raise RuntimeError(
                    "Unable to automatically detect any scale bar lines (HoughLinesP failed). "
                    "Please enter the scale manually."
                )

            bar_candidates = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = math.hypot(x2 - x1, y2 - y1)
                if length < 0.1 * w_img:
                    continue

                
                angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                if angle > 15:
                    continue

                
                y_mid = (y1 + y2) / 2.0
                x_min = min(x1, x2)
                x_max = max(x1, x2)
                if not (
                    y_mid > 0.6 * h_img
                    or x_min < 0.15 * w_img
                    or x_max > 0.85 * w_img
                ):
                    continue

                bar_candidates.append((x1, y1, x2, y2, length))

            if not bar_candidates:
                raise RuntimeError(
                    "Detected some lines, but none look like a horizontal scale bar. "
                    "Please enter the scale manually."
                )

            bar_candidates.sort(key=lambda t: -t[4])

            ocr = CnOcr()
            units_pattern = r"(nm|um|µm|μm|mm)"
            found_scale = False

            for (x1, y1, x2, y2, length) in bar_candidates:
                cx = int(round((x1 + x2) / 2.0))
                cy = int(round((y1 + y2) / 2.0))

                bar_half_thickness = max(int(0.01 * h_img), 3)
                x0_bar = max(cx - int(length / 2), 0)
                x1_bar = min(cx + int(length / 2), w_img - 1)
                y0_bar = max(cy - bar_half_thickness, 0)
                y1_bar = min(cy + bar_half_thickness, h_img - 1)

                bar_w = x1_bar - x0_bar + 1
                bar_h = y1_bar - y0_bar + 1
                if bar_w <= 0 or bar_h <= 0:
                    continue

                px_length_candidate = bar_w

                roi_x0 = max(x0_bar - bar_w // 2, 0)
                roi_x1 = min(x1_bar + bar_w // 2, w_img - 1)
                roi_y0 = max(y0_bar - 4 * bar_h, 0)
                roi_y1 = min(y1_bar + 4 * bar_h, h_img - 1)

                roi = image[roi_y0:roi_y1 + 1, roi_x0:roi_x1 + 1]

                out = ocr.ocr(roi)
                if not out:
                    continue

                texts = [
                    item.get("text", "").replace(" ", "")
                    for item in out
                    if item.get("text")
                ]
                if not texts:
                    continue

                candidate_text = None
                n = len(texts)

                for i in range(n):
                    combos = [texts[i]]
                    if i + 1 < n:
                        combos.append(texts[i] + texts[i + 1])

                    matched = False
                    for combo in combos:
                        if (re.search(r"\d", combo) and
                                re.search(units_pattern, combo, re.IGNORECASE)):
                            candidate_text = combo
                            matched = True
                            break
                    if matched:
                        break

                if candidate_text is None:
                    full = "".join(texts)
                    if (re.search(r"\d", full) and
                            re.search(units_pattern, full, re.IGNORECASE)):
                        candidate_text = full

                if candidate_text is None:
                    continue

                m_num = re.search(r"(\d+)", candidate_text)
                m_unit = re.search(units_pattern, candidate_text, re.IGNORECASE)
                if not m_num or not m_unit:
                    continue

                self.px_length = int(px_length_candidate)
                self.distance = int(m_num.group(1))
                self.unit = m_unit.group(1)
                found_scale = True
                break

            if not found_scale:
                raise RuntimeError(
                    "Detected horizontal line candidates, but failed to read scale text. "
                    "Please enter the scale manually."
                )

        if self.manual_px_length and self.manual_px_length > 0:
            self.px_length = self.manual_px_length * self.scale_factor
        if self.manual_distance != 0:
            self.distance = self.manual_distance
        if self.manual_unit != '':
            self.unit = self.manual_unit
        return pre1_picture