import json
import math
import re
import os
import time
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from matplotlib.backends.backend_agg import FigureCanvasAgg
import PIL.Image as Image
import os
from datetime import datetime
from zoneinfo import ZoneInfo
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

    def set_boxes(self, boxes_xyxy):
        self.boxes_xyxy = boxes_xyxy if boxes_xyxy and len(boxes_xyxy) > 0 else None      

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
        res_fig = plt.figure(figsize=(5, 5))
        plt.imshow(self.img)
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

        print(f"不规则微粒数量: {len(sorted_anns)}")

        areas = [ann['area'] for ann in sorted_anns[1:]]
        if self.max_d == 0:
            self.max_d = self.px_to_real(s_to_d(areas[0]))
            q1 = np.percentile(areas, 25)
            q3 = np.percentile(areas, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
        else:
            lower_bound = d_to_s(self.min_slider)
            upper_bound = d_to_s(self.max_slider)

        filtered_anns = [ann for ann in sorted_anns if lower_bound <= ann['area'] <= upper_bound]

        if len(filtered_anns) == 0:
            print("没有找到范围内的粒子")
            return

        filtered_anns = sorted(filtered_anns, key=(lambda x: x['area']), reverse=True)

        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((filtered_anns[0]['segmentation'].shape[0], filtered_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0

        particle_color = [0, 0, 1, 0.5]

        particle_count = 0
        particle_areas = []

        self.particle_records = []
        for ann in filtered_anns[1:]:
            m = ann['segmentation']
            img[m] = particle_color
            particle_count += 1
            area = ann['area']
            particle_areas.append(ann['area'])
            ys, xs = np.where(m)
            center_x = int(xs.mean() / self.scale_factor)
            center_y = int(ys.mean() / self.scale_factor)            
            diameter = self.px_to_real(s_to_d(area))
            
            self.particle_records.append({
                'id': particle_count,
                'diameter': diameter,
                'center_x': center_x,
                'center_y': center_y
            })            

        ax.imshow(img)
        print(f"不规则微粒数量: {particle_count}")
        plt.axis('off')

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)

        particle_ds = []
        for area in particle_areas:
            particle_ds.append(self.px_to_real(s_to_d(area)))

        self.result_image = change_plt_to_np(res_fig)
        plt.close(res_fig)
        fig = plt.figure()
        counts, bins, patches = plt.hist(particle_ds, bins=self.fig_length, edgecolor='black')
        plt.xlabel('diameter/' + self.unit)
        plt.ylabel('number')
        plt.title('distribution')

        for count, patch in zip(counts, patches):
            height = patch.get_height()
            plt.annotate(f'{int(count)}',
                         xy=(patch.get_x() + patch.get_width() / 2, height),
                         xytext=(0, 5),
                         textcoords='offset points',
                         ha='center', va='bottom')

        self.result_image_analysis = change_plt_to_np(fig)
        self.result_csv = particle_ds

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

        if self.boxes_xyxy is None or len(self.boxes_xyxy) == 0:
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
            predictor = SamPredictor(sam)
            predictor.set_image(pre_img[:, :, ::-1] if pre_img.shape[2] == 3 else pre_img)

            scaled = []
            for (x1, y1, x2, y2) in self.boxes_xyxy:
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

                masks_np = (masks_t.squeeze(1).detach().cpu().numpy().astype(np.uint8))
                anns = []
                for m in masks_np:
                    area = int(m.sum())
                    anns.append({"segmentation": m.astype(bool), "area": area})

                fake_bg = {"segmentation": np.zeros((H, W), dtype=bool), "area": H * W + 1}
                self.masks = [fake_bg] + anns

        self.show_anns(self.masks)
        self.complete = True
        torch.cuda.empty_cache()
        return self.result_image, self.result_image_analysis
    def set_img(self, img):
        h = img.shape[0]
        w = img.shape[1]
        scale = 1024 / w
        self.scale_factor = 1.0
        if self.is_auto_scale and w > 1024:
            self.img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
            self.scale_factor = 1024 / w
        else:
            self.img = img

    def redraw(self, fig_min_slider, fig_max_slider, fig_length):
        self.max_slider = self.real_to_px(fig_max_slider)
        self.min_slider = self.real_to_px(fig_min_slider)
        self.fig_length = fig_length
        self.show_anns(self.masks)
        return self.result_image, self.result_image_analysis

    def pretreatment(self, image):
        img = cv2.medianBlur(image, 5)
        _, binary_image = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)

        y = binary_image.shape[1]
        pre1_picture = image[0:y]
        scale = image[y:]
        binary_scale = binary_image[y:]

        if self.is_auto_auto_scale_info:
            if scale.shape[0] == 0 or scale.shape[1] == 0:
                raise RuntimeError("无法自动识别比例尺，确保比例尺在图像最下方且为黑底白字，或进行手动输入")
            else:
                y_center = int(binary_scale.shape[0] / 2)
                flag = False
                once = True
                px_length = 0
                for i in range(binary_scale.shape[1]):
                    if binary_scale[y_center][i] == 255 and once:
                        flag = True
                    if flag:
                        once = False
                        px_length += 1
                        if binary_scale[y_center][i] == 0:
                            flag = False
                if px_length == 0:
                    raise RuntimeError("无法自动识别比例尺，请手动输入")
                else:
                    self.px_length = px_length
                ocr = CnOcr()
                out = ocr.ocr(scale)
                if len(out) == 0:
                    raise RuntimeError("无法自动识别比例尺，请手动输入")
                else:
                    text = ''.join([item['text'] for item in out])
                    self.unit = re.search(r"[a-zA-Z]+", text).group()
                    self.distance = int(re.search(r"[0-9]+", text).group())

        if self.manual_px_length and self.manual_px_length > 0:
            self.px_length = self.manual_px_length * self.scale_factor
        if self.manual_distance != 0:
            self.distance = self.manual_distance
        if self.manual_unit != '':
            self.unit = self.manual_unit

        return pre1_picture
