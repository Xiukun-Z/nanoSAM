import os
os.environ["GRADIO_LANGUAGE"] = "en"
import gradio as gr
import time
import zipfile
import io
import csv
from PIL import Image
import numpy as np
import cv2
from rsciio.digitalmicrograph import file_reader as dm_read
from rsciio.emd import file_reader as emd_read
from rsciio.tia import file_reader as tia_read
from rsciio.tiff import file_reader as tiff_read

from mutilprocess_img import ImgProcessing
from img import logo_img

# css
css = """
#max-image {
    height: 400px;
}
img {
    max-height: 100%;
    width: auto;
}

div.head-bar {
    height: 110px;
    display: flex;
    align-items: center;
    justify-content: center;    
    box-shadow: 0 15px 10px #000000;
}
.header-container {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 24px;
    width: auto;
}
.header-container a {
    display: flex;
    align-items: center;
}
.header-text {
    font-size: 2.5rem;
    color: #52c2f9;
    text-shadow: 1px 1px 2px black;
    text-align: center;
}
footer.svelte-1rjryqp {
    display:none !important;
}
.footer-text{
    display: flex;
    justify-content: center;
    margin-top: var(--size-4);
    color: var(--body-text-color-subdued);
}

#download-zip label {
    font-size: 1.5rem !important;
}
"""

footer_ele = """
<footer class="footer-text">
Copyright © 2024–2026 Smart Lab @ TJU. All rights reserved.
</footer>
"""


class WebUI:

    def __init__(self):
        self.pro_img = None
        self.input_img_tif = None
        self.roi_boxes = []
        self.roi_points = []
        self.batch_last_zip = None
        self.input_img_path = None

        with gr.Blocks(
            css=css,
            title="The Advanced Instrumental Analysis Center, School of Chemical Engineering and Technology, Tianjin University"
        ) as demo:
            with gr.Row(elem_classes='head-bar'):
                gr.Markdown(value=f"<div class='header-container'>"
                                  f"<a href='https://www.clickgene.org/about/'>"
                                  f"<img style='height:60px;width:auto' src='{logo_img}'/></a>"
                                  f"<div class='header-text'>"
                                  f"The Advanced Instrumental Analysis Center, School of Chemical Engineering and Technology, Tianjin University"
                                  f"</div></div>")
            with gr.Row():
                with gr.Column():
                    gr.Markdown('## Single Image Processing (Interactive)')
                    self.input_file = gr.File(
                        label="Single Input (image or raw data)",
                        file_types=["image", ".dm3", ".dm4", ".tif", ".tiff", ".emd", ".emi"]
                    )
                    self.input_img = gr.Image(
                        elem_id="max-image",
                        image_mode="RGBA",
                        interactive=False
                    )
                    gr.Markdown('## Batch Image Processing (same experiment, shared parameters)')
                    self.batch_files = gr.Files(
                        label="Batch Input Images (same experiment)",
                        file_types=["image", ".dm3", ".dm4", ".tif", ".tiff", ".emd", ".emi"]
                    )

                    gr.Markdown('## Parameter Settings (shared for single & batch)')
                    self.process_speed = gr.Dropdown(
                        ["Low Quality", "Medium Quality", "High Quality"],
                        label="Processing Quality",
                        value='Medium Quality'
                    )
                    self.points_per_side = 48
                    self.pred_iou_thresh = 0.4
                    self.stability_score_thresh = 0.5
                    self.crop_n_layers = 1
                    self.crop_n_points_downscale_factor = 5
                    self.min_mask_region_area = 50

                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                self.img_distance = gr.Number(
                                    value=0,
                                    label='Scale Physical Length'
                                )
                                self.img_unit = gr.Dropdown(
                                    ["nm", "um", "mm", "cm"],
                                    label="Scale Unit",
                                    value='nm'
                                )
                                self.px_length_input = gr.Number(
                                    value=0,
                                    label='Scale Length in Pixels (px)'
                                )
                            self.detect_scale_btn = gr.Button("Detect Scale from Current Image")

                    self.fig_length = gr.Number(
                        value=10,
                        label='Table Step Length'
                    )

                    self.box_info = gr.Markdown('No boxes selected')
                    self.select_box_btn = gr.Button("Select ROI")
                    self.upload_button = gr.Button("Start Nanoparticle Recognition (single image)")
                    self.batch_button = gr.Button("Start Batch Recognition (shared parameters)")
                    self.clear_box_btn = gr.Button("Clear Boxes (Click after recognition)")

                    self.process_speed.change(
                        self.update_dropdowns,
                        inputs=[self.process_speed],
                        outputs=[]
                    )

                with gr.Column():
                    gr.Markdown('# Processing Results (single image)')
                    self.output = gr.Image(elem_id="output-image")
                    gr.Markdown('# Nanoparticle Size Statistics Chart (single image)')
                    self.output_fig = gr.Image(elem_id="fig-image")
                    with gr.Row():
                        self.fig_min_slider = gr.Slider(label="Minimum Diameter", value=0)
                        self.fig_max_slider = gr.Slider(label="Maximum Diameter", value=0)
                    self.redraw_fig = gr.Button("Change Diameter")

                    gr.Markdown('## Export Results')
                    self.output_csv = gr.Button("Export Last Result (single image)")
                    self.download_zip = gr.File(
                        label="Click to download the latest zip results",
                        elem_id="download-zip"
                    )

            self.detect_scale_btn.click(
                self.detect_scale_from_img,
                inputs=[],
                outputs=[self.img_distance, self.img_unit, self.px_length_input]
            )

            self.select_box_btn.click(
                self.select_boxes_popup,
                inputs=[],
                outputs=[self.box_info]
            )
            self.clear_box_btn.click(
                self.clear_boxes,
                inputs=[],
                outputs=[self.box_info]
            )
            self.footer = gr.HTML(footer_ele)

            self.upload_button.click(
                self.handle_img,
                inputs=[
                    self.img_distance,
                    self.img_unit,
                    self.px_length_input,
                ],
                outputs=[
                    self.output,
                    self.output_fig,
                    self.fig_min_slider,
                    self.fig_max_slider,
                    self.img_distance,
                    self.img_unit,
                    self.px_length_input
                ]
            )

            self.batch_button.click(
                self.handle_batch,
                inputs=[
                    self.batch_files,
                    self.img_distance,
                    self.img_unit,
                    self.px_length_input,
                    self.fig_min_slider,
                    self.fig_max_slider,
                    self.fig_length,                
                ],
                outputs=[self.download_zip]
            )

            self.redraw_fig.click(
                self.redraw,
                inputs=[self.fig_min_slider, self.fig_max_slider, self.fig_length],
                outputs=[self.output, self.output_fig]
            )

            self.input_file.upload(
                self.handle_single_upload,
                inputs=[self.input_file, self.img_distance, self.img_unit, self.px_length_input],
                outputs=[self.input_img, self.img_distance, self.img_unit, self.px_length_input]
            )

            self.output_csv.click(
                self.export_results,
                inputs=[],
                outputs=[self.download_zip]
            )

        demo.launch(show_error=True, server_name="127.0.0.1", server_port=7860)
    def handle_single_upload(self, file, cur_distance, cur_unit, cur_px_length):

        if file is None:
            raise gr.Error("Please upload a file.")

        path = file.name if hasattr(file, "name") else file
        self.input_img_path = path

        img_rgb, pixel_size, meta_unit = load_with_metadata(path)

        if img_rgb is not None:
            distance = float(pixel_size)
            unit = meta_unit
            px_length = 1.0

        else:
            try:
                pil_img = Image.open(path).convert("RGB")
            except Exception:
                raise gr.Error("Unsupported file format: cannot decode as image or known raw data.")
            img_rgb = np.array(pil_img)

            distance = cur_distance
            unit = cur_unit
            px_length = cur_px_length

        self.input_img_tif = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        return img_rgb, distance, unit, px_length

    def fix_image(self, input_img_path):
        self.input_img_path = input_img_path
        img = cv2.imread(input_img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise gr.Error("Failed to read image. Please upload a valid image file.")

        self.input_img_tif = img
        return input_img_path

    def detect_scale_from_img(self):
        if self.input_img_path is not None:
            _, pixel_size, unit = load_with_metadata(self.input_img_path)
            if pixel_size is not None and unit is not None:
                return float(pixel_size), unit, 1.0

        if self.input_img_tif is None:
            raise gr.Error("Please upload an image before detecting scale bar.")

        proc = ImgProcessing()
        proc.set_config(
            self.points_per_side,
            self.pred_iou_thresh,
            self.stability_score_thresh,
            self.crop_n_layers,
            self.crop_n_points_downscale_factor,
            self.min_mask_region_area,
            manual_distance=0,
            manual_unit='',
            open_auto_scale_info=True,
            px_length_input=0
        )

        proc.set_img(self.input_img_tif)

        image = cv2.cvtColor(proc.img, cv2.COLOR_BGR2RGB)

        try:
            proc.pretreatment(image)
        except RuntimeError as e:
            raise gr.Error(str(e))

        return proc.distance, proc.unit, proc.px_length / proc.scale_factor

    def handle_img(
        self,
        img_distance,
        img_unit,
        px_length_input,
    ):

        self.pro_img = ImgProcessing()
        self.pro_img.set_config(
            self.points_per_side,
            self.pred_iou_thresh,
            self.stability_score_thresh,
            self.crop_n_layers,
            self.crop_n_points_downscale_factor,
            self.min_mask_region_area,
            img_distance,
            img_unit,
            False,
            px_length_input,
        )
        self.pro_img.set_img(self.input_img_tif)
        self.pro_img.set_boxes(self.roi_boxes)
        self.pro_img.set_points(self.roi_points)
        show_next = self.pro_img.show_img()
        res, res_fig = show_next

        unit = self.pro_img.unit
        scale_factor = self.pro_img.scale_factor
        max_text = "Minimum Diameter/" + unit
        min_text = "Maximum Diameter/" + unit
        self.fig_min_slider = gr.Slider(label=max_text, minimum=0, maximum=self.pro_img.max_d)
        self.fig_max_slider = gr.Slider(
            label=min_text,
            value=self.pro_img.max_d,
            minimum=0,
            maximum=self.pro_img.max_d
        )
        return (
            res,
            res_fig,
            self.fig_min_slider,
            self.fig_max_slider,
            self.pro_img.distance,
            self.pro_img.unit,
            self.pro_img.px_length / scale_factor
        )

    def redraw(self, fig_min_slider, fig_max_slider, fig_length):
        show_next = self.pro_img.redraw(fig_min_slider, fig_max_slider, fig_length)
        res, res_fig = show_next
        return res, res_fig

    def select_boxes_popup(self):
        if self.input_img_tif is None:
            raise gr.Error("Please upload an image before selecting boxes/points.")

        img = self.input_img_tif
        if img.ndim == 2:
            base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            base = img.copy()

        H, W = base.shape[:2]
        max_w, max_h = 1600, 900
        scale = min(max_w / W, max_h / H, 1.0)
        disp = cv2.resize(base, (int(W * scale), int(H * scale))) if scale != 1.0 else base.copy()

        boxes = []
        points = []
        drawing = False
        x0 = y0 = x1 = y1 = 0

        mode = "box"

        def on_mouse(event, x, y, flags, param):
            nonlocal drawing, x0, y0, x1, y1, boxes, points, mode
            if event == cv2.EVENT_LBUTTONDOWN:
                if mode == "box":
                    drawing = True
                    x0, y0 = x, y
                    x1, y1 = x, y
                else:
                    inv = 1.0 / scale
                    px = int(round(x * inv))
                    py = int(round(y * inv))
                    points.append((px, py))
            elif event == cv2.EVENT_MOUSEMOVE and drawing and mode == "box":
                x1, y1 = x, y
            elif event == cv2.EVENT_LBUTTONUP and drawing and mode == "box":
                drawing = False
                x1, y1 = x, y

        win = "ROI / Point Selector (B: box, P: point, Z: undo box, X: undo point, ESC: exit)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, on_mouse)

        while True:
            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            frame = disp.copy()

            for bx in boxes:
                dx1, dy1, dx2, dy2 = [int(v * scale) for v in bx]
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)

            if mode == "box" and (drawing or (x0 != x1 and y0 != y1)):
                cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)

            for (px, py) in points:
                dx = int(round(px * scale))
                dy = int(round(py * scale))
                cv2.circle(frame, (dx, dy), 4, (0, 255, 0), -1)

            mode_text = f"MODE: {mode.upper()}"
            cv2.putText(frame, mode_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(win, frame)
            key = cv2.waitKey(10) & 0xFF

            if key == 27:
                break
            elif key in (13, 10):
                if mode == "box":
                    xa, xb = sorted([x0, x1])
                    ya, yb = sorted([y0, y1])
                    if xb - xa > 2 and yb - ya > 2:
                        inv = 1.0 / scale
                        bx = [
                            int(round(xa * inv)),
                            int(round(ya * inv)),
                            int(round(xb * inv)),
                            int(round(yb * inv)),
                        ]
                        boxes.append(bx)
                    x0 = y0 = x1 = y1 = 0
            elif key in (ord('z'), ord('Z'), 8):
                if boxes:
                    boxes.pop()
            elif key in (ord('x'), ord('X')):
                if points:
                    points.pop()
            elif key in (ord('c'), ord('C')):
                boxes.clear()
                points.clear()
                x0 = y0 = x1 = y1 = 0
            elif key in (ord('b'), ord('B')):
                mode = "box"
            elif key in (ord('p'), ord('P')):
                mode = "point"

        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass

        self.roi_boxes = boxes
        self.roi_points = points
        return f"Selected {len(self.roi_boxes)} boxes, {len(self.roi_points)} points"


    def clear_boxes(self):
        self.roi_boxes = []
        self.roi_points = []
        return "Cleared boxes and points"

    def update_dropdowns(self, input):
        if input == 'Low Quality':
            self.points_per_side = 32
            self.pred_iou_thresh = 0.4
            self.stability_score_thresh = 0.5
            self.crop_n_layers = 1
            self.crop_n_points_downscale_factor = 5
            self.min_mask_region_area = 50
        elif input == 'Medium Quality':
            self.points_per_side = 48
            self.pred_iou_thresh = 0.4
            self.stability_score_thresh = 0.5
            self.crop_n_layers = 1
            self.crop_n_points_downscale_factor = 5
            self.min_mask_region_area = 50
        else:
            self.points_per_side = 64
            self.pred_iou_thresh = 0.4
            self.stability_score_thresh = 0.5
            self.crop_n_layers = 1
            self.crop_n_points_downscale_factor = 5
            self.min_mask_region_area = 50

    def export_results(self):
        if not hasattr(self.pro_img, "particle_records"):
            raise gr.Error("Please run the recognition process at least once before exporting.")

        tmp_dir = os.path.join("temp_outputs")
        os.makedirs(tmp_dir, exist_ok=True)

        timestamp = int(time.time())
        zip_path = os.path.join(tmp_dir, f"results_{timestamp}.zip")

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(['Blob Number', f'Diameter ({self.pro_img.unit})', 'Center X (px)', 'Center Y (px)'])
            for rec in self.pro_img.particle_records:
                writer.writerow([rec['id'], rec['diameter'], rec['center_x'], rec['center_y']])
            zf.writestr("statistics.csv", csv_buffer.getvalue())

            img1 = Image.fromarray(self.pro_img.result_image)
            buf1 = io.BytesIO()
            img1.save(buf1, format="PNG")
            zf.writestr("overlay.png", buf1.getvalue())

            img2 = Image.fromarray(self.pro_img.result_image_analysis)
            buf2 = io.BytesIO()
            img2.save(buf2, format="PNG")
            zf.writestr("histogram.png", buf2.getvalue())

            if getattr(self.pro_img, "label_mask", None) is not None:
                mask_raw = self.pro_img.label_mask

                binary_mask = self.pro_img.get_binary_mask_original_size()
                if binary_mask is not None:
                    mask_img = Image.fromarray(binary_mask)
                    buf3 = io.BytesIO()
                    mask_img.save(buf3, format="TIFF")
                    zf.writestr("label_mask.tiff", buf3.getvalue())

                max_id = int(mask_raw.max())
                if max_id > 0:
                    mask_norm = (mask_raw.astype(np.float32) / max_id) * 255.0
                    mask_norm = mask_norm.astype(np.uint8)

                    mask_color = cv2.cvtColor(mask_norm, cv2.COLOR_GRAY2BGR)

                    for rec in self.pro_img.particle_records:
                        pid = rec["id"]
                        cx_orig = rec["center_x"]
                        cy_orig = rec["center_y"]

                        cx = int(round(cx_orig * self.pro_img.scale_factor))
                        cy = int(round(cy_orig * self.pro_img.scale_factor))

                        if 0 <= cx < mask_color.shape[1] and 0 <= cy < mask_color.shape[0]:
                            cv2.putText(
                                mask_color,
                                str(pid),
                                (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                (0, 0, 255),
                                1,
                                cv2.LINE_AA
                            )

                    preview_img = Image.fromarray(mask_color)
                    buf4 = io.BytesIO()
                    preview_img.save(buf4, format="PNG")
                    zf.writestr("label_mask_preview.png", buf4.getvalue())


        self.batch_last_zip = zip_path
        return zip_path

    def handle_batch(
        self,
        batch_files,
        img_distance,
        img_unit,
        px_length_input,
        fig_min_slider,
        fig_max_slider,
        fig_length,        
    ):

        if not batch_files or len(batch_files) == 0:
            raise gr.Error("Please upload batch images in the 'Batch Input Images' area.")

        tmp_dir = os.path.join("temp_outputs")
        os.makedirs(tmp_dir, exist_ok=True)
        timestamp = int(time.time())
        zip_path = os.path.join(tmp_dir, f"batch_results_{timestamp}.zip")

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in batch_files:
                path = f.name

                img_rgb, _, _ = load_with_metadata(path)

                if img_rgb is None:
                    pil_img = Image.open(path).convert("RGB")
                    img_rgb = np.array(pil_img)

                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                proc = ImgProcessing()
                proc.set_config(
                    self.points_per_side,
                    self.pred_iou_thresh,
                    self.stability_score_thresh,
                    self.crop_n_layers,
                    self.crop_n_points_downscale_factor,
                    self.min_mask_region_area,
                    img_distance,
                    img_unit,
                    False,
                    px_length_input,
                )
                proc.set_img(img_bgr)
                proc.set_boxes(self.roi_boxes)
                proc.set_points(self.roi_points)
                proc.show_img()

                if fig_max_slider is not None and fig_max_slider > 0:
                    proc.redraw(fig_min_slider, fig_max_slider, fig_length)
                base_name = os.path.splitext(os.path.basename(f.name))[0]

                csv_buffer = io.StringIO()
                writer = csv.writer(csv_buffer)
                writer.writerow([
                    'Blob Number',
                    f'Diameter ({proc.unit})',
                    'Center X (px)',
                    'Center Y (px)'
                ])
                for rec in proc.particle_records:
                    writer.writerow([rec['id'], rec['diameter'], rec['center_x'], rec['center_y']])
                zf.writestr(f"{base_name}/statistics.csv", csv_buffer.getvalue())

                img1 = Image.fromarray(proc.result_image)
                buf1 = io.BytesIO()
                img1.save(buf1, format="PNG")
                zf.writestr(f"{base_name}/overlay.png", buf1.getvalue())

                img2 = Image.fromarray(proc.result_image_analysis)
                buf2 = io.BytesIO()
                img2.save(buf2, format="PNG")
                zf.writestr(f"{base_name}/histogram.png", buf2.getvalue())
                if getattr(proc, "label_mask", None) is not None:
                    mask_raw = proc.label_mask

                    binary_mask = proc.get_binary_mask_original_size()
                    if binary_mask is not None:
                        mask_img = Image.fromarray(binary_mask)
                        buf3 = io.BytesIO()
                        mask_img.save(buf3, format="TIFF")
                        zf.writestr(f"{base_name}/label_mask.tiff", buf3.getvalue())

                    max_id = int(mask_raw.max())
                    if max_id > 0:
                        mask_norm = (mask_raw.astype(np.float32) / max_id) * 255.0
                        mask_norm = mask_norm.astype(np.uint8)
                        mask_color = cv2.cvtColor(mask_norm, cv2.COLOR_GRAY2BGR)

                        for rec in proc.particle_records:
                            pid = rec["id"]
                            cx_orig = rec["center_x"]
                            cy_orig = rec["center_y"]
                            cx = int(round(cx_orig * proc.scale_factor))
                            cy = int(round(cy_orig * proc.scale_factor))
                            if 0 <= cx < mask_color.shape[1] and 0 <= cy < mask_color.shape[0]:
                                cv2.putText(
                                    mask_color,
                                    str(pid),
                                    (cx, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4,
                                    (0, 0, 255),
                                    1,
                                    cv2.LINE_AA
                                )

                        preview_img = Image.fromarray(mask_color)
                        buf4 = io.BytesIO()
                        preview_img.save(buf4, format="PNG")
                        zf.writestr(f"{base_name}/label_mask_preview.png", buf4.getvalue())

        self.batch_last_zip = zip_path
        return zip_path
def load_with_metadata(path):

    _, ext = os.path.splitext(path)
    ext = ext.lower()

    img_rgb = None
    pixel_size = None
    unit = None

    try:
        if ext in ('.dm3', '.dm4'):
            dm = dm_read(path)
            pixel_size = round(float(
                dm[0]['original_metadata']["ImageList"]["TagGroup0"]["ImageData"]
                 ["Calibrations"]["Dimension"]["TagGroup0"]["Scale"]
            ), 4)
            unit = "nm"

            detector = dm[0]['original_metadata']["ImageList"]["TagGroup0"]["ImageTags"][
                "Microscope Info"
            ]["Illumination Mode"]

            img = dm[0]["data"]
            img = (img / img.max()) * 255
            img = cv2.normalize(img, None, 0, 255.0,
                                cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if detector != "TEM":
                img_rgb = cv2.bitwise_not(img_rgb)

        elif ext in ('.tif', '.tiff'):
            tiff = tiff_read(path)
            pixel_size = round(float(tiff[0]['axes'][0]['scale']) * 1e6, 3)
            unit = "um"

            img = tiff[0]["data"]
            img = (img / img.max()) * 255
            img = cv2.normalize(img, None, 0, 255.0,
                                cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        elif ext == '.emd':
            emd = emd_read(path)
            pixel_size = round(float(
                emd[0]['original_metadata']["BinaryResult"]['PixelSize']["width"]
            ) * 1e9, 3)
            unit = "nm"

            detector = emd[0]['original_metadata']['BinaryResult']['Detector']
            img = emd[0]["data"]
            img = cv2.normalize(img, None, 0, 255.0,
                                cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if detector != 'BM-Ceta':
                img_rgb = cv2.bitwise_not(img_rgb)

        elif ext == '.emi':
            tia = tia_read(path)
            pixel_size = round(float(
                tia[0]['original_metadata']['ser_header_parameters']['CalibrationDeltaX']
            ) * 1e9, 3)
            unit = "nm"

            img = tia[0]['data']
            img = (img / img.max()) * 255
            img = cv2.normalize(img, None, 0, 255.0,
                                cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img_rgb = cv2.bitwise_not(img_rgb)

    except Exception:
        img_rgb = None
        pixel_size = None
        unit = None

    return img_rgb, pixel_size, unit