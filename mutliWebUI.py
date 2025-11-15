import gradio as gr
import os
import time
import zipfile
import io
import csv
from PIL import Image
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
    height: 60px;
    box-shadow: 0 15px 10px #000000;
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
        with gr.Blocks(css=css, title="天津大学化工学院大型仪器测试平台") as demo:
            with gr.Row(elem_classes='head-bar'):
                gr.Markdown(value=f"<div style='display: flex;justify-content: center;align-items: center;'>"
                                  f"<a style='position: absolute;left: 0;' href='https://www.clickgene.org/about/'>"
                                  f"<img style='height:60px;width:auto' src='{logo_img}'/></a>"
                                  f"<div style='font-size: 2.5rem;margin-left: 24px; color: #52c2f9;text-shadow: 1px 1px 2px black;'>"
                                  f"天津大学化工学院大型仪器测试平台"
                                  f"</div></div>")
            with gr.Row():
                with gr.Column():
                    gr.Markdown('# 输入图片')
                    self.input_img = gr.Image(elem_id="max-image", sources=['upload'], image_mode="RGBA")
                    gr.Markdown('## 参数设置')

                    self.process_speed = gr.Dropdown(["低质量", "中质量", "高质量"], label="处理质量", value='中质量')
                    self.points_per_side = 48
                    self.pred_iou_thresh = 0.4
                    self.stability_score_thresh = 0.5
                    self.crop_n_layers = 1
                    self.crop_n_points_downscale_factor = 5
                    self.min_mask_region_area = 50

                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                self.img_distance = gr.Number(value=0, label='比例尺物理长度')
                                self.img_unit = gr.Dropdown(["nm", "um", "mm", "cm"], label="比例尺单位",
                                                                     value='nm')
                                self.px_length_input = gr.Number(value=0, label='比例尺像素长度(px)')
                            with gr.Row():
                                self.open_auto_scale_info = gr.Checkbox(label="自动识别比例尺信息", value=True)
                    self.fig_length = gr.Number(value=10, label='表格步长')
                    self.roi_boxes = []
                    self.box_info = gr.Markdown('未选框')
                    self.select_box_btn = gr.Button("框选ROI（弹窗）")
                    self.upload_button = gr.Button("开始识别纳米粒子")
                    self.clear_box_btn = gr.Button("清空框（识别后再点击）")
                    self.process_speed.change(self.update_dropdowns, inputs=[self.process_speed])
                with gr.Column():
                    gr.Markdown('# 处理结果')
                    self.output = gr.Image(elem_id="output-image")
                    gr.Markdown('# 纳米粒径统计图表')
                    self.output_fig = gr.Image(elem_id="fig-image")
                    with gr.Row():
                        self.fig_min_slider = gr.Slider(label="最小直径", value=0)
                        self.fig_max_slider = gr.Slider(label="最大直径", value=0)
                    self.redraw_fig = gr.Button("更改直径")

                    self.output_csv = gr.Button("导出结果至本地")
                    self.download_zip = gr.File(
                        label="请点击右侧蓝色文本，下载当前结果",
                        elem_id="download-zip"
                    )                                        
            
            self.select_box_btn.click(self.select_boxes_popup, inputs=[], outputs=[self.box_info])
            self.clear_box_btn.click(self.clear_boxes, inputs=[], outputs=[self.box_info])
            self.footer = gr.HTML(footer_ele)
            self.upload_button.click(self.handle_img, inputs=[self.img_distance, self.img_unit
                                                              , self.open_auto_scale_info, self.px_length_input, 
                                                              ],
                                    outputs=[self.output, self.output_fig, self.fig_min_slider, self.fig_max_slider,
                                            self.img_distance, self.img_unit, self.px_length_input])
            self.redraw_fig.click(self.redraw,
                                  inputs=[self.fig_min_slider, self.fig_max_slider, self.fig_length],
                                  outputs=[self.output, self.output_fig])
            self.input_img.upload(self.fix_image, inputs=self.input_img, outputs=self.input_img)
            self.output_csv.click(self.export_results,inputs=[],outputs=[self.download_zip])
        demo.launch(show_error=True, server_name="127.0.0.1", server_port=7860)

    def fix_image(self, input_img):
        self.input_img_tif = input_img
        return input_img

    def handle_img(self,
                   img_distance,
                   img_unit,
                   open_auto_scale_info,
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
            open_auto_scale_info,
            px_length_input,
        )
        self.pro_img.set_img(self.input_img_tif)
        self.pro_img.set_boxes(self.roi_boxes)
        show_next = self.pro_img.show_img()
        res, res_fig = show_next

        unit = self.pro_img.unit
        scale_factor = self.pro_img.scale_factor
        max_text = "最小直径/" + unit
        min_text = "最大直径/" + unit
        self.fig_min_slider = gr.Slider(label=max_text, minimum=0, maximum=self.pro_img.max_d)
        self.fig_max_slider = gr.Slider(label=min_text, value=self.pro_img.max_d, minimum=0, maximum=self.pro_img.max_d)
        return res, res_fig, self.fig_min_slider, self.fig_max_slider, self.pro_img.distance, self.pro_img.unit, self.pro_img.px_length/scale_factor

    def redraw(self, fig_min_slider, fig_max_slider, fig_length):
        show_next = self.pro_img.redraw(fig_min_slider, fig_max_slider, fig_length)
        res, res_fig = show_next
        return res, res_fig
    def select_boxes_popup(self):
        if self.input_img_tif is None:
            raise gr.Error("请先上传图片再框选。")

        import cv2

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
        drawing = False
        x0 = y0 = x1 = y1 = 0

        def on_mouse(event, x, y, flags, param):
            nonlocal drawing, x0, y0, x1, y1
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                x0, y0 = x, y
                x1, y1 = x, y
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                x1, y1 = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                x1, y1 = x, y

        win = "ROI Selector (Enter: save box, Z: undo)"
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
            
            if drawing or (x0 != x1 and y0 != y1):
                cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)

            cv2.imshow(win, frame)
            key = cv2.waitKey(10) & 0xFF

            if key == 27:
                break
            elif key in (13, 10):
                xa, xb = sorted([x0, x1])
                ya, yb = sorted([y0, y1])
                if xb - xa > 2 and yb - ya > 2:
                    inv = 1.0 / scale
                    bx = [int(round(xa * inv)), int(round(ya * inv)),
                        int(round(xb * inv)), int(round(yb * inv))]
                    boxes.append(bx)
                
                x0 = y0 = x1 = y1 = 0
            elif key in (ord('z'), ord('Z'), 8):
                if boxes:
                    boxes.pop()
            elif key in (ord('c'), ord('C')):
                boxes.clear()
        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass
        self.roi_boxes = boxes
        return f"已选 {len(self.roi_boxes)} 个框"

    def clear_boxes(self):
        self.roi_boxes = []
        return "已清空框"
    def update_dropdowns(self, input):
        if input == '低质量':
            self.points_per_side = 32
            self.pred_iou_thresh = 0.4
            self.stability_score_thresh = 0.5
            self.crop_n_layers = 1
            self.crop_n_points_downscale_factor = 5
            self.min_mask_region_area = 50
        elif input == '中质量':
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
            raise gr.Error("请先运行一次识别流程，然后再导出。")

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

        return zip_path