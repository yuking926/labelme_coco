#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt6 图形界面：LabelMe ⇄ COCO 双向转换 + 一键生成打包数据集（Images + Annotations + zip）

功能：
1) 目录(包含 LabelMe JSON + 图片)  →  自动生成
   - 以“时间+随机”命名的数据集目录
   - 其中包含 Images/ (仅复制被 JSON 引用的图) 和 Annotations/coco_info.json
   - 同级目录生成对应 .zip 压缩包

2) COCO → LabelMe：
   - 输入 COCO 标注文件(coco_info.json) 与 Images 目录
   - 输出到目标目录：为每张图生成同名的 LabelMe JSON（图和 JSON 放在同一文件夹）

实现细节：
- LabelMe→COCO 采用“零变化”策略：保留原始 points 与顺序（rectangle 四点不重排、不裁剪）；
  仅当 rectangle 为 2点 表示时，补成 4 点。
- COCO→LabelMe：
  - segmentation(list) → polygon
  - 无 segmentation 但有 bbox → rectangle（两点对角表示）
- 线程化执行，避免阻塞 UI；进度条、日志输出；异常捕获。

依赖：PyQt6, numpy, opencv-python
"""

import os
import sys
import json
import glob
import cv2
import time
import shutil
import random
import string
import traceback
import numpy as np
from typing import List, Dict, Any

from PyQt6 import QtCore, QtGui, QtWidgets

# --------------------------- 通用工具 ---------------------------
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

def ts_rand_name(prefix: str = "") -> str:
    t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    rand = "".join(random.choices(string.hexdigits.lower(), k=6))
    return f"{prefix + '_' if prefix else ''}{t}_{rand}"

# ---------------------- LabelMe → COCO（零变化） ----------------------
class LabelMe2COCO:
    def __init__(self, json_dir: str, img_dir: str):
        self.json_dir = json_dir
        self.img_dir = img_dir
        self.labelme_json_files = sorted(glob.glob(os.path.join(self.json_dir, "*.json")))
        if not self.labelme_json_files:
            raise ValueError(f"No JSON files found in: {self.json_dir}")
        self.images: List[Dict[str, Any]] = []
        self.annotations: List[Dict[str, Any]] = []
        self.categories: List[Dict[str, Any]] = []
        self.category_map: Dict[str, int] = {}
        self.ann_id = 1

    @staticmethod
    def _rect_to_polygon(p0: List[float], p1: List[float]) -> List[List[float]]:
        x0, y0 = p0; x1, y1 = p1
        xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
        ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)
        return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

    @staticmethod
    def _bbox_from_points(points: List[List[float]]) -> List[float]]:
        xs = [p[0] for p in points]; ys = [p[1] for p in points]
        xmin, xmax = float(min(xs)), float(max(xs))
        ymin, ymax = float(min(ys)), float(max(ys))
        return [xmin, ymin, xmax - xmin, ymax - ymin]

    @staticmethod
    def _polygon_area(points: List[List[float]]) -> float:
        if len(points) < 3:
            return 0.0
        x = np.array([p[0] for p in points], dtype=np.float64)
        y = np.array([p[1] for p in points], dtype=np.float64)
        x2 = np.r_[x, x[0]]; y2 = np.r_[y, y[0]]
        return float(0.5 * abs(np.dot(x2[:-1], y2[1:]) - np.dot(y2[:-1], x2[1:])))

    @staticmethod
    def _flatten_points(points: List[List[float]]) -> List[float]:
        out = []
        for (x, y) in points:
            out.extend([float(x), float(y)])
        return out

    def _ensure_category(self, name: str) -> int:
        if name not in self.category_map:
            new_id = len(self.category_map) + 1
            self.category_map[name] = new_id
            self.categories.append({"id": new_id, "name": name, "supercategory": ""})
        return self.category_map[name]

    def _resolve_image_meta(self, data: Dict[str, Any], json_path: str, img_id: int) -> Dict[str, Any]:
        image = {"id": img_id}
        h = data.get("imageHeight"); w = data.get("imageWidth")
        img_basename = None
        if isinstance(data.get("imagePath"), str) and data["imagePath"].strip():
            img_basename = os.path.basename(data["imagePath"].strip())
        if not img_basename:
            stem = os.path.splitext(os.path.basename(json_path))[0]
            for ext in IMAGE_EXTS:
                cand = os.path.join(self.img_dir, stem + ext)
                if os.path.exists(cand):
                    img_basename = os.path.basename(cand)
                    break
        if (h is None or w is None) and img_basename:
            img_path = os.path.join(self.img_dir, img_basename)
            if os.path.exists(img_path):
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise FileNotFoundError(f"Failed to read image: {img_path}")
                h, w = int(img.shape[0]), int(img.shape[1])
        if h is None or w is None:
            raise ValueError(f"Image size not available for JSON: {json_path}")
        image["height"], image["width"], image["file_name"] = int(h), int(w), (img_basename or f"{img_id:06d}.jpg")
        return image

    def _one_annotation(self, shape: Dict[str, Any], img_id: int) -> Dict[str, Any]:
        raw_label = shape.get("label", "")
        cat_id = self._ensure_category(raw_label)
        shape_type = shape.get("shape_type", "polygon")
        pts = shape.get("points", [])
        score = shape.get("score", None)

        if shape_type == "rectangle":
            if len(pts) == 2:
                points = self._rect_to_polygon(pts[0], pts[1])
            elif len(pts) == 4:
                points = [[float(x), float(y)] for x, y in pts]  # 保留原顺序与数值
            else:
                return None
        else:
            if len(pts) < 3:
                return None
            points = [[float(x), float(y)] for x, y in pts]

        bbox = self._bbox_from_points(points)
        area = self._polygon_area(points)
        if area <= 0 or bbox[2] <= 0 or bbox[3] <= 0:
            return None

        seg = [self._flatten_points(points)]
        ann = {
            "id": self.ann_id,
            "image_id": img_id,
            "category_id": cat_id,
            "iscrowd": 0,
            "bbox": bbox,
            "area": area,
            "segmentation": seg,
        }
        if score is not None:
            try:
                ann["attributes"] = {"score": float(score)}
            except Exception:
                pass
        self.ann_id += 1
        return ann

    def convert(self) -> Dict[str, Any]:
        img_id = 1
        for json_path in self.labelme_json_files:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            image_rec = self._resolve_image_meta(data, json_path, img_id)
            self.images.append(image_rec)
            for shape in data.get("shapes", []):
                ann = self._one_annotation(shape, img_id)
                if ann is not None:
                    self.annotations.append(ann)
            img_id += 1
        coco = {
            "info": {"description": "Converted from LabelMe", "version": "1.0"},
            "licenses": [],
            "images": self.images,
            "annotations": self.annotations,
            "categories": sorted(self.categories, key=lambda c: c["id"]),
        }
        return coco

# ---------------------- COCO → LabelMe ----------------------
def coco_to_labelme(coco_json_path: str, images_dir: str, output_dir: str, write_imageData: bool = False):
    with open(coco_json_path, 'r', encoding='utf-8') as fp:
        coco_data = json.load(fp)
    categories = {c['id']: c['name'] for c in coco_data.get('categories', [])}
    annotations = coco_data.get('annotations', [])
    images = coco_data.get('images', [])

    os.makedirs(output_dir, exist_ok=True)
    # 按 image_id 归组
    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for ann in annotations:
        anns_by_img.setdefault(ann['image_id'], []).append(ann)

    for img in images:
        img_id = img['id']
        file_name = img['file_name']
        W = img.get('width', 0)
        H = img.get('height', 0)
        labelme = {
            "version": "5.3.1",
            "flags": {},
            "shapes": [],
            "imagePath": file_name,
            "imageData": None,
            "imageHeight": H,
            "imageWidth": W
        }
        for ann in anns_by_img.get(img_id, []):
            shape = {"label": categories.get(ann['category_id'], 'unknown'),
                     "flags": {}, "group_id": None, "shape_type": "polygon", "points": []}
            seg = ann.get('segmentation')
            if isinstance(seg, list) and len(seg) > 0:
                flat = seg[0]
                pts = [[float(flat[i]), float(flat[i+1])] for i in range(0, len(flat), 2)]
                shape["shape_type"] = "polygon"
                shape["points"] = pts
            else:
                x, y, w, h = ann.get('bbox', [0, 0, 0, 0])
                p0 = [x, y]; p1 = [x + w, y + h]
                shape["shape_type"] = "rectangle"
                shape["points"] = [p0, p1]
            labelme["shapes"].append(shape)

        out_json = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".json")
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(labelme, f, ensure_ascii=False, indent=2)

        # 复制图片（如在不同目录）
        src_img = os.path.join(images_dir, file_name)
        dst_img = os.path.join(output_dir, file_name)
        if os.path.exists(src_img) and os.path.abspath(src_img) != os.path.abspath(dst_img):
            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
            shutil.copy2(src_img, dst_img)

# --------------------------- 压缩 ---------------------------
def zip_folder(folder_path: str, zip_path_no_ext: str):
    base_dir = os.path.basename(folder_path)
    root_dir = os.path.dirname(folder_path)
    shutil.make_archive(zip_path_no_ext, 'zip', root_dir=root_dir, base_dir=base_dir)

# --------------------------- 线程 Worker ---------------------------
class Worker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)            # 0-100
    log = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(bool, str)          # success, message/path

    def __init__(self, mode: str, params: dict, parent=None):
        super().__init__(parent)
        self.mode = mode  # 'LM2COCO_PACK' or 'COCO2LM'
        self.params = params

    def run(self):
        try:
            if self.mode == 'LM2COCO_PACK':
                self._run_lm2coco_pack()
            elif self.mode == 'COCO2LM':
                self._run_coco2lm()
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
        except Exception as e:
            self.done.emit(False, f"错误: {e}\n{traceback.format_exc()}")

    def _run_lm2coco_pack(self):
        data_dir = self.params['data_dir']
        prefix = self.params.get('prefix', '')
        data_dir = os.path.abspath(data_dir)
        parent_dir = os.path.dirname(data_dir)
        dataset_name = ts_rand_name(prefix)
        dataset_root = os.path.join(parent_dir, dataset_name)
        images_dir = os.path.join(dataset_root, 'Images')
        ann_dir = os.path.join(dataset_root, 'Annotations')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)

        self.log.emit(f"读取 LabelMe 标注: {data_dir}")
        conv = LabelMe2COCO(json_dir=data_dir, img_dir=data_dir)
        coco = conv.convert()

        # 复制被 JSON 引用到的图片
        total = len(coco['images'])
        for i, im in enumerate(coco['images'], 1):
            src_img = os.path.join(data_dir, im['file_name'])
            if os.path.exists(src_img):
                shutil.copy2(src_img, os.path.join(images_dir, im['file_name']))
            self.progress.emit(int(i / max(1, total) * 80))

        # 写 COCO JSON
        out_json = os.path.join(ann_dir, 'coco_info.json')
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(coco, f, ensure_ascii=False, indent=2)
        self.log.emit(f"已生成: {out_json}")

        # 打包 zip
        zip_no_ext = os.path.join(parent_dir, dataset_name)
        zip_folder(dataset_root, zip_no_ext)
        zip_path = zip_no_ext + '.zip'
        self.progress.emit(100)
        self.done.emit(True, f"数据集目录: {dataset_root}\n压缩包: {zip_path}")

    def _run_coco2lm(self):
        coco_json = self.params['coco_json']
        images_dir = self.params['images_dir']
        out_dir = self.params['out_dir']
        self.log.emit(f"COCO→LabelMe: {coco_json}\nImages: {images_dir}\n输出: {out_dir}")
        coco_to_labelme(coco_json, images_dir, out_dir)
        self.progress.emit(100)
        self.done.emit(True, f"已输出到: {out_dir}")

# --------------------------- 主界面 ---------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LabelMe ⇄ COCO 转换工具")
        self.setMinimumSize(820, 560)
        self._build_ui()
        self.worker: Worker = None

    def _build_ui(self):
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_tab_lm2coco(), "LabelMe→COCO(打包)")
        tabs.addTab(self._build_tab_coco2lm(), "COCO→LabelMe")
        self.setCentralWidget(tabs)

    # --- Tab1: LabelMe→COCO 打包 ---
    def _build_tab_lm2coco(self):
        w = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(w)

        # 行1：数据目录 & 选择
        row1 = QtWidgets.QHBoxLayout()
        self.ed_data_dir = QtWidgets.QLineEdit()
        btn_browse1 = QtWidgets.QPushButton("选择数据目录(含JSON与图片)")
        btn_browse1.clicked.connect(self.pick_data_dir)
        row1.addWidget(self.ed_data_dir); row1.addWidget(btn_browse1)

        # 行2：可选前缀
        row2 = QtWidgets.QHBoxLayout()
        self.ed_prefix = QtWidgets.QLineEdit()
        self.ed_prefix.setPlaceholderText("可选：目录名前缀，如 MySet")
        row2.addWidget(QtWidgets.QLabel("目录名前缀：")); row2.addWidget(self.ed_prefix)

        # 行3：开始
        row3 = QtWidgets.QHBoxLayout()
        self.btn_start_pack = QtWidgets.QPushButton("开始一键打包")
        self.btn_start_pack.clicked.connect(self.start_pack)
        row3.addStretch(1); row3.addWidget(self.btn_start_pack)

        # 行4：进度 & 日志
        self.pb1 = QtWidgets.QProgressBar(); self.pb1.setValue(0)
        self.log1 = QtWidgets.QPlainTextEdit(); self.log1.setReadOnly(True)

        lay.addLayout(row1)
        lay.addLayout(row2)
        lay.addLayout(row3)
        lay.addWidget(self.pb1)
        lay.addWidget(self.log1)
        return w

    # --- Tab2: COCO→LabelMe ---
    def _build_tab_coco2lm(self):
        w = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(w)

        # 行1：COCO JSON
        row1 = QtWidgets.QHBoxLayout()
        self.ed_coco_json = QtWidgets.QLineEdit()
        btn_coco = QtWidgets.QPushButton("选择 coco_info.json")
        btn_coco.clicked.connect(self.pick_coco_json)
        row1.addWidget(self.ed_coco_json); row1.addWidget(btn_coco)

        # 行2：Images 目录
        row2 = QtWidgets.QHBoxLayout()
        self.ed_images_dir = QtWidgets.QLineEdit()
        btn_imgdir = QtWidgets.QPushButton("选择 Images 目录")
        btn_imgdir.clicked.connect(self.pick_images_dir)
        row2.addWidget(self.ed_images_dir); row2.addWidget(btn_imgdir)

        # 行3：输出目录
        row3 = QtWidgets.QHBoxLayout()
        self.ed_out_dir = QtWidgets.QLineEdit()
        btn_out = QtWidgets.QPushButton("选择输出目录")
        btn_out.clicked.connect(self.pick_out_dir)
        row3.addWidget(self.ed_out_dir); row3.addWidget(btn_out)

        # 行4：开始
        row4 = QtWidgets.QHBoxLayout()
        self.btn_start_c2l = QtWidgets.QPushButton("开始转换")
        self.btn_start_c2l.clicked.connect(self.start_coco2lm)
        row4.addStretch(1); row4.addWidget(self.btn_start_c2l)

        # 进度 & 日志
        self.pb2 = QtWidgets.QProgressBar(); self.pb2.setValue(0)
        self.log2 = QtWidgets.QPlainTextEdit(); self.log2.setReadOnly(True)

        lay.addLayout(row1)
        lay.addLayout(row2)
        lay.addLayout(row3)
        lay.addLayout(row4)
        lay.addWidget(self.pb2)
        lay.addWidget(self.log2)
        return w

    # --------- 事件 ---------
    def pick_data_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择数据目录")
        if path:
            self.ed_data_dir.setText(path)

    def pick_coco_json(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择 coco_info.json", filter="JSON (*.json)")
        if path:
            self.ed_coco_json.setText(path)

    def pick_images_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择 Images 目录")
        if path:
            self.ed_images_dir.setText(path)

    def pick_out_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.ed_out_dir.setText(path)

    def start_pack(self):
        data_dir = self.ed_data_dir.text().strip()
        if not data_dir or not os.path.isdir(data_dir):
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择包含 JSON 与图片的目录")
            return
        params = {"data_dir": data_dir, "prefix": self.ed_prefix.text().strip()}
        self._run_worker('LM2COCO_PACK', params, self.pb1, self.log1, self.btn_start_pack)

    def start_coco2lm(self):
        coco_json = self.ed_coco_json.text().strip()
        images_dir = self.ed_images_dir.text().strip()
        out_dir = self.ed_out_dir.text().strip()
        if not os.path.isfile(coco_json):
            QtWidgets.QMessageBox.warning(self, "提示", "请选择有效的 coco_info.json 文件")
            return
        if not os.path.isdir(images_dir):
            QtWidgets.QMessageBox.warning(self, "提示", "请选择有效的 Images 目录")
            return
        if not out_dir:
            QtWidgets.QMessageBox.warning(self, "提示", "请选择输出目录")
            return
        params = {"coco_json": coco_json, "images_dir": images_dir, "out_dir": out_dir}
        self._run_worker('COCO2LM', params, self.pb2, self.log2, self.btn_start_c2l)

    def _run_worker(self, mode: str, params: dict,
                    pb: QtWidgets.QProgressBar,
                    log: QtWidgets.QPlainTextEdit,
                    btn: QtWidgets.QPushButton):
        if hasattr(self, "worker") and self.worker and self.worker.isRunning():
            QtWidgets.QMessageBox.information(self, "提示", "任务正在进行中…")
            return
        self.worker = Worker(mode, params)
        btn.setEnabled(False)
        pb.setValue(0)
        log.clear()
        self.worker.progress.connect(pb.setValue)
        self.worker.log.connect(lambda s: log.appendPlainText(s))

        def on_done(ok: bool, msg: str):
            btn.setEnabled(True)
            if ok:
                pb.setValue(100)
                log.appendPlainText(msg)
                QtWidgets.QMessageBox.information(self, "完成", msg)
            else:
                log.appendPlainText(msg)
                QtWidgets.QMessageBox.critical(self, "失败", msg)

        self.worker.done.connect(on_done)
        self.worker.start()

# --------------------------- 入口 ---------------------------
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec())
