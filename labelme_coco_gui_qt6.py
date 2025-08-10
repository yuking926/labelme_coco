import os
import glob
import json
import cv2
import time
import shutil
import random
import string
import numpy as np
from typing import List, Dict, Any


# ---------- LabelMe -> COCO（零变化版） ----------
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
        x0, y0 = p0
        x1, y1 = p1
        xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
        ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)
        return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

    @staticmethod
    def _bbox_from_points(points: List[List[float]]) -> List[float]:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin, xmax = float(min(xs)), float(max(xs))
        ymin, ymax = float(min(ys)), float(max(ys))
        return [xmin, ymin, xmax - xmin, ymax - ymin]

    @staticmethod
    def _polygon_area(points: List[List[float]]) -> float:
        if len(points) < 3:
            return 0.0
        x = np.array([p[0] for p in points], dtype=np.float64)
        y = np.array([p[1] for p in points], dtype=np.float64)
        x2 = np.r_[x, x[0]]
        y2 = np.r_[y, y[0]]
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
        h = data.get("imageHeight")
        w = data.get("imageWidth")
        img_basename = None
        if isinstance(data.get("imagePath"), str) and data["imagePath"].strip():
            img_basename = os.path.basename(data["imagePath"].strip())
        if not img_basename:
            stem = os.path.splitext(os.path.basename(json_path))[0]
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
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
        image["height"] = int(h)
        image["width"] = int(w)
        image["file_name"] = img_basename if img_basename else f"{img_id:06d}.jpg"
        return image

    def _one_annotation(self, shape: Dict[str, Any], img_id: int) -> Dict[str, Any]:
        raw_label = shape.get("label", "")
        cat_name = raw_label
        cat_id = self._ensure_category(cat_name)

        shape_type = shape.get("shape_type", "polygon")
        pts = shape.get("points", [])
        score = shape.get("score", None)

        if shape_type == "rectangle":
            if len(pts) == 2:
                points = self._rect_to_polygon(pts[0], pts[1])
            elif len(pts) == 4:
                points = [[float(x), float(y)] for x, y in pts]
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
            ann["attributes"] = {"score": float(score)}
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


# ---------- 数据集自动生成 ----------
def gen_dataset_folder_name() -> str:
    t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    rand = "".join(random.choices(string.hexdigits.lower(), k=6))
    return f"{t}_{rand}"

def build_dataset(data_dir: str):
    data_dir = os.path.abspath(data_dir)
    parent_dir = os.path.dirname(data_dir)
    dataset_name = gen_dataset_folder_name()
    dataset_root = os.path.join(parent_dir, dataset_name)
    images_dir = os.path.join(dataset_root, "Images")
    ann_dir = os.path.join(dataset_root, "Annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    # 复制被 JSON 引用到的图片
    converter = LabelMe2COCO(json_dir=data_dir, img_dir=data_dir)
    for img_info in converter.convert()["images"]:
        src_img = os.path.join(data_dir, img_info["file_name"])
        if os.path.exists(src_img):
            shutil.copy2(src_img, os.path.join(images_dir, img_info["file_name"]))

    # 保存 COCO JSON
    coco_json_path = os.path.join(ann_dir, "coco_info.json")
    coco_data = {
        "info": {"description": "Converted from LabelMe", "version": "1.0"},
        "licenses": [],
        "images": converter.images,
        "annotations": converter.annotations,
        "categories": sorted(converter.categories, key=lambda c: c["id"]),
    }
    with open(coco_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)

    # 压缩
    zip_path = os.path.join(parent_dir, dataset_name + ".zip")
    shutil.make_archive(os.path.splitext(zip_path)[0], "zip", dataset_root)

    print(f"[OK] 数据集已生成：{dataset_root}")
    print(f"[OK] 压缩包路径：{zip_path}")


if __name__ == "__main__":
    # ======== 只需要改这一行路径 ========
    data_dir = "/home/yuking/Desktop/dist/formal6"
    build_dataset(data_dir)
