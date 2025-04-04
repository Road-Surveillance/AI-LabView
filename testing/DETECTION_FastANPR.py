import numpy as np
import torch
import os
import sys
import argparse
import shutil  # for removing directories
from typing import List
import cv2


sys.path.append(os.path.abspath("../yolov5"))
from utils.general import non_max_suppression, scale_coords
from models.experimental import attempt_load

# Add fastanpr import
from fastanpr import FastANPR
import asyncio  # add this import at the top


class Detection:
    def __init__(
        self,
        weights_path=".pt",
        size=(640, 640),
        device="cpu",
        iou_thres=None,
        conf_thres=None,
    ):
        self.device = device
        self.model, self.names = self.load_model(weights_path)
        self.size = size
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres

    def detect(self, frame):
        results, resized_img = self.yolo_detection(frame)
        return results, resized_img

    def preprocess_image(self, original_image):
        resized_img = self.ResizeImg(original_image, size=self.size)
        image = resized_img.copy()[:, :, ::-1].transpose(
            2, 0, 1
        )  # BGR to RGB, shape: 3 x H x W
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).to(self.device)
        image = image.float() / 255.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)
        return image, resized_img

    def yolo_detection(self, image, classes=None, agnostic_nms=True, max_det=1000):
        img, resized_img = self.preprocess_image(image.copy())
        pred = self.model(img, augment=False)[0]
        detections = non_max_suppression(
            pred,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            classes=classes,
            agnostic=agnostic_nms,
            multi_label=True,
            labels=(),
            max_det=max_det,
        )
        results = []
        for det in detections:
            det = det.tolist()
            if len(det):
                for *xyxy, conf, cls in det:
                    result = [
                        self.names[int(cls)],
                        str(conf),
                        (xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    ]
                    results.append(result)
        return results, resized_img

    def ResizeImg(self, img, size):
        h1, w1, _ = img.shape
        h, w = size
        if w1 < h1 * (w / h):
            img_rs = cv2.resize(img, (int(float(w1 / h1) * h), h))
            mask = np.zeros((h, w - int(float(w1 / h1) * h), 3), np.uint8)
            img = cv2.hconcat([img_rs, mask])
            trans_x = int(w / 2) - int(int(float(w1 / h1) * h) / 2)
            trans_y = 0
            trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
            height, width = img.shape[:2]
            img = cv2.warpAffine(img, trans_m, (width, height))
            return img
        else:
            img_rs = cv2.resize(img, (w, int(float(h1 / w1) * w)))
            mask = np.zeros((h - int(float(h1 / w1) * w), w, 3), np.uint8)
            img = cv2.vconcat([img_rs, mask])
            trans_x = 0
            trans_y = int(h / 2) - int(int(float(h1 / w1) * w) / 2)
            trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
            height, width = img.shape[:2]
            img = cv2.warpAffine(img, trans_m, (width, height))
            return img

    def load_model(self, path, train=False):
        model = attempt_load(path, map_location=self.device)
        names = model.module.names if hasattr(model, "module") else model.names
        model.train() if train else model.eval()
        return model, names

    def xyxytoxywh(self, x):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[0] = (x[0] + x[2]) / 2  # center x
        y[1] = (x[1] + x[3]) / 2  # center y
        y[2] = x[2] - x[0]  # width
        y[3] = x[3] - x[1]  # height
        return y


def detect_and_export_from_path(image_path, obj_detector, fast_anpr=None):
    """
    For a given image path, this function:
      1. Runs detection using object.pt to detect vehicles and license plates.
      2. Ignores any detections labeled as "person."
      3. Overlays vehicle detection bounding boxes and labels on the image.
      4. For each detection labeled as a license plate, crops the region, runs recognition using fastanpr,
         overlays the recognized text on the detection, and saves it to the folder "LPs."
      5. For all other detections (vehicle types), crops the region and saves it to the folder "Vehicle."
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return

    detections, resized_img = obj_detector.detect(image.copy())
    output_img = resized_img.copy()

    # Ensure output directories exist (remove if they exist)
    out_dir = "out"
    lp_dir = "LPs"
    vehicle_dir = "Vehicle"

    for folder in [lp_dir, vehicle_dir]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    os.makedirs(out_dir, exist_ok=True)

    for idx, det in enumerate(detections):
        label, conf, box = det
        # Ignore person detections
        if label.lower() == "person":
            continue

        x1, y1, x2, y2 = map(int, box)

        # Draw bounding box and label on output image
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            output_img,
            f"{label} {float(conf):.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )

        # Crop the detected region
        crop_img = resized_img[y1:y2, x1:x2].copy()

        if label.lower() in ["square license plate", "rectangle license plate"]:
            recognized_text = "N/A"
            if fast_anpr is not None:
                # Use asyncio.run to await the coroutine returned by fast_anpr.run
                results = asyncio.run(fast_anpr.run([crop_img]))
                if results and len(results[0]) > 0:
                    recognized_text = results[0][0].rec_text
            # Overlay recognized text below the bounding box
            cv2.putText(
                output_img,
                recognized_text,
                (x1, y2 + 30),  # position below the box
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            filename = os.path.join(
                lp_dir,
                f"{os.path.splitext(os.path.basename(image_path))[0]}_LP_{idx}.jpg",
            )
            cv2.imwrite(filename, crop_img)
            print(
                f"Saved license plate crop with recognized text '{recognized_text}' to {filename}"
            )
        else:
            # Otherwise, treat it as a vehicle detection and save to Vehicle folder.
            filename = os.path.join(
                vehicle_dir,
                f"{os.path.splitext(os.path.basename(image_path))[0]}_Vehicle_{idx}.jpg",
            )
            cv2.imwrite(filename, crop_img)
            print(f"Saved vehicle crop to {filename}")

    # Save and display the final annotated image
    out_path = os.path.join(out_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, output_img)
    cv2.imshow("Vehicle & License Plate Detection", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default="object.pt",
        help="model path for object detection",
    )
    parser.add_argument(
        "--source", type=str, default="testing_imgs", help="image path or directory"
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[1280],
        help="inference size for object detection",
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.1, help="confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.5, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--max-det", type=int, default=1000, help="maximum detections per image"
    )
    parser.add_argument("--device", default="cpu", help="cuda device, e.g. 0 or cpu")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand if needed
    return opt


if __name__ == "__main__":
    opt = parse_opt()

    # Instantiate the object detector (using object.pt and larger image size)
    obj_detector = Detection(
        size=tuple(opt.imgsz),
        weights_path="object.pt",
        device=opt.device,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
    )

    # Instantiate FastANPR recognizer for license plate OCR (compatible with Python 3.9)
    fast_anpr = FastANPR(device=opt.device)

    # Process a single image or all images in a directory, passing the FastANPR object
    if os.path.isdir(opt.source):
        img_names = os.listdir(opt.source)
        for img_name in img_names:
            img_path = os.path.join(opt.source, img_name)
            detect_and_export_from_path(img_path, obj_detector, fast_anpr)
    else:
        detect_and_export_from_path(opt.source, obj_detector, fast_anpr)
