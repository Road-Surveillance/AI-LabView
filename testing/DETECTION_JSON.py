import numpy as np
import torch
import os
import sys
import argparse
import shutil  # for removing directories
import json  # for JSON output
from typing import List
import cv2
import argparse
import tensorflow as tf
from PIL import Image


os.chdir(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.abspath("../yolov5"))
from utils.general import non_max_suppression, scale_coords
from models.experimental import attempt_load

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
        )  # BGR -> RGB, shape: 3 x H x W
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

color_model = tf.keras.models.load_model("../Car_Color_Detection.keras")

class_labels = ["beige", "black", "blue", "brown", "gold", "green", "grey",
                "orange", "pink", "purple", "red", "silver", "tan", "white", "yellow"]


def detect_car_color(cropped_image_path):
    try:
        img = Image.open(cropped_image_path)
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = color_model.predict(img_array)
        predicted_class = np.argmax(predictions)

        return class_labels[predicted_class] if predicted_class < len(class_labels) else "Unknown"
    except Exception as e:
        print(f"Error detecting color: {e}")
        return "Unknown"
    


def detect_and_export_json(image_path,weights_path="object.pt", img_size=1280, device="cpu", conf_thres=0.1,iou_thres=0.5):
    """
    Processes an image (from its path) to run detections,
    saves cropped regions, and returns a JSON string containing detection details.
    """
    # Create output directories if they don't exist
    os.makedirs("out", exist_ok=True)
    os.makedirs("LPs", exist_ok=True)
    os.makedirs("Vehicle", exist_ok=True)

    # Initialize detector with the provided parameters
    detector = Detection(
        size=(img_size, img_size),
        weights_path=weights_path,
        device=device,
        iou_thres=iou_thres,
        conf_thres=conf_thres,
    )


    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        return json.dumps({"error": f"Could not load image: {image_path}"})

    # Run detection
    detections, resized_img = detector.detect(image.copy())
    output_img = resized_img.copy()

    results_list = []  # List to store detection details

    for idx, det in enumerate(detections):
        label, conf, box = det
        if label.lower() == "person":
            continue

        x1, y1, x2, y2 = map(int, box)
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

        crop_img = resized_img[y1:y2, x1:x2].copy()
        if label.lower() in ["square license plate", "rectangle license plate"]:
            crop_folder = "LPs"
            det_type = "license_plate"
        else:
            crop_folder = "Vehicle"
            det_type = "vehicle"

        crop_filename = os.path.join(
            crop_folder,
            f"{os.path.splitext(os.path.basename(image_path))[0]}{det_type}{idx}.jpg",
        )
        cv2.imwrite(crop_filename, crop_img)
        print(f"Saved {det_type} crop to {crop_filename}")
        detected_color = detect_car_color(crop_filename) if det_type == "vehicle" else None

        results_list.append(
            {
                "label": label,
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2],
                "crop_path": crop_filename,
                "detection_type": det_type,
                "vehicle_color": detected_color,
            }
        )

    annotated_path = os.path.join("out", os.path.basename(image_path))
    cv2.imwrite(annotated_path, output_img)

    result_dict = {"annotated_image": annotated_path, "detections": results_list}
    return json.dumps(result_dict, indent=2)


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
    import argparse

    parser = argparse.ArgumentParser(description="Run object detection on an image.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the image for detection"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="object.pt",
        help="Model weights file path",
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280, help="Image size for inference"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Computation device (cpu/cuda)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.1, help="Confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.5, help="NMS IoU threshold"
    )

    args = parser.parse_args()

    # Initialisation du détecteur
    obj_detector = Detection(
        size=(args.imgsz, args.imgsz),
        weights_path=args.weights,
        device=args.device,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
    )

    # Exécuter la détection sur l'image
    json_output = detect_and_export_json(args.image)
    print(json_output)

    # Sauvegarder le JSON
    json_filename = f"json_out/{os.path.splitext(os.path.basename(args.image))[0]}.json"
    os.makedirs("json_out", exist_ok=True)
    with open(json_filename, "w") as jf:
        jf.write(json_output)

    print(f"Saved JSON output to {json_filename}")

