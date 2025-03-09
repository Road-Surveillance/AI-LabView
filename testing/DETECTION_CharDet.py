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


def recognize_characters(lp_crop, char_detector):
    """
    Given a cropped license plate image and a character detection model,
    preprocess the image, run detection, and return the recognized string.
    """
    # Optional: Convert the license plate crop to grayscale if the char model requires it
    lp_crop_gray = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)
    # Convert back to BGR if the model expects 3 channels
    lp_crop_gray = cv2.cvtColor(lp_crop_gray, cv2.COLOR_GRAY2BGR)

    # Resize directly to 128x128 for the character detector
    lp_crop_resized = cv2.resize(lp_crop_gray, (128, 128))

    # Run character detection on the processed image
    char_detections, _ = char_detector.detect(lp_crop_resized)
    # Debug print (can be commented out)
    # print("Raw character detections:", char_detections)

    # Filter out low-confidence detections
    char_detections = [det for det in char_detections if float(det[1]) > 0.1]
    if not char_detections:
        # Optionally, you could display lp_crop_resized here for debugging
        return ""

    # Sort detections by the left x-coordinate of the bounding box
    char_detections.sort(key=lambda det: det[2][0])
    recognized_plate = "".join([det[0] for det in char_detections])
    return recognized_plate


def detect_and_export_from_path(image_path, obj_detector, char_detector=None):
    """
    For a given image path, this function:
      1. Runs detection using object.pt to detect vehicles and license plates.
      2. Ignores any detections labeled as "person."
      3. Overlays vehicle detection bounding boxes and labels on the image.
      4. For each detection labeled as a license plate, crops the region and saves it to the folder "LPs."
         If a character detector is provided, also runs character recognition on the license plate.
         The recognized characters are then overlayed on the output image.
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

        # Save based on the detection label:
        # If the detection is a license plate, process with character recognition.
        if label.lower() in ["square license plate", "rectangle license plate"]:
            filename = os.path.join(
                lp_dir,
                f"{os.path.splitext(os.path.basename(image_path))[0]}_LP_{idx}.jpg",
            )
            cv2.imwrite(filename, crop_img)
            print(f"Saved license plate crop to {filename}")

            if char_detector is not None:
                recognized_plate = recognize_characters(crop_img, char_detector)
                print(f"Recognized License Plate: {recognized_plate}")
                # Overlay recognized characters on the output image below the bounding box
                if recognized_plate:
                    cv2.putText(
                        output_img,
                        recognized_plate,
                        (x1, y2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
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
        default=["object.pt"],
        help="model paths for detection; include char.pt for character recognition",
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

    # Instantiate the object detector with object.pt for vehicles and license plates
    obj_detector = Detection(
        size=tuple(opt.imgsz),
        weights_path="object.pt",
        device=opt.device,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
    )

    # Optionally instantiate the character recognition detector if char.pt is provided in the weights argument
    char_detector = Detection(
        size=(128, 128),
        weights_path="char.pt",
        device=opt.device,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
    )

    # Process a single image or all images in a directory
    if os.path.isdir(opt.source):
        img_names = os.listdir(opt.source)
        for img_name in img_names:
            img_path = os.path.join(opt.source, img_name)
            detect_and_export_from_path(img_path, obj_detector, char_detector)
    else:
        detect_and_export_from_path(opt.source, obj_detector, char_detector)
