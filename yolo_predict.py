import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from models.yolo import Model
from utils import check_img_size, letterbox, non_max_suppression, get_color

def load_model(weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(cfg="yolov8.cfg", ch=3, nc=80).to(device)
    checkpoint = torch.load(weights, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, device

def predict(image_dir, model, device, conf_thres=0.25, iou_thres=0.45):
    image_paths = list(Path(image_dir).glob("*.jpg"))
    results = []
    for image_path in image_paths:
        image = Image.open(image_path)
        img_size = check_img_size(image.size, s=640)
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(image)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        results.append((image_path, pred))
    return results

if __name__ == "__main__":
    weights = "runs/detect/train6/weights/best.pt"
    image_dir = "images"
    model, device = load_model(weights)
    results = predict(image_dir, model, device)
    for image_path, pred in results:
        print(f"Image: {image_path}")
        print(f"Predictions: {pred}")

# yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=images line_width=2
