import torch
from ultralytics import YOLO

def train_model():

    #model = YOLO('yolov8n-p2.yaml').load('yolov8n.pt')
    model = YOLO('runs/detect/train4/weights/last.pt')
    # 进行模型训练
    model.train(
	#从头开始训练
        data='data.yaml',
        epochs=50,
        device='mps',
        workers = 8,
        batch =2,
    )

    # 进行模型验证
    model.val()
if __name__ == "__main__":
    # 调用训练函数
    train_model()