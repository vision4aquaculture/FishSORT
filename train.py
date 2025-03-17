import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO, RTDETR

if __name__ == '__main__':
    from ultralytics.nn.tasks import RTDETRDetectionModel

    # model = RTDETRDetectionModel(cfg=r'ultralytics\cfg\models\rt-detr\rtdetr-r18.yaml', nc=1, verbose=True)
    model = RTDETR(r'ultralytics\cfg\models\rt-detr\rtdetr-r18.yaml')
    # model = RTDETR(r'ultralytics\cfg\models\rt-detr\rtdetr-r50.yaml')
    # # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/home/shengcli/my/ultralytics-main/dataset/data.yaml',
                cache=True,
                imgsz=640,
                epochs=150,
                batch=16,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )
