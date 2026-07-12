from ultralytics import YOLO
from utils import get_device

model = YOLO('models/best.pt')

results = model.predict('input_videos/08fd33_4.mp4',save=True,device=get_device())
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)