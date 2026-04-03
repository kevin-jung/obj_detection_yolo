import cv2
import torch
from ultralytics import YOLO
import numpy as np

def pad_to_square(image, pad_color=(114,114,114)):
    h, w = image.shape[:2]
    size = max(h, w)  # target square size (e.g., 1280)

    # Compute padding
    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left

    # Apply padding
    squared = cv2.copyMakeBorder(
        image,
        pad_top, pad_bottom,
        pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=pad_color
    )

    return squared, size, (pad_left, pad_top)


def resize_to_stride(image, stride_mul, interpolation=cv2.INTER_LINEAR):
    """
    Resize image to (32 * stride_mul, 32 * stride_mul)
    """
    target = 32 * stride_mul
    resized = cv2.resize(image, (target, target), interpolation=interpolation)
    return resized, target


# 모델 로드 (GPU 지원)
model = YOLO('yolo26n.pt').to('cuda')  # yolov26n.pt는 사전 훈련된 모델 파일입니다.

# 웹캠 설정
# cap = cv2.VideoCapture(1)  # 0번 카메라 사용, 다른 카메라면 번호 변경
cap = cv2.VideoCapture("DJI_20251108145841_0106_D.mp4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()  # 프레임 읽기
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # 1) pad to square
    square, square_size, pad = pad_to_square(frame)

    # 2) resize to stride × multiplier
    resized_frame, final_size = resize_to_stride(square, stride_mul=50)  # 704×704

    # GPU로 이동
    frame_gpu = torch.from_numpy(resized_frame).to('cuda').permute(2, 0, 1).float() / 255.0
    frame_gpu = frame_gpu.unsqueeze(0)

    # 객체 인식 수행
    results = model(frame_gpu, imgsz=final_size, verbose=True)  # confidence threshold를 설정할 수 있습니다.

    # 결과 출력
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            r = box.xyxy[0].astype(int)
            cv2.rectangle(resized_frame, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = box.conf[0]
            cv2.putText(resized_frame, f'{label} {conf:.2f}', (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 결과 이미지 표시
    cv2.imshow('Object Detection', resized_frame)

    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):  # 'q' 키를 누르면 종료
        break

# 웹캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
