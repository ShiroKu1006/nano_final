import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import os
import requests
from datetime import datetime

# --- 參數設定 ---
CONF_THRESH = 0.5      
NMS_THRESH = 0.45      
TARGET_CLASS_ID = 0    
MODEL_INPUT_SIZE = (640, 640) 
DISCORD_API_URL = "http://localhost:8061/send"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# 載入 FP16 引擎
engine = load_engine("yolo11n_fp16.engine")
context = engine.create_execution_context()

# --- TensorRT 記憶體分配 ---
inputs, outputs, bindings = [], [], []
stream = cuda.Stream()

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = engine.get_tensor_shape(name)
    dtype = trt.nptype(engine.get_tensor_dtype(name))
    size = trt.volume(shape)
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(device_mem))
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        inputs.append({'host': host_mem, 'device': device_mem, 'name': name, 'shape': shape})
        context.set_input_shape(name, shape)
    else:
        outputs.append({'host': host_mem, 'device': device_mem, 'name': name, 'shape': shape})

def postprocess_yolo(raw_output, origin_w, origin_h):
    """
    執行 YOLO 輸出解析與 NMS
    """
    predictions = np.transpose(raw_output[0], (1, 0))
    boxes_np = predictions[:, 0:4] 
    scores_np = predictions[:, 4:] 
    class_ids = np.argmax(scores_np, axis=1)
    confidences = np.max(scores_np, axis=1)
    mask = (confidences > CONF_THRESH) & (class_ids == TARGET_CLASS_ID)
    
    filtered_boxes = boxes_np[mask]
    filtered_scores = confidences[mask]

    if len(filtered_scores) == 0:
        return [], 0.0

    x_factor = origin_w / MODEL_INPUT_SIZE[0]
    y_factor = origin_h / MODEL_INPUT_SIZE[1]
    final_boxes_list = []
    
    for i in range(len(filtered_boxes)):
        cx, cy, w, h = filtered_boxes[i]
        left = int((cx - w / 2) * x_factor)
        top = int((cy - h / 2) * y_factor)
        width = int(w * x_factor)
        height = int(h * y_factor)
        final_boxes_list.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(final_boxes_list, filtered_scores.tolist(), CONF_THRESH, NMS_THRESH)
    results = []
    max_score = 0.0
    if len(indices) > 0:
        for i in indices.flatten():
            results.append((final_boxes_list[i], filtered_scores[i]))
            max_score = max(max_score, filtered_scores[i])
    return results, max_score

def send_discord_alert(message, file_path=None):
    """
    呼叫 API 進行遠端通知
    """
    try:
        payload = {
            "message": message,
            "file_path": os.path.abspath(file_path) if file_path else None
        }
        requests.post(DISCORD_API_URL, json=payload, timeout=5)
    except Exception as e:
        print(f"API Call Error: {e}")

# --- 主迴圈初始化 ---
cap = cv2.VideoCapture(0)
is_recording = False
last_alert_time = 0    # 10 秒週期計時器
video_writer = None
last_seen_time = 0     
record_fps = 20.0
w, h = int(cap.get(3)), int(cap.get(4))
prev_frame_time = 0

def get_video_writer(filename, fps, width, height):
    """
    使用 Jetson 硬體加速編碼器
    """
    gst_str = (
        f"appsrc ! videoconvert ! video/x-raw, format=I420 ! "
        f"omxh264enc bitrate=4000000 ! h264parse ! qtmux ! "
        f"filesink location={filename}"
    )
    writer = cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, fps, (width, height))
    if not writer.isOpened():
        return cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    return writer



while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    t1 = time.time()
    fps_val = 1 / (t1 - prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = t1

    # 1. 影像預處理
    img_prep = cv2.resize(frame, MODEL_INPUT_SIZE)
    img_prep = img_prep.astype(np.float32) / 255.0
    img_prep = img_prep.transpose((2, 0, 1))
    data = np.ascontiguousarray(img_prep)

    # 2. 推論
    np.copyto(inputs[0]['host'], data.ravel())
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    for i in range(engine.num_io_tensors):
        context.set_tensor_address(engine.get_tensor_name(i), bindings[i])
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()

    # 3. 後處理
    raw_output = outputs[0]['host'].reshape(outputs[0]['shape'])
    detected_objects, max_person_score = postprocess_yolo(raw_output, w, h)

    intruder_detected = len(detected_objects) > 0
    current_time = time.time()

    if intruder_detected:
        last_seen_time = current_time 
        
        # 10 秒週期性警告邏輯
        if (current_time - last_alert_time) > 10.0:
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_name = f"alert_{ts_str}.jpg"
            cv2.imwrite(screenshot_name, frame)
            
            alert_msg = f"🚨 **持續偵測到入侵者！**\n時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n分數：{max_person_score:.2f}"
            send_discord_alert(alert_msg, screenshot_name)
            last_alert_time = current_time

        # 繪製結果
        for box, score in detected_objects:
            l, t, bw, bh = box
            cv2.rectangle(frame, (l, t), (l + bw, t + bh), (0, 255, 0), 2)
            cv2.putText(frame, f"Person: {score:.2f}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 4. 錄影邏輯與重置
    should_record = intruder_detected or (is_recording and (current_time - last_seen_time < 5.0))

    if should_record:
        if not is_recording:
            is_recording = True
            rec_start = current_time
            vid_name = f"intruder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            video_writer = get_video_writer(vid_name, record_fps, w, h)
        elif current_time - rec_start > 30:
            if video_writer: video_writer.release()
            rec_start = current_time
            vid_name = f"intruder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            video_writer = get_video_writer(vid_name, record_fps, w, h)

        if video_writer: video_writer.write(frame)
    else:
        if is_recording:
            is_recording = False
            last_alert_time = 0 # 人消失後歸零計時器
            if video_writer:
                video_writer.release()
                video_writer = None

    # UI 顯示
    cv2.putText(frame, f"FPS: {fps_val:.2f}", (w - 180, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

if video_writer: video_writer.release()
cap.release()
cv2.destroyAllWindows()