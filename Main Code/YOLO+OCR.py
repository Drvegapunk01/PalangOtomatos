import cv2
import torch
import numpy as np
from pathlib import Path
import pathlib
import pytesseract
import time
import warnings
import os

warnings.filterwarnings("ignore")

# Hilangkan semua warning yang tidak perlu
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;panic"
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["GST_DEBUG"] = "0"

# Konfigurasi path Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'E:\Project\Deteksi_Tulisan\ML\yolov5\Tesseract-OCR\tesseract.exe'

# Fix Windows path issue
pathlib.PosixPath = pathlib.WindowsPath

# Path ke folder YOLOv5
YOLOV5_PATH = r"E:\Project\Deteksi_Tulisan\ML\yolov5"
MODEL_PATH = r"E:\Project\Deteksi_Tulisan\ML\yolov5\best1.pt"

# Load model custom
try:
    model = torch.hub.load(YOLOV5_PATH, 'custom', path=MODEL_PATH, source='local')
    print("Model YOLOv5 berhasil dimuat")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Konfigurasi Tesseract
custom_config = r'--oem 3 --psm 6'

# Fungsi untuk menginisialisasi koneksi RTSP
def initialize_rtsp_connection():
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        try:
            rtsp_url = "rtsp://admin:rendani123@192.168.1.64:5548/streaming/channels/101"
            
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)
            
            # Set resolusi langsung ke 640x640
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
            time.sleep(0.5)
            
            ret, frame = cap.read()
            if ret and frame is not None:
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"RTSP berhasil - Resolusi: {actual_width}x{actual_height}")
                return cap
            
            cap.release()
            attempt += 1
            print(f"Percobaan {attempt}/{max_attempts} gagal, mencoba lagi...")
            time.sleep(2)
            
        except Exception as e:
            print(f"Error: {e}")
            attempt += 1
            time.sleep(2)
    
    print("Tidak dapat terhubung ke RTSP setelah beberapa percobaan")
    return None

# Inisialisasi koneksi RTSP
print("Menghubungkan ke CCTV RTSP...")
cap = initialize_rtsp_connection()
if cap is None:
    print("Tidak dapat mengakses stream RTSP")
    exit(1)

# Dapatkan resolusi aktual
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolusi CCTV: {actual_width}x{actual_height}")

# Variabel untuk melacak status koneksi
connection_retries = 0
max_retries = 5
frame_count = 0

print("Memulai proses deteksi dengan resolusi 640x640...")

while True:
    try:
        # Baca frame
        ret, frame = cap.read()
        
        if not ret:
            print("Gagal membaca frame, mencoba reconnect...")
            connection_retries += 1
            
            if connection_retries > max_retries:
                print("Terlalu banyak kegagalan, menghentikan program...")
                break
                
            # Coba reconnect
            cap.release()
            time.sleep(2)
            cap = initialize_rtsp_connection()
            if cap is None:
                print("Tidak dapat reconnect, menghentikan program...")
                break
            continue
        
        # Reset counter retry jika berhasil
        connection_retries = 0
        
        # Process every 2nd frame to reduce load
        frame_count += 1
        if frame_count % 2 != 0:
            continue
        
        # RESIZE KE 640x640 (jika resolusi belum tepat)
        if frame.shape[1] != 640 or frame.shape[0] != 640:
            frame = cv2.resize(frame, (640, 640))
        
        # Deteksi objek dengan YOLOv5
        results = model(frame)
        
        # Proses hasil deteksi
        detections = results.pandas().xyxy[0]
        
        for _, detection in detections.iterrows():
            if detection['confidence'] >= 0.25:
                # Dapatkan koordinat bounding box
                x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                class_name = detection['name']
                confidence = detection['confidence']
                
                # Pastikan koordinat dalam batas frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                #lebarkan sedikit cropnya agar outline dari plat yang bocor tidak terdeteksi sebagai teks
                expand_pixels = 5  # Bisa disesuaikan 3-10 pixels
                x1 = max(0, x1 - expand_pixels)
                y1 = max(0, y1 - expand_pixels)
                x2 = min(frame.shape[1], x2 + expand_pixels) 
                y2 = min(frame.shape[0], y2 + expand_pixels)

                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Crop area untuk OCR
                cropped_img = frame[y1:y2, x1:x2]
                
                if cropped_img.size == 0:
                    continue
                
                try:
                    # Pra-pemrosesan untuk OCR
                    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                    
                    # Enhanced preprocessing
                    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                  cv2.THRESH_BINARY, 11, 2)
                    kernel = np.ones((1, 1), np.uint8)
                    dilated = cv2.dilate(thresh, kernel, iterations=1)
                    
                    # OCR
                    text = pytesseract.image_to_string(dilated, config=custom_config)
                    text = text.strip()
                    
                    if text:
                        print(f"Class: {class_name}, Confidence: {confidence:.2f}, Text: {text}")
                        
                except Exception as e:
                    # Abaikan error OCR kecil
                    pass
                
                # Gambar bounding box dan label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Tampilkan frame (optional)
        # cv2.imshow('YOLOv5 RTSP Detection - 640x640', frame)
        
        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    except KeyboardInterrupt:
        print("Program dihentikan oleh user")
        break
    except Exception as e:
        print(f"Error dalam loop utama: {e}")
        time.sleep(1)

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Program selesai")



#update 1-9-2025: memodifikasi fungsi crop untuk OCR agar tidak ada noice dari outline plat