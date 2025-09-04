import cv2
import torch
import numpy as np
from pathlib import Path
import pathlib
import pytesseract
import time
import warnings
import serial as SER
import os
import serial.tools.list_ports
import pygame  # Import pygame untuk memutar suara
warnings.filterwarnings("ignore")

# Hilangkan semua warning yang tidak perlu
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;panic"
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["GST_DEBUG"] = "0"

# Inisialisasi pygame mixer untuk audio
pygame.mixer.init()

# Konfigurasi path Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'E:\Project\Deteksi_Tulisan\ML\yolov5\Tesseract-OCR\tesseract.exe'

# Fix Windows path issue
pathlib.PosixPath = pathlib.WindowsPath

# Load suara "Hidup Jokowi"
try:
    # Ganti dengan path file suara Anda
    sound_file = "hidupjokowi.mp3"  
    pygame.mixer.music.load(sound_file)
    print("Suara 'Hidup Jokowi' berhasil dimuat")
except Exception as e:
    print(f"Error loading sound file: {e}")

#Serial configuration
def find_board_by_name(keywords=["Arduino", "CH340", "CP210x"], baudrate=115200, timeout=1):
    try:
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            # Periksa apakah nama/deskripsi mengandung keyword
            if any(keyword.lower() in (port.description or "").lower() for keyword in keywords):
                try:
                    ser = SER.Serial(port.device, baudrate, timeout=timeout)
                    print(f"Board ditemukan: {port.device} ({port.description})")
                    return ser
                except Exception as e:
                    print(f"Gagal membuka {port.device}: {e}")
        print("Board tidak ditemukan berdasarkan nama.")
        return None
    
    except Exception as e:
        print(f"Error selama pencarian port: {e}")
        print("Menggunakan Port COM3 sebagai fallback")
        
        try:
            ser = SER.Serial(
                port='COM3',      # ganti dengan port yang sesuai di PC/laptopmu
                baudrate=baudrate,
                timeout=timeout
            )
            return ser
        except Exception as fallback_error:
            print(f"Fallback ke COM3 juga gagal: {fallback_error}")
            return None


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

# Fungsi untuk menginisialisasi koneksi RTSP dengan timeout yang diperbaiki
def initialize_rtsp_connection():
    max_attempts = 5  # Increased from 3 to 5
    attempt = 0
    
    while attempt < max_attempts:
        try:
            rtsp_url = "rtsp://admin:Rendani122.2@192.168.1.88:554/streaming/channels/101"
            
            # Set timeout options for RTSP connection
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            
            # PERBAIKAN: Set timeout yang lebih realistis
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # Increased to 10 seconds
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)   # Added read timeout
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)
            
            # Set resolusi langsung ke 640x640
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
            
            # PERBAIKAN: Tunggu lebih lama untuk inisialisasi
            time.sleep(2)  # Increased from 0.5 to 2 seconds
            
            # PERBAIKAN: Coba baca beberapa frame untuk memastikan koneksi stabil
            for i in range(3):
                ret, frame = cap.read()
                if ret and frame is not None:
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"RTSP berhasil - Resolusi: {actual_width}x{actual_height}")
                    return cap
                time.sleep(0.5)
            
            cap.release()
            attempt += 1
            print(f"Percobaan {attempt}/{max_attempts} gagal, mencoba lagi dalam 3 detik...")
            time.sleep(3)  # Increased from 2 to 3 seconds
            
        except Exception as e:
            print(f"Error: {e}")
            attempt += 1
            time.sleep(3)  # Increased from 2 to 3 seconds
    
    print("Tidak dapat terhubung ke RTSP setelah beberapa percobaan")
    return None

# Inisialisasi koneksi RTSP
print("Menghubungkan ke CCTV RTSP...")
cap = initialize_rtsp_connection()
if cap is None:
    print("Tidak dapat mengakses stream RTSP")
    exit(1)

    
# Contoh penggunaan
ser = find_board_by_name()
if ser:
    print("board terdeteksi.")
else:
    print("Tidak ada board terdeteksi.")


# Dapatkan resolusi aktual
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolusi CCTV: {actual_width}x{actual_height}")

# Variabel untuk melacak status koneksi
connection_retries = 0
max_retries = 10  # Increased from 5 to 10
frame_count = 0

# Variabel untuk melacak kapan terakhir kali suara diputar
last_platform_detection_time = 0
platform_cooldown = 5  # Cooldown 5 detik antara pemutaran suara

print("Memulai proses deteksi dengan resolusi 640x640...")

while True:
    try:
        # Baca frame dengan timeout handling
        start_time = time.time()
        ret, frame = cap.read()
        read_time = time.time() - start_time
        
        # PERBAIKAN: Log waktu pembacaan frame untuk debugging
        if read_time > 1.0:
            print(f"Peringatan: Pembacaan frame memakan waktu {read_time:.2f} detik")
        
        if not ret:
            print("Gagal membaca frame, mencoba reconnect...")
            connection_retries += 1
            
            if connection_retries > max_retries:
                print("Terlalu banyak kegagalan, menghentikan program...")
                break
                
            # Coba reconnect dengan timeout yang lebih baik
            cap.release()
            time.sleep(3)  # Increased from 2 to 3 seconds
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
            frame = cv2.resize(frame, (1920, 1920))
        
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
                    
                    # OCR dengan timeout handling
                    try:
                        text = pytesseract.image_to_string(dilated, config=custom_config, timeout=2)  # Added 2-second timeout
                        text = text.strip()
                        
                        if text:
                            print(f"Class: {class_name}, Confidence: {confidence:.2f}, Text: {text}")
                            
                            # Cek jika teks mengandung "platform" (case insensitive)
                            if "platform" in text.lower():
                                current_time = time.time()
                                # Cek cooldown untuk menghindari spam suara
                                if current_time - last_platform_detection_time > platform_cooldown:
                                    print("Tulisan 'platform' terdeteksi! Memutar suara 'Hidup Jokowi'")
                                    pygame.mixer.music.play()
                                    last_platform_detection_time = current_time
                                    
                                    # Kirim sinyal ke Arduino jika terhubung
                                    if ser:
                                        try:
                                            ser.write(b'P')  # Kirim karakter 'P' ke Arduino
                                            print("Sinyal dikirim ke Arduino")
                                        except Exception as e:
                                            print(f"Gagal mengirim ke Arduino: {e}")
                            
                    except RuntimeError as e:
                        if "timeout" in str(e).lower():
                            print("OCR timeout, melanjutkan...")
                        else:
                            print(f"OCR error: {e}")
                        
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
