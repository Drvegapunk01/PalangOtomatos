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
import pygame
from ultralytics import YOLO
import mysql.connector
from mysql.connector import Error  # Untuk error handling
import threading
import time

warnings.filterwarnings("ignore")

# Hilangkan semua warning yang tidak perlu
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;panic"
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["GST_DEBUG"] = "0"

# Inisialisasi pygame mixer untuk audio
pygame.mixer.init()

# Konfigurasi path Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'D:\Project_andre\DeteksiPlatNomor\yolov5\Tesseract-OCR\tesseract.exe'   

# Fix Windows path issue
pathlib.PosixPath = pathlib.WindowsPath

# Load suara
try:
    sound_file = "hidupjokowi.mp3"  
    pygame.mixer.music.load(sound_file)
    print("Suara berhasil dimuat")
except Exception as e:
    print(f"Error loading sound file: {e}")

# Serial configuration
def find_board_by_name(keywords=["Arduino", "CH340", "CP210x"], baudrate=115200, timeout=1):
    try:
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
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
        try:
            ser = SER.Serial(port='COM3', baudrate=baudrate, timeout=timeout)
            return ser
        except Exception as fallback_error:
            print(f"Fallback ke COM3 juga gagal: {fallback_error}")
            return None

# Load model YOLOv8
try:
    model = YOLO('best2.pt')
    print("Model YOLOv8 berhasil dimuat")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Konfigurasi Tesseract - khusus untuk teks kapital dan angka
custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Inisialisasi kamera
print("Mengakses kamera...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolusi Kamera: {actual_width}x{actual_height}")

# Koneksi Mysql
def create_connection():
    try:
        connection = mysql.connector.connect(
            host='127.0.0.1',     
            user='root',
            password='',           
            database='plat_nomor',
            port=4306  
            #in some case pake 4306
            #jika tidak bisa pake port default: 3306
            #jika masih error silahkan lihat port Mysql di XAMPP 
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

# Ambil data database
def get_data(connection):
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT Plat FROM plat")
        results = cursor.fetchall()
        return results
    except Error as e:
        print(f"Error: {e}")
        return []

# Variabel global untuk menyimpan data plat
database_plates = []

# Ngambil data tiap 10 detik
def do_something():
    print(f"[{time.time():.0f}] Mengambil data plat nomor ...")
    try:
        conn = create_connection()
        if conn:
            data = get_data(conn)
            for row in data:
                print(row)
            conn.close()
            print("data plat nomor telah di ambil")
            return data
        else:
            print("gagal mengambil data")
            return []
    except Error as e:
        print("Error saat mengambil data dari database")
        return []

#Fungsi untuk mengambil data dari database setiap 10 detik
def every_10_seconds():
    global database_plates
    while True:
        data = do_something()
        if data:
            database_plates = data
        time.sleep(10)

# Jalankan di background thread
thread = threading.Thread(target=every_10_seconds, daemon=True)
thread.start()

# Cari board Arduino
ser = find_board_by_name()
if ser:
    print("Board terdeteksi.")
else:
    print("Tidak ada board terdeteksi.")

# Variabel status
last_platform_detection_time = 0
platform_cooldown = 5
frame_count = 0

print("Memulai proses deteksi...")

def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    return dilated

def clean_ocr_text(text):
    cleaned = ''.join(c for c in text if c.isalnum())
    return cleaned.upper()

# Koneksi DB untuk OCR
conn_ocr = create_connection()
cursor = conn_ocr.cursor(dictionary=True) if conn_ocr else None

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame")
            time.sleep(1)
            continue
        
        frame_count += 1
        if frame_count % 2 != 0:
            continue
        
        if frame.shape[1] != 640 or frame.shape[0] != 480:
            frame = cv2.resize(frame, (640, 480))
        
        results = model(frame, verbose=False)
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    confidence = box.conf[0].item()
                    if confidence >= 0.25:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        margin_x = int((x2 - x1) * 0.05)
                        margin_y = int((y2 - y1) * 0.05)
                        x1 = max(0, x1 - margin_x)
                        y1 = max(0, y1 - margin_y)
                        x2 = min(frame.shape[1], x2 + margin_x)
                        y2 = min(frame.shape[0], y2 + margin_y)
                        cropped_img = frame[y1:y2, x1:x2]
                        if cropped_img.size == 0:
                            continue
                        try:
                            processed_img = preprocess_for_ocr(cropped_img)
                            ocr_result = pytesseract.image_to_string(processed_img, config=custom_config)
                            cleaned_text = clean_ocr_text(ocr_result)
                            
                            if cleaned_text and cursor:
                                cursor.execute("SELECT Plat FROM plat")
                                rows = cursor.fetchall()
                                plate_found = False
                                for row in rows:
                                    db_text = str(row['Plat']).upper().strip()
                                    if cleaned_text == db_text:
                                        plate_found = True
                                        break                  
                                if plate_found:
                                    current_time = time.time()
                                    if current_time - last_platform_detection_time > platform_cooldown:
                                        print(f"✅ PLAT NOMOR TERDAFTAR! '{cleaned_text}' Memutar suara")
                                        if pygame.mixer.music.get_busy():
                                            pygame.mixer.music.stop()
                                        pygame.mixer.music.play()
                                        last_platform_detection_time = current_time
                                        if ser:
                                            try:
                                                ser.write(b'P')
                                                print("Sinyal dikirim ke Arduino")
                                            except Exception as e:
                                                print(f"Gagal mengirim ke Arduino: {e}")
                                else:
                                    print(f"Teks terdeteksi: '{cleaned_text}' (tidak terdaftar di database)")
                        except Exception as e:
                            print(f"Error dalam OCR: {e}")
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Plate {confidence:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        if 'cleaned_text' in locals() and cleaned_text:
                            cv2.putText(frame, f"Text: {cleaned_text}", (x1, y2 + 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.imshow('License Plate Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    except Exception as e:
        print(f"Error dalam loop utama: {e}")
        time.sleep(1)

if conn_ocr:
    conn_ocr.close()

cap.release()
cv2.destroyAllWindows()
print("Program selesai")

# (fitur tambahan)
#tambahkakn fitur Pesan LOG

# (Fitur wajib)
# -Kemampuan untuk mengambil gambar dari hasil detection saat awal besetra dengan data waktunya
# -konfigurasi GUI kedua di opratornya agar bisa bisa mendaftarkan Plat nomor

# ==========================================================================Tantangan==============================================================================
# -OCR dan YOLO yang belum di test di real case:
#     dataset yang saya gunakan adalah hasil download dan kemungkinan tidak akan sama dengan yang ada di lapangan, 
#     belum di test untuk cuaca spesifik dan juga pencahayaan tertentu
#       Solusi:
#         jika memungkinkan saya akan menggunakan model yang sudah di buat ini, jika berjalan lancar maka saya akan gunakan
#         untuk sementara waktu sambil kita mengumpulkan dataset untuk case di lapangannya dan kita akan latih OCR dan Yolov8n nya
#         tapi saya kira model yolonya sudah 70% baik ketika di uji di workshop
#         hal yang masih saya khawatirkan adalah model OCR yang tidak bisa membaca teks dengan font tertentu dan bahkan dengan sudut dan pencahayaan tertentu
#         tetapi Model OCR yang saya gunakan memiliki fitur costum model yang bisa di latih, jika di perlukan saya akan melatih modelnya 
#         tapi saat ini (9,24,2025) saya masih dalam proses pembuatan dan baru selesai 60% dan ingin uji coba di mini pc yang di sediakan dengan spek yg cukup baik
#         tapi memang komputasinya masih menggunakan CPU, di rekomendasikan untuk menggunakan GPU, tapi kayaknya bisa di ganti ketika project ini sudah jalan dan 
#         pimpinan merasa puas
# -GUI Yang memiliki kemungkinan untuk memberatkan sistem dan mengubah semua algoritma pemrosesan:
#     beberapa waktu lalu saya membuat software GUI untuk tombol kuis dan di laptop dengan spesifikasi agak lambat program GUI itu cukup membebani
#     tapi ini masih spekulasi saja, karena menurut beberapa orang library GUI python itu ringan jika di kelola dengan baik
#         solusi:
#             saya akan menggunakan java untuk membuat software GUI nya, (terkhusus untuk software registrasi plat nya) dan saya juga mempertimbangkan 
#             untuk membuat GUI di mini PC yang menjalankan YOLO dan ocr ini, tapi saya akan membuatnya menjadi dua program berbeda atau entah gimana pokoknya 
#             GUI ini tidak boleh membebani model Yolo dan ocr ini
# -RTSP yang sangat lambat:
#     saya agak takut menggunakan RTSP sebagai input videonya karena RTSP sendiri menghasilkan delay yang sangat gila gilaan, ditambah Object detection dan OCR yang akan 
#     membuat delay sekitar 5-8 detik (yang dimana sangat buruk)
#         Solusi0: menggunakan Webcam USB yang di arahkan langsung ke Plat nomornya yang memberikan kelebihan kualitas yg bagus dan gambar yang jernih serta kecepatan yang baik
        # Task: -Jika kamu melihat Alat IPC Tester, video yg di tampilkan memiliki delay yg lebih sedikit dibandingkan teknik yg kami kugakan (RTSO), lalu muncul pertanyaan,
        #        apakah protokol yg digunakan sama?, atau emang arsitektur IPC Tester emang di khususkan untuk tugas seperti ini?, jika saya dapat meniru atau menggunakan protokol yg sama
        #        dengan menggunakan protokol ini Model OCR saya akam mendeteksi dengan lebih baik karena gambarnya tidak di kompres secara ugal ugalan?
        #        jika ini hanya menggunakan protokol yg lebih baik dari yang saya gunakan ini kemungkinan akan meningkatkan akurasi dan kecepatan program
        #        Jadi task kali ini saya harus mencari tahu protokol yang di gunakan oleh IPC tester

# Bacotan author: sistem yg mirip dan lebih stabil serta sudah sukses digunakan di Industri adalah tilang digital, kelebihannya adalah dapat membaca meskipun 
#                 jaraknya agak nggk ngotak, sistem tersebut mengfokuskan proses pembacaan platnya pada hardware, jadi dia lebih stabil dan cepat
#                 kami ingin mereplikanya dengan Buget yang lebih sedikit dan hasil yang mendekati atau lebih baik(kyknya sulit sih wkwkwkwk)
#                 Akan menjadi sebuah Prestasi jika kita dapat mereplika sistem yg sudah jadi itu kemudian membuatnya ulang dengan dana yg alakadarnya
#                 Dan saya yakin, sistem palang seperti ini tidak memerlukan alat ratusan juta, cukup kopi dan alat alat seadanya kita sudah bisa mereplikanya
#                Jika saya berhasil membuat project ini, maka saya akan mempublikasikan kodenya, saya yakin ada banyak programer yang lebih hebat dari saya dan bisa membuat
#                sistem yang lebih baik dari ini, tetapi sangat berharap dengan mempublikasikan ini saya sedikit membantu orang lain dan mereka tidak perlu melakukan proses Riset 
#                dan debugging yg sudah saya lakukan.

# ==========================================================================Alur kerja==============================================================================
# Pertama tama, yolo akan mengcrop bagian  yang di deteksi sebagai plat nomor, kemudian hasil crop nya di deteksi oleh OCR untuk mengekstrak tulisan di gambar itu menjadi
# bentuk String,
# Setelah itu Stringnya akan di bandingkan dengan satu persatu data yang di simpan di database, jika sesuai dengan salah satunya maka program akan mengirim signal ke mikrokontroler
# secara serial untuk mengaktifkan palang tersebut, 
#     note0: palang ini sudah ada fitur keamanan sendiri yang membuatnya tidak bisa menutup ketika masih ada modil di atas sensornya
#     note1: program akan mengambil data dari database setiap beberapa detik atau menit lalu di simpan di array buffer, kemudian akan di bandingkan jika ada plat yang terdeteksi
#            dengan begini program tidak perlu membaca database dulu setiap ada plat karena sudah di simpan di array buffer, ini akan meningkatkan kecepatan program dalam mengeksekusi code 
# kemudain akan ada GUI yang menampilkan gambar plat yang di baca, in case ada error (amit amit) kemudian membandingkannya manual
# di GUI ini juga saya akan memberikan option untuk menampilkan log programnya agar mempermudah teknisi atau programer dalam melakukan maintanace.

# project ini saya buat sebagai bahan laporan akhir Praktek lapangan kerja juga buah tangan saya ketika magang di unit elektronika bandara rendani 
# Catatn pribadi Author: saya sangat senang di berikan kesempatan untuk membuat project se Penting ini, saya sangat berterima kasih kepada senior senior yang sudah membimbing saya
#                        selama pembuatan project ini, saya harap project yang saya buat ini akan membantu instansi terkait dalam melayani masyarakat.
    #                         -Andrith Bllaer Nichy Imanuel Reba/Andre/@Kuuhaku/@Drvegapunk01 24/9/2025
    #                         -16 Tahun
    #                         -SMK 2 Manokwari
#"Efesus 4:13 Segala perkara dapat kutanggung di dalam dia yg memberi kekuatan kepadaku"
