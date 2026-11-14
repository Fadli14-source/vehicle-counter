# 🚦 Smart Traffic Monitoring: Vehicle Counting & Classification

Sistem Computer Vision berbasis *deep learning* untuk mendeteksi, melacak, dan menghitung volume kendaraan secara otomatis berdasarkan jenisnya (Mobil, Motor, Bus, dan Truk) dari rekaman video lalu lintas.

## 🎯 Fitur Utama
- **Real-Time Object Detection:** Menggunakan model **YOLOv8** untuk mengenali berbagai kategori kendaraan dengan cepat.
- **Multi-Object Tracking:** Menggunakan algoritma **ByteTrack** untuk memberikan ID unik pada setiap kendaraan sehingga tidak terjadi perhitungan ganda (*double counting*).
- **Virtual Line Crossing:** Menghitung kendaraan masuk (`In`) dan keluar (`Out`) menggunakan titik jangkar bagian bawah kendaraan (`BOTTOM_CENTER`).
- **Automated Reporting:** Menampilkan statistik secara langsung di layar video dan mengekspor rekapitulasi data ke dalam format `.csv`.

## 🛠️ Teknologi yang Digunakan
- **Python 3.10+**
- **Ultralytics YOLOv8** (Object Detection & Tracking)
- **Supervision** (LineZone & Visual Annotation)
- **OpenCV** (Video Processing)

## 🚀 Cara Menjalankan Proyek
1. Clone repository ini:
   ```bash
   git clone https://github.com/username-kamu/vehicle-counter-cv.git
   cd vehicle-counter-cv
