import os
import csv
from collections import defaultdict
import cv2
import supervision as sv
from ultralytics import YOLO

# 1. Konfigurasi Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "traffic.mp4")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "output_traffic.mp4")
CSV_REPORT_PATH = os.path.join(BASE_DIR, "laporan_trafik.csv")

if not os.path.exists(VIDEO_PATH):
    print(f"ERROR: File tidak ditemukan di -> {VIDEO_PATH}")
    exit()

# 2. Load Model & Informasi Video
model = YOLO("yolov8n.pt")
SELECTED_CLASSES = [2, 3, 5, 7]  # 2: car, 3: motorcycle, 5: bus, 7: truck
CLASS_NAMES_ID = {2: "Mobil", 3: "Motor", 5: "Bus", 7: "Truk"}

video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

# Siapkan VideoWriter untuk menyimpan hasil video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_video = cv2.VideoWriter(
    OUTPUT_VIDEO_PATH, fourcc, video_info.fps, (video_info.width, video_info.height)
)

# 3. Konfigurasi Garis Penghitung & Annotator
LINE_Y = int(video_info.height * 0.75)
line_zone = sv.LineZone(
    start=sv.Point(0, LINE_Y),
    end=sv.Point(video_info.width, LINE_Y),
    triggering_anchors=[sv.Position.BOTTOM_CENTER],
)

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=0.8)

# Penyimpan statistik per jenis kendaraan
counts_in = defaultdict(int)
counts_out = defaultdict(int)

print("Memproses video & merekam output...")

# 4. Loop Pemrosesan Frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, classes=SELECTED_CLASSES, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    if detections.tracker_id is not None:
        # trigger() mengembalikan boolean array kendaraan yang baru saja melintasi garis
        crossed_in, crossed_out = line_zone.trigger(detections)

        for class_id, is_in, is_out in zip(detections.class_id, crossed_in, crossed_out):
            vehicle_type = CLASS_NAMES_ID.get(class_id, "Lainnya")
            if is_in:
                counts_in[vehicle_type] += 1
            if is_out:
                counts_out[vehicle_type] += 1

        labels = [
            f"#{tracker_id} {CLASS_NAMES_ID.get(class_id, '')}"
            for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
        ]

        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    else:
        annotated_frame = frame.copy()

    annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

    # Gambar Panel Statistik di Pojok Kiri Atas
    cv2.rectangle(annotated_frame, (15, 15), (310, 175), (0, 0, 0), -1)
    cv2.putText(annotated_frame, "STATISTIK KENDARAAN", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    y_offset = 72
    for v_type in ["Mobil", "Motor", "Bus", "Truk"]:
        text = f"{v_type}: In={counts_in[v_type]} | Out={counts_out[v_type]}"
        cv2.putText(annotated_frame, text, (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y_offset += 25

    # Simpan frame ke video output & tampilkan di layar
    out_video.write(annotated_frame)

    cv2.namedWindow("Vehicle Counter", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Vehicle Counter", 450, 800)
    cv2.imshow("Vehicle Counter", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out_video.release()
cv2.destroyAllWindows()

# 5. Ekspor Rekap Data ke CSV
with open(CSV_REPORT_PATH, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Jenis Kendaraan", "Masuk (In)", "Keluar (Out)", "Total"])
    for v_type in ["Mobil", "Motor", "Bus", "Truk"]:
        total = counts_in[v_type] + counts_out[v_type]
        writer.writerow([v_type, counts_in[v_type], counts_out[v_type], total])

print(f"\nSelesai! Video hasil tersimpan di: {OUTPUT_VIDEO_PATH}")
print(f"Laporan statistik tersimpan di: {CSV_REPORT_PATH}")