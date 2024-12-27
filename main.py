import cv2
import numpy as np
import time

# YOLO model dosyalarının yolları (önceden mevcut olmalıdır)
weights_path = "yolov3.weights"
config_path = "yolov3.cfg"
names_path = "coco.names"


# COCO sınıf isimlerini yükleme
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# YOLO ağını yükleme
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Video dosyasından okuma
video_path = "video1.mp4"
cap = cv2.VideoCapture(video_path)

# Orijinal video FPS, çözünürlük ve gecikme süresini hesaplama
original_fps = cap.get(cv2.CAP_PROP_FPS)  # Videonun gerçek FPS değeri
frame_delay = 1 / original_fps  # Her bir kare için teorik gecikme süresi (saniye cinsinden)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Video genişliği
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Video yüksekliği
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # Toplam kare sayısı (video süresi kontrolü için)

print(f"Process starting.")
print(f"Original video FPS: {original_fps}")
print(f"Frame delay: {frame_delay} s")
print(f"Frame width: {frame_width} px")
print(f"Frame height: {frame_height} px")
print(f"Total frames: {total_frames}")


# Çıkış videosunu ayarlama
out = cv2.VideoWriter(f"output_video.avi", cv2.VideoWriter_fourcc(*'XVID'), original_fps,
                      (frame_width, frame_height))

# İşleme döngüsü
frame_count = 0
while cap.isOpened():
    start_time = time.time()  # İşlem başlama zamanı
    ret, frame = cap.read()
    if not ret:
        break  # Video bittiğinde döngüyü bitir
    frame_count += 1
    if frame_count % 30 == 0:  # Her 30 karede bir ilerlemeyi göster
        processing_frame = frame_count / total_frames * 100
        print(f"Processing frame {round(processing_frame)}")

    # YOLO için ön işleme
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    # Tespit edilen nesneleri işleme
    class_ids = []
    confidences = []
    boxes = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]  # Bu değerlerden en yüksek olanı sınıfı ifade eder
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Güven eşiği
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aşırı bindirme tespiti: Non-Maximum Suppression (NMS)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Algılanan nesneleri çizme
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Yeşil renk (tespit edilen nesne çerçevesi)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Çıkış videosunu kaydetme ve ekranda gösterme
    out.write(frame)  # Kareyi çıktı videosuna yazma
    cv2.imshow("YOLO Object Detection", frame)  # Görüntüyü ekranda gösterme

    # İşlenen bir kare için geçen süreyi ölç
    processing_time = time.time() - start_time  # İşleme süresi
    if processing_time < frame_delay:
        time.sleep(frame_delay - processing_time)  # Gerekirse uyuyarak FPS'yi eşitle

    # "q" tuşuna basarak çıkışı sağlama
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\nProcessing completed")
# Kaynakları serbest bırakma
cap.release()
out.release()
cv2.destroyAllWindows()