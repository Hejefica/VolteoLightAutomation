import cv2
import numpy as np

# Cargar el modelo YOLOv3 pre-entrenado
net = cv2.dnn.readNet("./darknet/data/yolov3.weights", './darknet/cfg/yolov3.cfg')

# Cargar las clases que YOLOv3 puede detectar
classes = []
with open("./darknet/data/coco.names", "r") as f:
    classes = f.read().strip().split('\n')

# Inicializar la cámara
cap = cv2.VideoCapture(0)  # 0 para la cámara por defecto, cambiar esto si hay múltiples cámaras

while True:
    ret, frame = cap.read()

    # Obtener la altura y el ancho de la imagen
    height, width, _ = frame.shape

    # Preprocesar la imagen para que sea compatible con YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Pasar la imagen a través de la red YOLO
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Inicializar listas para las cajas, confianzas y clases detectadas
    boxes = []
    confidences = []
    class_ids = []

    # Filtrar detecciones con confianza superior a un umbral
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar eliminación de no máxima supresión
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
