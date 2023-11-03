import cv2

import rtsp


#print(cv2.getBuildInformation())

global frame 
frame = None
with rtsp.Client("rtsp://volteoluz:12345@10.0.0.157/live") as client: # previews USB webcam 0
    cap = cv2.VideoCapture("rtsp://volteoluz:12345@10.0.0.157/live")  # 0 para la cámara por defecto, cambiar esto si hay múltiples cámaras
    #cap = cv2.VideoCapture("rtsp://volteoluz:12345@10.0.0.157/live")

    while True:
        ret,frame = cap.read()
    
        cv2.imshow("Current frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):

            break
    cap.release()

    cv2.destroyAllWindows()
# you can pass now the frame to your application for further processing
