import rtsp
with rtsp.Client("rtsp://volteoluz:12345@10.0.0.157/live") as client: # previews USB webcam 0
    client.preview()



RTSP_URL = "rtsp://volteoluz:12345@10.0.0.157/live"
client = rtsp.Client(rtsp_server_uri = RTSP_URL)

width = 640
height = 480

client.read().resize([width, height]).show()
client.close()