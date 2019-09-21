from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import cv2
import time
import socket
import threading
import numpy as np

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

MIN_MATCH_COUNT = 10

reg = []
reglock = threading.Lock()
framelock = threading.Lock()

calib_cap = cv2.VideoCapture(2)
calib_cap.set(3, 1920)
calib_cap.set(4, 1080)

#cap = cv2.VideoCapture(0)
res, init_frame = calib_cap.read()

# iz veery hecky
frame_container = [init_frame]

def cap_routine(frame_container):
    while True:
        res, new_frame = cap.read()
        with framelock:
            frame_container[0] = new_frame
            
t = threading.Thread(target=cap_routine, args=(frame_container,))
t.daemon = True
#t.start()

def detect_features(detector, frame):
    keypoints, descrs = detector.detectAndCompute(frame, None)
    if descrs is None:
        descrs = []
    return keypoints, descrs

class Handler(BaseHTTPRequestHandler):

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    
    def handle(self):
        try:
            return BaseHTTPRequestHandler.handle(self)
        except (socket.error, socket.timeout) as e:
            pass

    def enc_frame(self, frame):
        ret, jpg = cv2.imencode('.jpg', frame, self.encode_param)
        return jpg
        
    def send_frame(self, jpg):
        self.wfile.write(b"--jpegbound")
        self.send_header('Content-type', 'image/jpeg')
        self.send_header('Content-length', str(jpg.size))
        self.end_headers()
        self.wfile.write(jpg.tostring())
        
    def do_GET(self):
        if self.path.startswith("/p"):
            path = self.path.split("&")
            width = path[0][path[0].find("w")+2:]
            height = path[1][path[1].find("h")+2:]
            width = int(width)
            height = int(height)

            self.send_response(200)
            self.send_header("Content-type", "multipart/x-mixed-replace; boundary=--jpegbound")
            self.end_headers()
            
            calib = cv2.imread("calib.png")
            calib = cv2.resize(calib, (width, height))
            rect = (0, 0, width, height)
            tx_calib = self.enc_frame(calib)
            black = np.zeros(shape=[width, height, 3], dtype=np.uint8)
            tx_black = self.enc_frame(black)

            detector = cv2.ORB_create( nfeatures = 1000 )
            matcher = cv2.FlannBasedMatcher(flann_params, {})

            calib_points, calib_descs = detect_features(detector, calib)
            calib_descs = np.uint8(calib_descs)
            matcher.add([calib_descs])
            
            reg_item = None
            with reglock:
                print(width,"x",height,"client got reglock")
                self.send_frame(tx_calib)
                self.send_frame(tx_calib)
                time.sleep(3)

                res, calib_frame = calib_cap.read()

                frame_points, frame_descrs = detect_features(detector, calib_frame)
                matches = matcher.knnMatch(frame_descrs, k = 2)
                matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
                if len(matches) >= MIN_MATCH_COUNT:
                    p0 = [calib_points[m.trainIdx].pt for m in matches]
                    p1 = [frame_points[m.queryIdx].pt for m in matches]
                    p0, p1 = np.float32((p0, p1))
                    H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
                    status = status.ravel() != 0
                    if status.sum() >= MIN_MATCH_COUNT:
                        p0, p1 = p0[status], p1[status]
                        x0, y0, x1, y1 = rect
                        quad_orig = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                        quad_transform = cv2.perspectiveTransform(quad_orig.reshape(1, -1, 2), H).reshape(-1, 2)

                        reg_item = (H, quad_orig, quad_transform)
                        reg.append(reg_item)

                self.send_frame(tx_black)
                self.send_frame(tx_black)
                
            if not (reg_item is None):
                print(width,"x",height,"client registered")
            else:
                print(width,"x",height,"registration failed")
                return
            
            last_frame = None
            try:
                while True:
                    tx_frame = None
                    with framelock:
                        frame = frame_container[0]
                        if not (frame is last_frame):
                            M = cv2.getPerspectiveTransform(reg_item[2], reg_item[1])
                            trans_frame = cv2.warpPerspective(frame, M, (width, height))

                            tx_frame = self.enc_frame(trans_frame)
                            last_frame = frame
                            
                    if tx_frame is not None:
                        self.send_frame(tx_frame)
                        self.send_frame(tx_frame)
                        
                    time.sleep(1/60.0)
                    
            except (socket.error, socket.timeout) as e:
                with reglock:
                    reg.remove(reg_item)
                print(width,"x",height, "client unregistered")

        else:
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            with open("index.html", "rb") as fh:
                self.wfile.write(fh.read())

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass

if __name__ == '__main__':
    server = ThreadedHTTPServer(("", 8080), Handler)
    server.serve_forever()
