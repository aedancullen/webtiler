from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import sys
import cv2
import time
import socket
import threading
import numpy as np
from mss import mss

CALIB_W = 1920
CALIB_H = 1080

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1)

MIN_MATCH_COUNT = 10

reglock = threading.Lock()

def calib_cap_read():
    calib_cap = cv2.VideoCapture(int(sys.argv[1]))
    codec = cv2.VideoWriter_fourcc(	'M', 'J', 'P', 'G'	)
    calib_cap.set(6, codec)
    calib_cap.set(5, 30)
    calib_cap.set(3, CALIB_W)
    calib_cap.set(4, CALIB_H)
    _, frame = calib_cap.read()
    frame = frame[CALIB_H//4:3*CALIB_H//4, CALIB_W//4:3*CALIB_W//4]
    frame = cv2.resize(frame, (CALIB_W, CALIB_H))
    calib_cap.release()
    return _, frame

if len(sys.argv) > 2:
    calib_cap = cv2.VideoCapture(int(sys.argv[1]))
    codec = cv2.VideoWriter_fourcc(	'M', 'J', 'P', 'G'	)
    calib_cap.set(6, codec)
    calib_cap.set(5, 30)
    calib_cap.set(3, CALIB_W)
    calib_cap.set(4, CALIB_H)
    while True:
        _, frame = calib_cap.read()
        frame = frame[CALIB_H//4:3*CALIB_H//4, CALIB_W//4:3*CALIB_W//4]
        frame = cv2.resize(frame, (CALIB_W, CALIB_H))
        cv2.imshow("frame", frame);
        cv2.waitKey(1);

frame_container = [None]

def cap_routine(frame_container):
    with mss(with_cursor=True) as sct:
        while True:
            sct_img = sct.grab(sct.monitors[1])
            new_frame = np.array(sct_img)
            new_frame = cv2.resize(new_frame, (CALIB_W, CALIB_H))
            frame_container[0] = new_frame
            time.sleep(0.001)

t = threading.Thread(target=cap_routine, args=(frame_container,))
t.daemon = True
t.start()

def detect_features(detector, frame):
    keypoints, descrs = detector.detectAndCompute(frame, None)
    if descrs is None:
        descrs = []
    return keypoints, descrs

class Handler(BaseHTTPRequestHandler):

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

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
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', str(jpg.size))
        self.end_headers()
        self.wfile.write(jpg.tobytes())

    def do_GET(self):
        if self.path.startswith("/p"):
            path = self.path.split("&")
            width = path[0][path[0].find("w")+2:]
            height = path[1][path[1].find("h")+2:]
            width = int(width)
            height = int(height)

            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=--jpegbound")
            self.end_headers()

            calib = cv2.imread("calib3.jpg")
            calib = cv2.resize(calib, (width, height))
            rect = (0, 0, width, height)
            tx_calib = self.enc_frame(calib)
            black = np.zeros(shape=[width, height, 3], dtype=np.uint8)
            tx_black = self.enc_frame(black)

            detector = cv2.ORB_create(nfeatures = 10000)
            matcher = cv2.FlannBasedMatcher(flann_params, {})

            calib_points, calib_descs = detect_features(detector, calib)
            calib_descs = np.uint8(calib_descs)
            matcher.add([calib_descs])

            self.send_frame(tx_black)
            self.send_frame(tx_black)

            reg_item = None
            with reglock:
                print(width,"x",height,"client got reglock")
                self.send_frame(tx_calib)
                self.send_frame(tx_calib)

                res, calib_frame = calib_cap_read()

                frame_points, frame_descrs = detect_features(detector, calib_frame)
                if len(frame_points) >= MIN_MATCH_COUNT:
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

                            reg_item = (quad_orig, quad_transform)

                self.send_frame(tx_black)
                self.send_frame(tx_black)

            if reg_item is not None:
                print(width,"x",height,"client registered")
            else:
                print(width,"x",height,"registration failed")
                return

            M = cv2.getPerspectiveTransform(reg_item[1], reg_item[0])

            out_container = [None]
            term_container = [None]
            def proc_thread(in_container, out_container, M):
                last_frame_proc = None
                while True:
                    frame = in_container[0]
                    if not (frame is last_frame_proc):
                        trans_frame = cv2.warpPerspective(frame, M, (width, height))
                        out_container[0] = self.enc_frame(trans_frame)
                        last_frame_proc = frame
                    time.sleep(0.001)

                    if term_container[0] is not None:
                        print(width,"x",height,"proc_thread terminating")
                        break

            t = threading.Thread(target=proc_thread, args=(frame_container, out_container, M))
            t.daemon = True
            t.start()

            last_frame = None
            try:
                while True:
                    frame = out_container[0]
                    if not (frame is last_frame):
                        self.send_frame(frame)
                        last_frame = frame
                    time.sleep(0.001)

            except (socket.error, socket.timeout) as e:
                print(width,"x",height,"client unregistered")
                term_container[0] = True

        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            with open("index.html", "rb") as fh:
                self.wfile.write(fh.read())

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    pass

if __name__ == '__main__':
    server = ThreadedHTTPServer(("", 80), Handler)
    server.serve_forever()
