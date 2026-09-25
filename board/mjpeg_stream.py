#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.
"""Tiny dependency-light MJPEG server — publish the exact frames the model sees.

The control loop calls `update(name, frame_bgr)` each tick; browsers watch the live streams.
Serves:
  /                 HTML page with the wrist + top feeds side by side (+ live action/state readout)
  /stream/<name>    multipart/x-mixed-replace MJPEG for that camera
  /status           JSON: latest target/state degrees + fps

Stdlib http.server + cv2 (already on the board). Reachable from the host via
`adb forward tcp:8080 tcp:8080` -> open http://localhost:8080.
"""
import json, threading, time
import cv2
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Streamer:
    def __init__(self, port=8080, quality=80):
        self.port = port
        self.quality = quality
        self._lock = threading.Lock()
        self._jpg = {}                 # name -> latest encoded jpg bytes
        self._status = {}              # arbitrary json-able dict
        self._httpd = None

    def update(self, name, frame_bgr):
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            with self._lock:
                self._jpg[name] = buf.tobytes()

    def set_status(self, **kw):
        with self._lock:
            self._status.update(kw)

    def _get(self, name):
        with self._lock:
            return self._jpg.get(name)

    def start(self):
        streamer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/" or self.path.startswith("/index"):
                    self._page()
                elif self.path == "/status":
                    self._json()
                elif self.path.startswith("/stream/"):
                    self._mjpeg(self.path.rsplit("/", 1)[-1])
                else:
                    self.send_error(404)

            def _page(self):
                html = ("""<!doctype html><html><head><meta charset="utf-8"><title>ACT - what the model sees</title>
<style>body{background:#111;color:#ddd;font-family:sans-serif;text-align:center;margin:0;padding:12px}
.row{display:flex;gap:16px;justify-content:center;flex-wrap:wrap}
figure{margin:0}img{width:min(46vw,640px);image-rendering:pixelated;border:1px solid #333;background:#000}
figcaption{margin-top:6px;color:#9cf}#status{margin-top:14px;font-family:monospace;white-space:pre;color:#9f9}</style></head>
<body><h2>ACT policy — live camera views (as fed to the model)</h2>
<div class="row">
<figure><img src="/stream/wrist"><figcaption>wrist</figcaption></figure>
<figure><img src="/stream/top"><figcaption>top</figcaption></figure></div>
<div id="status">connecting...</div>
<script>setInterval(async()=>{try{const r=await fetch('/status');const s=await r.json();
document.getElementById('status').textContent=
'mode:  '+(s.mode||'?')+'\\nfps:   '+(s.fps||0).toFixed(1)+'   NPU: '+(s.nss_ms||0).toFixed(0)+' ms/replan'+
'\\nstate (deg): '+(s.state||[]).map(v=>v.toFixed(1).padStart(7)).join(' ')+
'\\ntarget(deg): '+(s.target||[]).map(v=>v.toFixed(1).padStart(7)).join(' ');}catch(e){}},300);</script>
</body></html>""").encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)

            def _json(self):
                with streamer._lock:
                    body = json.dumps(streamer._status).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _mjpeg(self, name):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        jpg = streamer._get(name)
                        if jpg is not None:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                             b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n")
                            self.wfile.write(jpg)
                            self.wfile.write(b"\r\n")
                        time.sleep(1 / 20.0)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._httpd = ThreadingHTTPServer(("0.0.0.0", self.port), H)
        t = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        t.start()
        return self

    def stop(self):
        if self._httpd:
            self._httpd.shutdown()
