#!/usr/bin/env python3
import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils_tasks.goal_capture import GoalCaptureStore


class FrameStore:
    def __init__(self):
        self.condition = threading.Condition()
        self.jpeg = None
        self.sequence = 0

    def update(self, jpeg: bytes) -> None:
        with self.condition:
            self.jpeg = jpeg
            self.sequence += 1
            self.condition.notify_all()

    def latest(self):
        with self.condition:
            return self.jpeg, self.sequence

    def wait_after(self, sequence: int, timeout: float = 5.0):
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence > sequence,
                timeout=timeout,
            )
            return self.jpeg, self.sequence


class VisualizationNode(Node):
    def __init__(
        self,
        topic: str,
        rgb_topic: str,
        jpeg_quality: int,
        store: FrameStore,
        goal_store: GoalCaptureStore,
    ):
        super().__init__("navdp_visualization_server")
        self.bridge = CvBridge()
        self.jpeg_quality = jpeg_quality
        self.store = store
        self.goal_store = goal_store
        self.visualization_subscription = self.create_subscription(
            Image,
            topic,
            self._visualization_callback,
            qos_profile_sensor_data,
        )
        self.rgb_subscription = self.create_subscription(
            Image,
            rgb_topic,
            self._rgb_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(f"subscribed to {topic}")
        self.get_logger().info(f"subscribed to {rgb_topic} for goal capture")

    def _visualization_callback(self, message: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            encoded, jpeg = cv2.imencode(
                ".jpg",
                image,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            if not encoded:
                raise RuntimeError("OpenCV JPEG encoding failed")
            self.store.update(jpeg.tobytes())
        except Exception as error:
            self.get_logger().error(f"JPEG conversion failed: {error}")

    def _rgb_callback(self, message: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            self.goal_store.update(image)
        except Exception as error:
            self.get_logger().error(f"D435 RGB conversion failed: {error}")


def make_handler(store: FrameStore, goal_store: GoalCaptureStore):
    class VisualizationHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                page = (
                    "<!doctype html><html><head><meta charset='utf-8'>"
                    "<title>NavDP Live</title>"
                    "<style>html,body{margin:0;background:#111;color:#eee;"
                    "font-family:sans-serif}header{display:flex;align-items:center;"
                    "gap:12px;padding:10px 14px}button{padding:7px 12px;"
                    "cursor:pointer}#status{font-size:14px;color:#bbb}"
                    "img{display:block;width:100%;height:auto}</style></head>"
                    "<body><header><strong>NavDP live visualization</strong>"
                    "<button id='capture' type='button'>拍摄 Goal</button>"
                    "<span id='status'></span></header>"
                    "<img src='/stream.mjpg' alt='waiting for NavDP frame'>"
                    "<script>const button=document.getElementById('capture');"
                    "const status=document.getElementById('status');"
                    "button.onclick=async()=>{button.disabled=true;"
                    "status.textContent='正在拍摄...';try{"
                    "const response=await fetch('/capture-goal',{method:'POST'});"
                    "const data=await response.json();"
                    "if(!response.ok)throw new Error(data.error||'拍摄失败');"
                    "status.textContent='已保存 '+data.latest;"
                    "}catch(error){status.textContent=error.message;}"
                    "finally{button.disabled=false;}};</script></body></html>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            if self.path == "/latest.jpg":
                jpeg, _ = store.latest()
                if jpeg is None:
                    self.send_error(503, "No NavDP visualization frame yet")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(jpeg)
                return
            if self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                sequence = -1
                try:
                    while True:
                        jpeg, next_sequence = store.wait_after(sequence)
                        if jpeg is None or next_sequence == sequence:
                            continue
                        sequence = next_sequence
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                        )
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            self.send_error(404)

        def do_POST(self):
            if self.path != "/capture-goal":
                self.send_error(404)
                return
            try:
                archive, latest = goal_store.capture()
            except RuntimeError as error:
                self._send_json(503, {"ok": False, "error": str(error)})
                return
            except OSError as error:
                self._send_json(500, {"ok": False, "error": str(error)})
                return
            self._send_json(
                200,
                {
                    "ok": True,
                    "archive": str(archive),
                    "latest": str(latest),
                },
            )

        def log_message(self, format, *args):
            return

    return VisualizationHandler


def parse_args():
    parser = argparse.ArgumentParser(description="Serve /navdp/visualization as MJPEG")
    parser.add_argument("--topic", default="/navdp/visualization")
    parser.add_argument("--rgb-topic", default="/cam_head/d435/color/image_raw")
    parser.add_argument("--goal-dir", default="goals")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = FrameStore()
    goal_store = GoalCaptureStore(Path(args.goal_dir))
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(store, goal_store),
    )
    server_thread = threading.Thread(
        target=server.serve_forever,
        name="navdp_mjpeg_http",
        daemon=True,
    )
    rclpy.init(args=[])
    node = VisualizationNode(
        args.topic,
        args.rgb_topic,
        args.jpeg_quality,
        store,
        goal_store,
    )
    server_thread.start()
    node.get_logger().info(f"viewer ready: http://{args.host}:{args.port}")
    node.get_logger().info(f"goal captures: {goal_store.goal_dir.resolve()}")
    try:
        rclpy.spin(node)
    finally:
        server.shutdown()
        server.server_close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
