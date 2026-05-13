#!/usr/bin/env python3
# ros2_metal_obb/ros2_metal_obb/metal_obb_node.py

import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped, Point32
from ultralytics import YOLO


class MetalOBBNode(Node):

    def __init__(self):
        super().__init__("metal_obb")

        # ── 参数 ──────────────────────────────────
        self.declare_parameter("model_path", "best.pt")
        self.declare_parameter("conf",       0.5)
        self.declare_parameter("iou",        0.5)
        self.declare_parameter("img_topic",  "/camera/color/image_raw")
        self.declare_parameter("pub_vis",    True)

        model_path = self.get_parameter("model_path").value
        conf       = self.get_parameter("conf").value
        iou        = self.get_parameter("iou").value
        img_topic  = self.get_parameter("img_topic").value
        self.pub_vis = self.get_parameter("pub_vis").value

        # ── 模型 ──────────────────────────────────
        self.model  = YOLO(model_path, task="obb")
        self.conf   = conf
        self.iou    = iou
        self.bridge = CvBridge()

        # ── 订阅 ──────────────────────────────────
        self.sub = self.create_subscription(
            Image,
            img_topic,
            self.cb,
            QoSPresetProfiles.SENSOR_DATA.value,
        )

        # ── 发布 ──────────────────────────────────
        # 角点像素坐标（顺时针4点，z=0）
        self.pub_corners = self.create_publisher(
            PolygonStamped, "/metal_obb/corners", 10
        )
        # 可视化图
        if self.pub_vis:
            self.pub_result = self.create_publisher(
                Image, "/metal_obb/result", 10
            )

        # ── 统计 ──────────────────────────────────
        self._cnt = 0
        self._acc = 0.0

        self.get_logger().info(
            f"\n  model : {model_path}"
            f"\n  conf  : {conf}  iou: {iou}"
            f"\n  topic : {img_topic}"
            f"\n  pub_vis: {self.pub_vis}"
        )

    # ──────────────────────────────────────────
    def cb(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"imgmsg_to_cv2 失败: {e}")
            return

        t0 = time.perf_counter()
        result = self.model(bgr, conf=self.conf, iou=self.iou, verbose=False)[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 统计耗时
        self._cnt += 1
        self._acc += elapsed_ms
        if self._cnt % 30 == 0:
            self.get_logger().info(
                f"avg {self._acc/self._cnt:.1f}ms | "
                f"last {elapsed_ms:.1f}ms | "
                f"det {0 if result.obb is None else len(result.obb)}"
            )

        if result.obb is None or len(result.obb) == 0:
            return

        # 取置信度最高的目标
        confs  = result.obb.conf.cpu().numpy()
        xywhrs = result.obb.xywhr.cpu().numpy()
        best   = int(np.argmax(confs))

        corners = self._to_corners(*xywhrs[best])

        # 发布角点
        poly            = PolygonStamped()
        poly.header     = msg.header
        for x, y in corners:
            p = Point32(x=float(x), y=float(y), z=0.0)
            poly.polygon.points.append(p)
        self.pub_corners.publish(poly)

        # 发布可视化
        if self.pub_vis:
            vis = self._draw(bgr, xywhrs, confs, elapsed_ms)
            vis_msg         = self.bridge.cv2_to_imgmsg(vis, "bgr8")
            vis_msg.header  = msg.header
            self.pub_result.publish(vis_msg)

    # ──────────────────────────────────────────
    @staticmethod
    def _to_corners(cx, cy, w, h, theta) -> np.ndarray:
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx = np.array([-w/2,  w/2,  w/2, -w/2])
        dy = np.array([-h/2, -h/2,  h/2,  h/2])
        x  = cx + dx * cos_t - dy * sin_t
        y  = cy + dx * sin_t + dy * cos_t
        return np.stack([x, y], axis=1).astype(np.float32)

    def _draw(self, bgr, xywhrs, confs, elapsed_ms) -> np.ndarray:
        vis = bgr.copy()
        for xywhr, c in zip(xywhrs, confs):
            corners = self._to_corners(*xywhr).astype(np.int32)
            cv2.polylines(vis, [corners.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
            for pt in corners:
                cv2.circle(vis, tuple(pt), 4, (0, 0, 255), -1)
            cx, cy = corners.mean(axis=0).astype(int)
            cv2.putText(vis, f"{c:.2f}", (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis, f"{elapsed_ms:.1f}ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        return vis


# ──────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = MetalOBBNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
