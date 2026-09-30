#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Docking autónomo Create 3 solo con LiDAR  (Kalman-Robotics/create3_dock_challenge)
 
Firma que se busca (medidas oficiales del README):
  - 2 cajas de 8 cm de ancho que sobresalen 8 cm de la pared
  - hueco libre entre cajas: 9.5 cm  (eje del dock = centro del hueco)
  - el plano del LiDAR corta las cajas a media altura
 
Percepción:
  1. /scan -> (x, y) en base_link usando TF laser_link->base_link
     (el LiDAR está girado 180° y descentrado: sin TF el eje sale corrido).
  2. RANSAC de paredes (la sala tiene 4 paredes lisas).
  3. En cada pared: puntos que sobresalen 3-11.5 cm hacia el robot -> se agrupan
     -> se busca el PAR de grupos (cajas) separados ~9.5 cm.
  4. Eje = centro del hueco (sobre la pared); normal = normal de la pared
     (ajustada con cientos de puntos, muy precisa).
  5. Se filtra en el frame odom para no perder el dock entre lecturas.
 
FSM: SEARCH -> GO_TO_PREDOCK -> ALIGN_HEADING -> DOCKING_APPROACH -> DOCKED
     (+ RECOVER si el acople sale torcido).
Se sigue empujando lento hasta que /dock_status diga is_docked (no se frena
por distancia: "quedarse corto" es el fallo más común del reto).
"""
 
import math
from enum import IntEnum
 
import numpy as np
import rclpy
from rclpy.exceptions import ParameterAlreadyDeclaredException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
 
import tf2_ros
from geometry_msgs.msg import Twist
from irobot_create_msgs.msg import DockStatus
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
 
 
def clamp(x, lo, hi):
    return max(lo, min(hi, x))
 
 
def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))
 
 
def rot(v, ang):
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
 
 
class State(IntEnum):
    SEARCH = 0
    GO_TO_PREDOCK = 1
    ALIGN_HEADING = 2
    DOCKING_APPROACH = 3
    DOCKED = 4
    RECOVER = 5
    EXPLORE = 6      # ir hacia el espacio libre más grande y volver a buscar
 
 
# --------------------------------------------------------------------------- #
# Percepción (numpy puro)
# --------------------------------------------------------------------------- #
class MarkerDetector:
    def __init__(self, box_w=0.08, box_depth=0.08, gap=0.095, line_tol=0.012,
                 band_lo=0.03, min_wall_pts=25, max_lines=5, iters=120, seed=1):
        self.box_w, self.box_depth, self.gap = box_w, box_depth, gap
        self.tol, self.band_lo = line_tol, band_lo
        self.band_hi = box_depth + 0.035
        self.min_wall_pts, self.max_lines, self.iters = min_wall_pts, max_lines, iters
        self.rng = np.random.default_rng(seed)
 
    def _ransac(self, pts):
        n = len(pts)
        if n < self.min_wall_pts:
            return None
        best, best_cnt = None, 0
        for _ in range(self.iters):
            i, j = self.rng.choice(n, 2, replace=False)
            dv = pts[j] - pts[i]
            L = float(np.hypot(dv[0], dv[1]))
            if L < 0.30:                        # pares muy juntos = línea inestable
                continue
            dv /= L
            nv = np.array([-dv[1], dv[0]])
            mask = np.abs((pts - pts[i]) @ nv) < self.tol
            c = int(mask.sum())
            if c > best_cnt:
                best_cnt, best = c, mask
        if best is None or best_cnt < self.min_wall_pts:
            return None
        inl = pts[best]
        m = inl.mean(axis=0)
        _, _, vt = np.linalg.svd(inl - m, full_matrices=False)
        d = vt[0]
        return best, m, d, np.array([-d[1], d[0]])
 
    def _find_pair(self, pts, m, d, nrm, ang_inc):
        """Busca las dos cajas sobresaliendo de la pared (m, d, nrm)."""
        s = 1.0 if float(-m @ nrm) > 0 else -1.0     # normal hacia el robot
        n = s * nrm
        rel = pts - m
        u, vp = rel @ d, s * (rel @ nrm)
        idx = np.nonzero((vp > self.band_lo) & (vp < self.band_hi))[0]
        if len(idx) < 4:
            return None
        order = idx[np.argsort(u[idx])]
        ub = u[order]
        rb = np.hypot(pts[order, 0], pts[order, 1])
        thr = np.clip(2.5 * rb[:-1] * ang_inc, 0.025, 0.055)
        cuts = np.nonzero(np.diff(ub) > thr)[0] + 1
        clusters = []
        for seg in np.split(np.arange(len(ub)), cuts):
            if len(seg) < 3:
                continue
            w = ub[seg[-1]] - ub[seg[0]]
            if 0.03 <= w <= self.box_w + 0.05:
                clusters.append((ub[seg[0]], ub[seg[-1]], float(rb[seg].mean())))
 
        best, best_score = None, 1e9
        for a, b in zip(clusters[:-1], clusters[1:]):
            gap_meas = b[0] - a[1]
            spacing = 0.5 * (a[2] + b[2]) * ang_inc
            tol = 0.02 + 2.5 * spacing
            err = abs(gap_meas - self.gap)
            if err > tol:
                continue
            wa, wb = a[1] - a[0], b[1] - b[0]
            if abs(wa - wb) > 0.05:
                continue
            score = err + 0.5 * abs(wa - self.box_w) + 0.5 * abs(wb - self.box_w)
            if score < best_score:
                if 0.06 <= wa <= 0.10 and 0.06 <= wb <= 0.10:
                    uc = 0.25 * (a[0] + a[1] + b[0] + b[1])   # 4 bordes
                else:
                    uc = 0.5 * (a[1] + b[0])                   # solo hueco
                best, best_score = (m + uc * d, n, gap_meas), score
        if best is None:
            best = self._find_coarse(ub, rb, ang_inc, m, d, n)
        return best
 
    def _find_coarse(self, ub, rb, ang_inc, m, d, n):
        """Detección a larga distancia (> ~3 m): con 0.5 deg de resolución, a 6 m
        los puntos quedan a 5 cm y el hueco de 9.5 cm ya no se resuelve. Se busca
        entonces el marcador COMPLETO (cajas+hueco = 25.5 cm) como un solo bulto
        que sobresale de la pared. Da un centro con error ~ +-2 cm, suficiente
        para llegar al pre-dock, donde el detector fino toma el control."""
        full = 2 * self.box_w + self.gap
        spacing = rb * ang_inc
        thr = self.gap + 1.5 * spacing[:-1]          # une ambas cajas
        cuts = np.nonzero(np.diff(ub) > thr)[0] + 1
        best, best_score = None, 1e9
        for seg in np.split(np.arange(len(ub)), cuts):
            if len(seg) < 3:
                continue
            ext = ub[seg[-1]] - ub[seg[0]]
            sp = float(rb[seg].mean() * ang_inc)
            if full - 2 * sp - 0.03 <= ext <= full + 0.03:
                score = abs(ext - (full - sp))
                if score < best_score:
                    uc = 0.5 * (ub[seg[0]] + ub[seg[-1]])
                    best, best_score = (m + uc * d, n, ext), score
        return best
 
    def detect(self, pts, ang_inc):
        """-> (centro_sobre_pared, normal_hacia_robot, hueco_medido) o None."""
        if pts is None or len(pts) < self.min_wall_pts:
            return None
        remaining = pts
        for _ in range(self.max_lines):
            fit = self._ransac(remaining)
            if fit is None:
                break
            mask, m, d, nrm = fit
            res = self._find_pair(pts, m, d, nrm, ang_inc)
            if res is not None:
                return res
            remaining = remaining[~mask]
        return None
 
 
# --------------------------------------------------------------------------- #
# Nodo
# --------------------------------------------------------------------------- #
class AutoDockerNode(Node):
 
    def __init__(self):
        super().__init__('auto_docker_node',
                         parameter_overrides=[Parameter(
                             'use_sim_time', Parameter.Type.BOOL, True)])
        try:
            self.declare_parameter('use_sim_time', True)
        except ParameterAlreadyDeclaredException:
            pass
 
        def P(name, default):
            self.declare_parameter(name, default)
            return self.get_parameter(name).value
 
        # Medidas del marcador (README oficial)
        self.detector = MarkerDetector(
            box_w=P('box_width', 0.08), box_depth=P('box_depth', 0.08),
            gap=P('box_gap', 0.095), line_tol=P('line_tol', 0.012))
        self.max_range = P('max_use_range', 7.5)
        # Montaje del LiDAR: solo se usa si TF aún no está disponible
        self.fb_lidar = (P('lidar_x', -0.050502), P('lidar_y', -0.017960),
                         P('lidar_yaw', math.pi))
        self.predock = P('predock_dist', 0.8)
        self.predock_tol = P('predock_tol', 0.04)
        self.align_tol = math.radians(P('align_tol_deg', 1.5))
        self.v_go = P('v_go_max', 0.30)
        self.v_far, self.v_mid, self.v_near = P('v_dock_far', 0.08), P('v_dock_mid', 0.05), P('v_dock_near', 0.03)
        self.lookahead = P('lookahead', 0.15)
        self.k_head = P('k_heading', 2.5)
        self.est_alpha = P('est_alpha', 0.35)
        self.debug_markers = P('debug_markers', True)
 
        self.state = State.SEARCH
        self.state_t0 = self.get_clock().now()
        self.pose = np.zeros(2)
        self.yaw = 0.0
        self.have_odom = False
        self.est_c = self.est_n = None
        self.reject = 0
        self.hit_times = []             # instantes de detecciones recientes
        self.last_detect = None
        self.front_clear = 5.0          # distancia libre delante (base_link)
        self.free_bearing = 0.0         # rumbo (base_link) del mayor espacio libre
        self.explore_yaw = 0.0
        self.last_scan = None
        self.contact_checked = False
        self.n_pts = 0
        self.last_log = self.get_clock().now()
        self._laser_tf = None
 
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
 
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(DockStatus, '/dock_status', self.dock_cb, qos_profile_sensor_data)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_marker = self.create_publisher(Marker, '/dock_marker', 10)
        self.create_timer(0.05, self.control_loop)
        self.get_logger().info('auto_docker listo. Estado: SEARCH')
 
    # ---------------- utilidades ---------------- #
    def set_state(self, new):
        if new != self.state:
            self.get_logger().info(f'FSM: {self.state.name} -> {new.name}')
            self.state, self.state_t0 = new, self.get_clock().now()
            self.contact_checked = False
 
    def age(self, t):
        return float('inf') if t is None else (self.get_clock().now() - t).nanoseconds * 1e-9
 
    def publish(self, v=0.0, w=0.0):
        m = Twist()
        m.linear.x, m.angular.z = float(v), float(w)
        self.pub_cmd.publish(m)
 
    def lidar_to_base(self, frame):
        """(tx, ty, yaw) laser->base_link. Usa TF; si no hay, el montaje del README."""
        if self._laser_tf is not None:
            return self._laser_tf
        try:
            t = self.tf_buffer.lookup_transform('base_link', frame, Time())
            q = t.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            self._laser_tf = (t.transform.translation.x, t.transform.translation.y, yaw)
            self.get_logger().info(f'TF {frame}->base_link: {self._laser_tf}')
            return self._laser_tf
        except Exception:
            self.get_logger().warn('TF laser->base_link no disponible; uso montaje del README',
                                   throttle_duration_sec=5.0)
            return self.fb_lidar
 
    # ---------------- callbacks ---------------- #
    def odom_cb(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.pose = np.array([p.x, p.y])
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.have_odom = True
 
    def dock_cb(self, msg):
        if msg.is_docked and self.state != State.DOCKED:
            self.publish(0.0, 0.0)
            self.set_state(State.DOCKED)
            self.get_logger().info('*** ACOPLADO: is_docked == true. cmd_vel = 0 ***')
 
    def scan_cb(self, msg):
        self.last_scan = self.get_clock().now()
        if not self.have_odom or self.state == State.DOCKED:
            return
        try:
            r = np.asarray(msg.ranges, dtype=np.float64)
            ang = msg.angle_min + np.arange(r.size) * msg.angle_increment
            ok = (np.isfinite(r) & (r >= max(msg.range_min, 0.05))
                  & (r <= min(msg.range_max, self.max_range)))
            r, ang = r[ok], ang[ok]
            self.n_pts = int(r.size)
            tx, ty, yaw = self.lidar_to_base(msg.header.frame_id or 'laser_link')
            xl, yl = r * np.cos(ang), r * np.sin(ang)
            c, s = math.cos(yaw), math.sin(yaw)
            pts = np.column_stack((tx + c * xl - s * yl, ty + s * xl + c * yl))
            self.update_free_space(pts)
            res = self.detector.detect(pts, msg.angle_increment)
        except Exception as e:
            self.get_logger().warn(f'Error de percepción: {e}', throttle_duration_sec=2.0)
            res = None
        if res is None:
            return
        now = self.get_clock().now()
        self.hit_times = [t for t in self.hit_times if self.age(t) < 1.5] + [now]
        self.last_detect = now
        c_b, n_b, _ = res
        self.update_estimate(c_b, n_b)
        if self.debug_markers:
            self.publish_marker(c_b, n_b)
 
    def update_free_space(self, pts):
        """Distancia libre al frente y rumbo del mayor espacio libre, en base_link
        (las lecturas ya están rotadas: OJO, el ángulo crudo del scan apunta hacia
        ATRÁS porque el LiDAR va girado 180 deg)."""
        if len(pts) == 0:
            return
        x, y = pts[:, 0], pts[:, 1]
        corridor = (x > 0.0) & (np.abs(y) < 0.25)          # ancho del robot + margen
        self.front_clear = float(x[corridor].min()) if np.any(corridor) else 9.0
        ang, rr = np.arctan2(y, x), np.hypot(x, y)
        nb = 36
        bins = ((ang + math.pi) / (2 * math.pi) * nb).astype(int) % nb
        rng_bin = np.full(nb, self.max_range)
        np.minimum.at(rng_bin, bins, rr)
        sm = np.minimum(np.minimum(np.roll(rng_bin, 1), rng_bin), np.roll(rng_bin, -1))
        centers = -math.pi + (np.arange(nb) + 0.5) * 2 * math.pi / nb
        score = sm - 0.05 * np.abs(centers) / math.pi        # leve preferencia por girar poco
        self.free_bearing = float(centers[int(np.argmax(score))])
 
    def publish_marker(self, c_b, n_b):
        m = Marker()
        m.header.frame_id = 'base_link'
        m.header.stamp = self.get_clock().now().to_msg()
        m.type, m.action = Marker.ARROW, Marker.ADD
        m.scale.x, m.scale.y, m.scale.z = 0.4, 0.03, 0.03
        m.color.a, m.color.g = 1.0, 1.0
        m.pose.position.x, m.pose.position.y = float(c_b[0]), float(c_b[1])
        a = math.atan2(n_b[1], n_b[0])
        m.pose.orientation.z, m.pose.orientation.w = math.sin(a / 2), math.cos(a / 2)
        self.pub_marker.publish(m)
 
    # ---------------- estimación (frame odom) ---------------- #
    def update_estimate(self, c_b, n_b):
        c_o = self.pose + rot(c_b, self.yaw)
        n_o = rot(n_b, self.yaw)
        if self.est_c is None:
            self.est_c, self.est_n = c_o, n_o
            return
        if float(np.linalg.norm(c_o - self.est_c)) > 0.25:
            self.reject += 1
            if self.reject >= 5:
                self.est_c, self.est_n, self.reject = c_o, n_o, 0
            return
        self.reject = 0
        a = self.est_alpha
        self.est_c = (1 - a) * self.est_c + a * c_o
        n = (1 - a) * self.est_n + a * n_o
        self.est_n = n / np.linalg.norm(n)
 
    def relative(self):
        """c_b, n_b, d_w (centro robot->pared), e_y (lateral), phi (rumbo)."""
        if self.est_c is None:
            return None
        c_b = rot(self.est_c - self.pose, -self.yaw)
        n_b = rot(self.est_n, -self.yaw)
        t_b = np.array([-n_b[1], n_b[0]])
        return (c_b, n_b, -float(c_b @ n_b), -float(c_b @ t_b),
                math.atan2(-n_b[1], -n_b[0]))
 
    # ---------------- FSM ---------------- #
    def control_loop(self):
        if self.state == State.DOCKED:
            self.publish(0.0, 0.0)
            return
        if not self.have_odom or self.age(self.last_scan) > 1.0:
            self.publish(0.0, 0.0)
            return
        rel = self.relative()
        lost = self.age(self.last_detect) > 3.0
 
        seen = rel is not None and len(self.hit_times) >= 3 and self.age(self.last_detect) < 0.5
 
        if self.state == State.SEARCH:
            if seen:
                self.publish(0.0, 0.0)
                self.set_state(State.GO_TO_PREDOCK)
            elif self.age(self.state_t0) > 11.0:          # ~1 vuelta completa sin verlo
                self.explore_yaw = wrap(self.yaw + self.free_bearing)
                self.set_state(State.EXPLORE)
            else:
                self.publish(0.0, 0.6)
            return
 
        if self.state == State.EXPLORE:
            err = wrap(self.explore_yaw - self.yaw)
            if seen:
                self.publish(0.0, 0.0)
                self.set_state(State.GO_TO_PREDOCK)
            elif abs(err) > 0.15 and self.age(self.state_t0) < 6.0:
                self.publish(0.0, math.copysign(max(abs(1.5 * err), 0.2), err) if abs(err) < 0.5
                             else math.copysign(0.8, err))
            elif self.front_clear < 0.6 or self.age(self.state_t0) > 14.0:
                self.publish(0.0, 0.0)
                self.set_state(State.SEARCH)
            else:
                self.publish(0.20, clamp(1.0 * err, -0.5, 0.5))
            return
 
        if rel is None:
            self.set_state(State.SEARCH)
            return
        c_b, n_b, d_w, e_y, phi = rel
        self._status(d_w, e_y, phi)
 
        if self.state == State.GO_TO_PREDOCK:
            if lost:
                self.set_state(State.SEARCH)
                return
            p = c_b + self.predock * n_b
            dist, bearing = float(np.hypot(*p)), math.atan2(p[1], p[0])
            if dist < self.predock_tol:
                self.publish(0.0, 0.0)
                self.set_state(State.ALIGN_HEADING)
            elif abs(bearing) > 0.35:
                w = clamp(2.0 * bearing, -1.0, 1.0)
                self.publish(0.0, math.copysign(max(abs(w), 0.15), w))
            else:
                v = clamp(0.8 * dist, 0.06, self.v_go) * math.cos(bearing)
                self.publish(v, clamp(2.0 * bearing, -1.0, 1.0))
 
        elif self.state == State.ALIGN_HEADING:
            if lost:
                self.set_state(State.SEARCH)
                return
            if abs(phi) < self.align_tol:
                self.publish(0.0, 0.0)
                self.set_state(State.DOCKING_APPROACH if abs(e_y) < 0.05
                               else State.GO_TO_PREDOCK)
            else:
                w = clamp(2.5 * phi, -1.0, 1.0)
                self.publish(0.0, math.copysign(max(abs(w), 0.10), w))
 
        elif self.state == State.DOCKING_APPROACH:
            if lost and d_w > 0.5:
                self.set_state(State.SEARCH)
                return
            if (abs(e_y) > 0.08 or abs(phi) > math.radians(10)) and d_w > 0.5:
                self.publish(0.0, 0.0)
                self.set_state(State.GO_TO_PREDOCK)
                return
            if d_w < 0.45 and not self.contact_checked:       # antes de entrar bajo las cajas
                self.contact_checked = True
                self.get_logger().info(
                    f'Entrada: e_y={e_y * 1000:.1f} mm  phi={math.degrees(phi):.2f} deg  d_w={d_w:.3f}')
                if abs(e_y) > 0.02 or abs(phi) > math.radians(6.0):
                    self.set_state(State.RECOVER)
                    return
            if d_w < 0.20 or self.age(self.state_t0) > 60.0:   # se pasó o se atascó
                self.set_state(State.RECOVER)
                return
            v = self.v_far if d_w > 0.5 else (self.v_mid if d_w > 0.36 else self.v_near)
            if abs(e_y) > 0.02 or abs(phi) > math.radians(4.0):
                v *= 0.5
            w = clamp(self.k_head * (phi + math.atan2(e_y, self.lookahead)), -0.4, 0.4)
            self.publish(max(v, 0.02), w)         # nunca frenar por distancia
 
        elif self.state == State.RECOVER:
            if d_w > 0.70:
                self.publish(0.0, 0.0)
                self.set_state(State.GO_TO_PREDOCK)
            else:
                self.publish(-0.10, 0.0)
 
    def _status(self, d_w, e_y, phi):
        if self.age(self.last_log) > 1.0:
            self.last_log = self.get_clock().now()
            self.get_logger().info(
                f'[{self.state.name}] pts={self.n_pts} hits={len(self.hit_times)} '
                f'd_w={d_w:.3f} m  e_y={e_y * 1000:.1f} mm  phi={math.degrees(phi):.2f} deg')
 
 
def main(args=None):
    rclpy.init(args=args)
    node = AutoDockerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()
