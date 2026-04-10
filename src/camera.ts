// camera.ts — Arcball + free-fly camera
// Task 4: Arcball controls (quaternion-based, orbits around scene center)
// Task 9: Zoom via scroll wheel

import type { Mat4, Vec3 } from "./math";
import { mat4, vec3, quat } from "./math";

export class ArcballCamera {
  /** World-space point we orbit around */
  target: Vec3 = [0, 0, 0];

  /** Distance from target */
  radius = 5;
  minRadius = 0.1;
  maxRadius = 5000;

  /** Arcball orientation as quaternion */
  private orientation = quat.identity();



  // ── Drag state
  private dragging = false;
  private lastNDC: [number, number] = [0, 0];
  private canvas!: HTMLCanvasElement;

  // ── WASD state (secondary free-fly mode)
  moveSpeed = 1.5;
  private keys = new Set<string>();

  bind(canvas: HTMLCanvasElement) {
    this.canvas = canvas;

    canvas.addEventListener("mousedown", e => {
      if (e.button === 0) {
        this.dragging = true;
        this.lastNDC = this.toNDC(e.clientX, e.clientY);
        e.preventDefault();
      }
    });

    window.addEventListener("mouseup",   e => { if (e.button === 0) this.dragging = false; });

    window.addEventListener("mousemove", e => {
      if (!this.dragging) return;
      const cur = this.toNDC(e.clientX, e.clientY);
      this.onDrag(this.lastNDC, cur);
      this.lastNDC = cur;
    });

    // Zoom via scroll
    canvas.addEventListener("wheel", e => {
      e.preventDefault();
      const factor = 1 + e.deltaY * 0.001;
      this.radius = Math.max(this.minRadius, Math.min(this.maxRadius, this.radius * factor));
    }, { passive: false });

    // Touch support (single-finger = orbit, pinch = zoom)
    let lastTouches: TouchList | null = null;
    canvas.addEventListener("touchstart", e => { lastTouches = e.touches; e.preventDefault(); }, { passive: false });
    canvas.addEventListener("touchmove",  e => {
      e.preventDefault();
      if (!lastTouches) return;
      if (e.touches.length === 1 && lastTouches.length === 1) {
        const cur  = this.toNDC(e.touches[0].clientX, e.touches[0].clientY);
        const prev = this.toNDC(lastTouches[0].clientX, lastTouches[0].clientY);
        this.onDrag(prev, cur);
      } else if (e.touches.length === 2 && lastTouches.length === 2) {
        const d0 = Math.hypot(lastTouches[0].clientX - lastTouches[1].clientX,
                              lastTouches[0].clientY - lastTouches[1].clientY);
        const d1 = Math.hypot(e.touches[0].clientX - e.touches[1].clientX,
                              e.touches[0].clientY - e.touches[1].clientY);
        const factor = d0 / (d1 || 1);
        this.radius = Math.max(this.minRadius, Math.min(this.maxRadius, this.radius * factor));
      }
      lastTouches = e.touches;
    }, { passive: false });
    canvas.addEventListener("touchend", () => { lastTouches = null; });

    window.addEventListener("keydown", e => this.keys.add(e.key));
    window.addEventListener("keyup",   e => this.keys.delete(e.key));
  }

  /** Reset to face the object straight-on */
  reset(center: Vec3, radius: number) {
    this.target     = [...center] as Vec3;
    this.radius     = radius * 2.5;
    this.maxRadius  = radius * 20;
    this.minRadius  = radius * 0.1;
    this.orientation = quat.identity();

  }

  update(dt: number) {
    // WASD nudges the target (panning)
    const step = this.moveSpeed * dt * this.radius * 0.3;
    const viewMat = this.getViewMatrix();
    const right: Vec3  = [viewMat[0], viewMat[4], viewMat[8] ];
    const up: Vec3     = [viewMat[1], viewMat[5], viewMat[9] ];
    const forward: Vec3 = [-viewMat[2], -viewMat[6], -viewMat[10]];

    if (this.keys.has("a") || this.keys.has("ArrowLeft"))  this.target = vec3.add(this.target, vec3.scale(right,    -step));
    if (this.keys.has("d") || this.keys.has("ArrowRight")) this.target = vec3.add(this.target, vec3.scale(right,     step));
    if (this.keys.has("w") || this.keys.has("ArrowUp"))    this.target = vec3.add(this.target, vec3.scale(forward,   step));
    if (this.keys.has("s") || this.keys.has("ArrowDown"))  this.target = vec3.add(this.target, vec3.scale(forward,  -step));
    if (this.keys.has("q")) this.target = vec3.add(this.target, vec3.scale(up, -step));
    if (this.keys.has("e")) this.target = vec3.add(this.target, vec3.scale(up,  step));
    if (this.keys.has("r")) this.reset(this.target, this.radius / 2.5);
  }

  getViewMatrix(): Mat4 {
    const rotMat = quat.toMat4(this.orientation);
    // Camera is at `target + R * (0, 0, radius)`
    const camOffset = [
      rotMat[8]  * this.radius,
      rotMat[9]  * this.radius,
      rotMat[10] * this.radius,
    ] as Vec3;
    const eye = vec3.add(this.target, camOffset);
    const up: Vec3 = [rotMat[4], rotMat[5], rotMat[6]];
    return mat4.lookAt(eye, this.target, up);
  }

  /** Camera world position (for lighting) */
  getPosition(): Vec3 {
    const v = this.getViewMatrix();
    // eye = -R^T * t where t is translation column of view matrix
    return [
      -(v[0]*v[12] + v[1]*v[13] + v[2]*v[14]),
      -(v[4]*v[12] + v[5]*v[13] + v[6]*v[14]),
      -(v[8]*v[12] + v[9]*v[13] + v[10]*v[14]),
    ];
  }

  // ── Private helpers ─────────────────────────────────────────────────────────

  /** Convert pixel coords to Normalized Device Coordinates [-1,1]×[-1,1] */
  private toNDC(px: number, py: number): [number, number] {
    const rect = this.canvas.getBoundingClientRect();
    return [
       (px - rect.left)  / rect.width  * 2 - 1,
      -((py - rect.top) / rect.height * 2 - 1), // flip Y
    ];
  }

  /**
   * Map a 2-D NDC point to a point on the virtual arcball sphere.
   * Points outside the sphere are projected onto the hyperbolic sheet.
   */
  private toSphere(ndc: [number, number]): Vec3 {
    const [x, y] = ndc;
    const r2 = x*x + y*y;
    if (r2 <= 0.5) {
      return vec3.normalize([x, y, Math.sqrt(1 - r2)]);
    } else {
      return vec3.normalize([x, y, 0.5 / Math.sqrt(r2)]);
    }
  }

  private onDrag(from: [number, number], to: [number, number]) {
    const v0 = this.toSphere(from);
    const v1 = this.toSphere(to);
    const axis = vec3.cross(v0, v1);
    const axLen = vec3.length(axis);
    if (axLen < 1e-8) return;
    const angle = Math.asin(Math.min(1, axLen)) * 2;
    const dq = quat.fromAxisAngle(vec3.normalize(axis), angle);
    this.orientation = quat.normalize(quat.multiply(dq, this.orientation));
  }
}
