/// <reference types="@webgpu/types" />

import "./style.css";
import shaderCode from "./shader.wgsl?raw";
import { ArcballCamera } from "./camera";
import { mat4 } from "./math";
import type { Vec3 } from "./math";
import { loadOBJ } from "./objLoader";
import type { IndexedMesh } from "./objLoader";
import { loadTexture, createSampler } from "./texture";
import { initGUI } from "./gui";

// ─── WebGPU init ───────────────────────────────────────────────────────────────
if (!navigator.gpu) throw new Error("WebGPU not supported");
const canvas = document.querySelector("#gfx-main") as HTMLCanvasElement;

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error("No GPU adapter found");
const device = await adapter.requestDevice();
const context = canvas.getContext("webgpu")!;
const format  = navigator.gpu.getPreferredCanvasFormat();

// ─── Depth texture ────────────────────────────────────────────────────────────
let depthTexture: GPUTexture | null = null;
function resize() {
  canvas.width  = Math.max(1, Math.floor(window.innerWidth  * devicePixelRatio));
  canvas.height = Math.max(1, Math.floor(window.innerHeight * devicePixelRatio));
  context.configure({ device, format, alphaMode: "premultiplied" });
  depthTexture?.destroy();
  depthTexture = device.createTexture({
    size: [canvas.width, canvas.height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
}
resize();
window.addEventListener("resize", resize);

// ─── Texture ──────────────────────────────────────────────────────────────────
const sampler    = createSampler(device);
const defaultTex = await loadTexture(device, null); // checkerboard

// ─── Uniform layout (288 bytes) ───────────────────────────────────────────────
// 0:   mvp        mat4   (64 B)
// 64:  model      mat4   (64 B)
// 128: normalMat  mat4   (64 B)
// 192: lightPos   vec3   + f32 pad
// 208: lightColor vec3   + f32 pad
// 224: ambient f32, diffuse f32, specular f32, shininess f32
// 240: camPos vec3, model_id u32
// 256: objectColor vec3, time f32
// 272: wireThick f32, _p2 f32, _p3 f32, _p4 f32
const UNIFORM_SIZE = 288;

// ─── Pipeline ─────────────────────────────────────────────────────────────────
const shader = device.createShaderModule({ label: "Main Shader", code: shaderCode });

const pipeline = device.createRenderPipeline({
  label: "Main Pipeline",
  layout: "auto",
  vertex: {
    module: shader,
    entryPoint: "vs_main",
    buffers: [
      {
        // Buffer 0: position(3) + normal(3) + uv(2) = 8 floats
        arrayStride: 8 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0,     format: "float32x3" }, // position
          { shaderLocation: 1, offset: 3 * 4, format: "float32x3" }, // normal
          { shaderLocation: 2, offset: 6 * 4, format: "float32x2" }, // uv
        ],
      },
      {
        // Buffer 1: barycentric coords (3 floats) per vertex
        arrayStride: 3 * 4,
        attributes: [
          { shaderLocation: 3, offset: 0, format: "float32x3" }, // barycentric
        ],
      },
    ],
  },
  fragment: { module: shader, entryPoint: "fs_main", targets: [{ format }] },
  primitive: { topology: "triangle-list", cullMode: "back" },
  depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
});

// ─── Scene object definition ──────────────────────────────────────────────────
interface SceneObject {
  uid:          number;   // unique per-instance ID
  name:         string;
  mesh:         IndexedMesh;
  vertBuf:      GPUBuffer;
  idxBuf:       GPUBuffer;
  baryBuf:      GPUBuffer;
  uniformBuf:   GPUBuffer;
  bindGroup:    GPUBindGroup;
  color:        [number, number, number];
  modelMatrix:  Float32Array;
  shadingMode:  number;  // 0=flat,1=gouraud,2=phong,3=wireframe,4=normbuf,5=texture
  wireThick:    number;
}

let sceneObjects: SceneObject[] = [];
let _uidCounter = 0;

// ─── Barycentric buffer builder ───────────────────────────────────────────────
function buildBarycentricBuffer(indexCount: number): GPUBuffer {
  // Each triangle gets vertices with bary coords [1,0,0],[0,1,0],[0,0,1]
  const triCount = indexCount / 3;
  const data = new Float32Array(triCount * 9);
  for (let t = 0; t < triCount; t++) {
    const base = t * 9;
    data[base + 0] = 1; data[base + 1] = 0; data[base + 2] = 0;
    data[base + 3] = 0; data[base + 4] = 1; data[base + 5] = 0;
    data[base + 6] = 0; data[base + 7] = 0; data[base + 8] = 1;
  }
  const buf = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, data as any);
  return buf;
}

// ─── Create GPU buffers for a mesh ───────────────────────────────────────────
function buildGPUBuffers(mesh: IndexedMesh) {
  const vertBuf = device.createBuffer({
    size: mesh.vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertBuf, 0, mesh.vertices as any);

  const idxBuf = device.createBuffer({
    size: Math.max(mesh.indices.byteLength, 4),
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(idxBuf, 0, mesh.indices as any);

  const baryBuf = buildBarycentricBuffer(mesh.indices.length);

  return { vertBuf, idxBuf, baryBuf };
}

// ─── Build bind group for one object ─────────────────────────────────────────
function buildBindGroup(uniformBuf: GPUBuffer, tex: GPUTexture): GPUBindGroup {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: sampler },
      { binding: 2, resource: tex.createView() },
    ],
  });
}

// ─── Compute model matrix so object fits in a normalized unit cube ──────────
function computeModelMatrix(mesh: IndexedMesh, offsetX: number, offsetY = 0): Float32Array {
  const { center, radius } = mesh.bounds;
  const scale = 1 / radius;
  const t = mat4.translation(
    -center[0] * scale + offsetX,
    -center[1] * scale + offsetY,
    -center[2] * scale,          // Z always 0 — objects stay in front of camera
  );
  const s = mat4.scaling(scale, scale, scale);
  return mat4.multiply(t, s);
}

// ─── Add a loaded mesh to the scene ──────────────────────────────────────────
function addObject(
  name: string,
  mesh: IndexedMesh,
  offsetX: number,
  color: [number, number, number],
  tex: GPUTexture = defaultTex,
): SceneObject {
  const { vertBuf, idxBuf, baryBuf } = buildGPUBuffers(mesh);

  const uniformBuf = device.createBuffer({
    size: UNIFORM_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bindGroup = buildBindGroup(uniformBuf, tex);
  const modelMatrix = computeModelMatrix(mesh, offsetX);
  const uid = _uidCounter++;

  return { uid, name, mesh, vertBuf, idxBuf, baryBuf, uniformBuf, bindGroup, color, modelMatrix, shadingMode: 2, wireThick: 1.5 };
}

// ─── Camera ──────────────────────────────────────────────────────────────────
const camera = new ArcballCamera();
camera.bind(canvas);
camera.reset([0, 0, 0], 3);

// ─── GUI state ────────────────────────────────────────────────────────────────
export const gui = {
  ambient:      0.12,
  diffuse:      0.75,
  specular:     0.60,
  shininess:    32,
  lightX:       2.0,
  lightY:       5.0,
  lightZ:       2.0,
  autoRotLight: true,
  lightColor:   "#ffffff",
  wireThick:    1.5,
};

// ─── Procedural geometry → IndexedMesh ───────────────────────────────────────
function makeCubeMesh(): IndexedMesh {
  type FaceDef = { n: Vec3; verts: number[][] };
  const faces: FaceDef[] = [
    { n: [ 0, 0, 1], verts: [[-1,-1, 1,0,1],[1,-1, 1,1,1],[1, 1, 1,1,0],[-1, 1, 1,0,0]] },
    { n: [ 0, 0,-1], verts: [[ 1,-1,-1,0,1],[-1,-1,-1,1,1],[-1, 1,-1,1,0],[ 1, 1,-1,0,0]] },
    { n: [-1, 0, 0], verts: [[-1,-1,-1,0,1],[-1,-1, 1,1,1],[-1, 1, 1,1,0],[-1, 1,-1,0,0]] },
    { n: [ 1, 0, 0], verts: [[ 1,-1, 1,0,1],[ 1,-1,-1,1,1],[ 1, 1,-1,1,0],[ 1, 1, 1,0,0]] },
    { n: [ 0, 1, 0], verts: [[-1, 1, 1,0,1],[ 1, 1, 1,1,1],[ 1, 1,-1,1,0],[-1, 1,-1,0,0]] },
    { n: [ 0,-1, 0], verts: [[-1,-1,-1,0,1],[ 1,-1,-1,1,1],[ 1,-1, 1,1,0],[-1,-1, 1,0,0]] },
  ];
  const verts: number[] = [];
  const idxs:  number[] = [];
  for (const face of faces) {
    const base = verts.length / 8;
    for (const v of face.verts) verts.push(v[0],v[1],v[2], ...face.n, v[3],v[4]);
    idxs.push(base,base+1,base+2, base,base+2,base+3);
  }
  const center: Vec3 = [0,0,0];
  return {
    vertices: new Float32Array(verts),
    indices:  new Uint32Array(idxs),
    faceNormals: new Float32Array(6*3),
    bounds: { min:[-1,-1,-1], max:[1,1,1], center, radius: Math.sqrt(3) },
    triangleCount: idxs.length/3,
    vertexCount:   verts.length/8,
  };
}

function makeSphereMesh(stacks = 48, slices = 64): IndexedMesh {
  const verts: number[] = [];
  const idxs:  number[] = [];
  for (let i = 0; i <= stacks; i++) {
    const phi = Math.PI * i / stacks;
    for (let j = 0; j <= slices; j++) {
      const theta = 2 * Math.PI * j / slices;
      const nx = Math.sin(phi) * Math.cos(theta);
      const ny = Math.cos(phi);
      const nz = Math.sin(phi) * Math.sin(theta);
      verts.push(nx, ny, nz, nx, ny, nz, j/slices, i/stacks);
    }
  }
  for (let i = 0; i < stacks; i++) {
    for (let j = 0; j < slices; j++) {
      const a = i*(slices+1)+j, b = a+slices+1;
      idxs.push(a,b,a+1, a+1,b,b+1);
    }
  }
  const center: Vec3 = [0,0,0];
  return {
    vertices: new Float32Array(verts),
    indices:  new Uint32Array(idxs),
    faceNormals: new Float32Array(0),
    bounds: { min:[-1,-1,-1], max:[1,1,1], center, radius: 1 },
    triangleCount: idxs.length/3,
    vertexCount:   verts.length/8,
  };
}

// ─── Load models ─────────────────────────────────────────────────────────────
async function loadBeacon() {
  const mesh = await loadOBJ("/models/KAUST_Beacon.obj");
  return addObject("Beacon", mesh, 0, [0.9, 0.6, 0.2]);
}
async function loadTeapot() {
  const mesh = await loadOBJ("/models/teapot.obj");
  return addObject("Teapot", mesh, 0, [0.3, 0.7, 1.0]);
}
function spawnCube()   { return addObject("Cube",   makeCubeMesh(),   0, [0.8, 0.3, 0.9]); }
function spawnSphere() { return addObject("Sphere", makeSphereMesh(), 0, [0.2, 0.9, 0.5]); }

// ─── GUI integration ─────────────────────────────────────────────────────────
initGUI({
  onLoadBeacon: async () => {
    const obj = await loadBeacon();
    sceneObjects.push(obj);
    repositionObjects();
    fitCamera();
  },
  onLoadTeapot: async () => {
    const obj = await loadTeapot();
    sceneObjects.push(obj);
    repositionObjects();
    fitCamera();
  },
  onLoadCube: () => {
    sceneObjects.push(spawnCube());
    repositionObjects();
    fitCamera();
  },
  onLoadSphere: () => {
    sceneObjects.push(spawnSphere());
    repositionObjects();
    fitCamera();
  },
  onRemoveObject: (uid: number) => {
    sceneObjects = sceneObjects.filter(o => o.uid !== uid);
    repositionObjects();
    fitCamera();
  },
  onShadingChange: (uid: number, mode: number) => {
    const obj = sceneObjects.find(o => o.uid === uid);
    if (obj) obj.shadingMode = mode;
  },
  onColorChange: (uid: number, hex: string) => {
    const obj = sceneObjects.find(o => o.uid === uid);
    if (obj) {
      const n = parseInt(hex.slice(1), 16);
      obj.color = [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255];
    }
  },
  onWireThickChange: (v: number) => { gui.wireThick = v; },
  getObjects: () => sceneObjects.map(o => ({ uid: o.uid, name: o.name })),
  gui,
});

/**
 * Arrange N objects in a 2D grid (X and Y), Z stays fixed at 0.
 * Grid columns = ceil(sqrt(N)), rows fill as needed.
 */
function repositionObjects() {
  const count = sceneObjects.length;
  if (count === 0) return;

  const cols = Math.ceil(Math.sqrt(count));
  const gap  = 2.5;

  // Compute grid extents for centering
  const totalCols = Math.min(count, cols);
  const totalRows = Math.ceil(count / cols);
  const gridW = (totalCols - 1) * gap;
  const gridH = (totalRows - 1) * gap;

  for (let i = 0; i < count; i++) {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const offsetX = -gridW / 2 + col * gap;
    const offsetY =  gridH / 2 - row * gap; // top row is positive Y
    sceneObjects[i].modelMatrix = computeModelMatrix(sceneObjects[i].mesh, offsetX, offsetY);
  }
}

/** Fit camera radius to frame the entire grid, keeping Z fixed */
function fitCamera() {
  if (sceneObjects.length === 0) return;
  const count  = sceneObjects.length;
  const cols   = Math.ceil(Math.sqrt(count));
  const rows   = Math.ceil(count / cols);
  const gap    = 2.5;
  // Half-diagonal of the grid + 1 unit margin per cell
  const halfW  = ((cols - 1) * gap) / 2 + 1.5;
  const halfH  = ((rows - 1) * gap) / 2 + 1.5;
  const radius = Math.sqrt(halfW * halfW + halfH * halfH) + 1;
  camera.reset([0, 0, 0], radius);
}

// Start with Beacon loaded by default
const defaultObj = await loadBeacon();
sceneObjects.push(defaultObj);
fitCamera();

// ─── Render loop ──────────────────────────────────────────────────────────────
const uArrayBuf = new ArrayBuffer(UNIFORM_SIZE);
const uData     = new Float32Array(uArrayBuf);
const uData32   = new Uint32Array(uArrayBuf);

let lastTime    = performance.now();
const startTime = performance.now();

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255];
}

function frame(now: number) {
  const dt = Math.min(0.033, (now - lastTime) / 1000);
  lastTime = now;
  const t  = (now - startTime) / 1000;

  camera.update(dt);

  const aspect = canvas.width / canvas.height;
  // near/far set to handle models at any scale — far generous, near tight
  const proj = mat4.perspective((60 * Math.PI) / 180, aspect, 0.01, 10000);
  const view  = camera.getViewMatrix();
  const camPos = camera.getPosition();

  let lx = gui.lightX, ly = gui.lightY, lz = gui.lightZ;
  if (gui.autoRotLight) {
    lx = Math.cos(t * 0.5) * 3;
    lz = Math.sin(t * 0.5) * 3;
  }
  const [lr, lg, lb] = hexToRgb(gui.lightColor);

  // ── Write per-object uniforms and render ──────────────────────────────────
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      clearValue: { r: 0.06, g: 0.06, b: 0.10, a: 1 },
      loadOp: "clear", storeOp: "store",
    }],
    depthStencilAttachment: {
      view: depthTexture!.createView(),
      depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store",
    },
  });

  pass.setPipeline(pipeline);

  for (const obj of sceneObjects) {
    if (!obj.mesh.indices.length) continue;

    const normM = mat4.normalMatrix(obj.modelMatrix);
    const mvp   = mat4.multiply(mat4.multiply(proj, view), obj.modelMatrix);

    uData.set(mvp,            0);   // mvp
    uData.set(obj.modelMatrix, 16); // model
    uData.set(normM,          32);  // normalMat
    uData[48] = lx; uData[49] = ly; uData[50] = lz; uData[51] = 0;
    uData[52] = lr; uData[53] = lg; uData[54] = lb; uData[55] = 0;
    uData[56] = gui.ambient; uData[57] = gui.diffuse;
    uData[58] = gui.specular; uData[59] = gui.shininess;
    uData[60] = camPos[0]; uData[61] = camPos[1]; uData[62] = camPos[2];
    uData32[63] = obj.shadingMode;
    uData[64] = obj.color[0]; uData[65] = obj.color[1]; uData[66] = obj.color[2];
    uData[67] = t;
    uData[68] = obj.wireThick; uData[69] = 0; uData[70] = 0; uData[71] = 0;

    device.queue.writeBuffer(obj.uniformBuf, 0, uArrayBuf as any);

    pass.setBindGroup(0, obj.bindGroup);
    pass.setVertexBuffer(0, obj.vertBuf);
    pass.setVertexBuffer(1, obj.baryBuf);
    pass.setIndexBuffer(obj.idxBuf, "uint32");
    pass.drawIndexed(obj.mesh.indices.length);
  }

  pass.end();
  device.queue.submit([encoder.finish()]);
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
