// objLoader.ts — Hand-written OBJ parser producing an indexed mesh
// Task 1: Indexed mesh data structure
// Task 3: Per-face and per-vertex normals

import type { Vec3 } from "./math";
import { vec3 } from "./math";

export interface MeshBounds {
  min:    Vec3;
  max:    Vec3;
  center: Vec3;
  radius: number;  // bounding sphere radius
}

/**
 * Indexed mesh with interleaved vertex data.
 * Vertex layout: [x,y,z, nx,ny,nz, u,v]  (8 floats = 32 bytes)
 */
export interface IndexedMesh {
  /** Interleaved vertex data: pos(3) + normal(3) + uv(2)  per vertex */
  vertices:    Float32Array;
  /** Triangle index list (3 indices per triangle) */
  indices:     Uint32Array;
  /** Per-face normals (3 floats per face) — computed as cross product */
  faceNormals: Float32Array;
  bounds:      MeshBounds;
  triangleCount: number;
  vertexCount:   number;
}

// ─── Internal structures used during parsing ──────────────────────────────────

interface FaceVertex { pi: number; ti: number; ni: number; }

function parseFaceVertex(token: string): FaceVertex {
  // token: "1" or "1/2" or "1/2/3" or "1//3"
  const parts = token.split("/");
  const pi = (parseInt(parts[0]) || 1) - 1;
  const ti = parts.length > 1 && parts[1] ? (parseInt(parts[1]) - 1) : -1;
  const ni = parts.length > 2 && parts[2] ? (parseInt(parts[2]) - 1) : -1;
  return { pi, ti, ni };
}

// ─── Main parser ──────────────────────────────────────────────────────────────

export function parseOBJ(text: string): IndexedMesh {
  const rawPos:  number[] = [];  // flat: x,y,z...
  const rawNorm: number[] = [];  // flat: nx,ny,nz...
  const rawUV:   number[] = [];  // flat: u,v...

  // Unique vertex cache: "pi/ti/ni" → index in finalVertices
  const vertexCache = new Map<string, number>();
  const finalVertices: number[] = [];   // interleaved [pos3, norm3, uv2]
  const finalIndices:  number[] = [];

  // Face records for per-face normal calculation
  interface FaceRecord { indices: number[]; faceNormal: Vec3; }
  const faces: FaceRecord[] = [];

  // Vertex → list of face normals it participates in (for smooth normals)
  const vertFaceNormals: Map<number, Vec3[]> = new Map();

  const lines = text.split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;

    const parts = trimmed.split(/\s+/);
    const token = parts[0];

    if (token === "v") {
      rawPos.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]));
    } else if (token === "vt") {
      rawUV.push(parseFloat(parts[1]), parseFloat(parts[2] ?? "0"));
    } else if (token === "vn") {
      rawNorm.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]));
    } else if (token === "f") {
      // Fan-triangulate polygon faces
      const fvs: FaceVertex[] = [];
      for (let i = 1; i < parts.length; i++) fvs.push(parseFaceVertex(parts[i]));

      for (let i = 1; i < fvs.length - 1; i++) {
        const tri = [fvs[0], fvs[i], fvs[i + 1]];
        const triIndices: number[] = [];

        for (const fv of tri) {
          const key = `${fv.pi}/${fv.ti}/${fv.ni}`;
          let idx = vertexCache.get(key);
          if (idx === undefined) {
            idx = finalVertices.length / 8;
            vertexCache.set(key, idx);

            const px = rawPos[fv.pi * 3 + 0] ?? 0;
            const py = rawPos[fv.pi * 3 + 1] ?? 0;
            const pz = rawPos[fv.pi * 3 + 2] ?? 0;

            const nx = fv.ni >= 0 ? (rawNorm[fv.ni * 3 + 0] ?? 0) : 0;
            const ny = fv.ni >= 0 ? (rawNorm[fv.ni * 3 + 1] ?? 0) : 0;
            const nz = fv.ni >= 0 ? (rawNorm[fv.ni * 3 + 2] ?? 0) : 1;

            const u = fv.ti >= 0 ? (rawUV[fv.ti * 2 + 0] ?? 0) : 0;
            const v = fv.ti >= 0 ? (rawUV[fv.ti * 2 + 1] ?? 0) : 0;

            finalVertices.push(px, py, pz, nx, ny, nz, u, v);
          }
          triIndices.push(idx!);
        }

        // Compute per-face normal (cross product of two edges)
        const vidB = triIndices[0] * 8;
        const vidC = triIndices[1] * 8;
        const vidD = triIndices[2] * 8;
        const p0: Vec3 = [finalVertices[vidB], finalVertices[vidB+1], finalVertices[vidB+2]];
        const p1: Vec3 = [finalVertices[vidC], finalVertices[vidC+1], finalVertices[vidC+2]];
        const p2: Vec3 = [finalVertices[vidD], finalVertices[vidD+1], finalVertices[vidD+2]];
        const e1 = vec3.sub(p1, p0);
        const e2 = vec3.sub(p2, p0);
        const faceN = vec3.normalize(vec3.cross(e1, e2));

        faces.push({ indices: triIndices, faceNormal: faceN });
        for (const idx of triIndices) finalIndices.push(idx);

        // Accumulate face normals per vertex for smooth vertex normals
        for (const vIdx of triIndices) {
          if (!vertFaceNormals.has(vIdx)) vertFaceNormals.set(vIdx, []);
          vertFaceNormals.get(vIdx)!.push(faceN);
        }
      }
    }
  }

  // ── Compute vertex normals by averaging adjacent face normals (if OBJ had no vn)
  // Always recompute from face normals for correctness.
  const vertCount = finalVertices.length / 8;
  for (let vi = 0; vi < vertCount; vi++) {
    const faceNs = vertFaceNormals.get(vi);
    if (!faceNs || faceNs.length === 0) continue;
    let nx = 0, ny = 0, nz = 0;
    for (const fn of faceNs) { nx += fn[0]; ny += fn[1]; nz += fn[2]; }
    const len = Math.hypot(nx, ny, nz) || 1;
    finalVertices[vi * 8 + 3] = nx / len;
    finalVertices[vi * 8 + 4] = ny / len;
    finalVertices[vi * 8 + 5] = nz / len;
  }

  // ── Bounding box & sphere
  let minX= Infinity, minY= Infinity, minZ= Infinity;
  let maxX=-Infinity, maxY=-Infinity, maxZ=-Infinity;
  for (let i = 0; i < finalVertices.length; i += 8) {
    const x=finalVertices[i], y=finalVertices[i+1], z=finalVertices[i+2];
    if (x<minX) minX=x; if (y<minY) minY=y; if (z<minZ) minZ=z;
    if (x>maxX) maxX=x; if (y>maxY) maxY=y; if (z>maxZ) maxZ=z;
  }
  const center: Vec3 = [(minX+maxX)/2, (minY+maxY)/2, (minZ+maxZ)/2];
  let radius = 0;
  for (let i = 0; i < finalVertices.length; i += 8) {
    const dx=finalVertices[i]-center[0], dy=finalVertices[i+1]-center[1], dz=finalVertices[i+2]-center[2];
    radius = Math.max(radius, Math.hypot(dx, dy, dz));
  }
  if (radius === 0) radius = 1;

  const faceNormals = new Float32Array(faces.length * 3);
  for (let fi = 0; fi < faces.length; fi++) {
    faceNormals[fi*3+0] = faces[fi].faceNormal[0];
    faceNormals[fi*3+1] = faces[fi].faceNormal[1];
    faceNormals[fi*3+2] = faces[fi].faceNormal[2];
  }

  return {
    vertices:    new Float32Array(finalVertices),
    indices:     new Uint32Array(finalIndices),
    faceNormals,
    bounds: {
      min: [minX, minY, minZ],
      max: [maxX, maxY, maxZ],
      center,
      radius,
    },
    triangleCount: finalIndices.length / 3,
    vertexCount:   vertCount,
  };
}

/** Fetch and parse an OBJ file from a URL */
export async function loadOBJ(url: string): Promise<IndexedMesh> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to load OBJ: ${url} (${resp.status})`);
  const text = await resp.text();
  return parseOBJ(text);
}
