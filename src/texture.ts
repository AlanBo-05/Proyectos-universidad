// texture.ts — GPU texture loading + procedural checkerboard fallback

/** Create a 128×128 procedural checkerboard texture */
function makeCheckerboard(device: GPUDevice): GPUTexture {
  const SIZE = 128;
  const data = new Uint8Array(SIZE * SIZE * 4);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const tile = ((x >> 4) + (y >> 4)) & 1;
      const i = (y * SIZE + x) * 4;
      data[i + 0] = tile ? 240 : 30;
      data[i + 1] = tile ? 240 : 30;
      data[i + 2] = tile ? 240 : 30;
      data[i + 3] = 255;
    }
  }
  const tex = device.createTexture({
    size: [SIZE, SIZE, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture({ texture: tex }, data, { bytesPerRow: SIZE * 4 }, [SIZE, SIZE, 1]);
  return tex;
}

/** Load an image URL into a GPUTexture, falling back to checkerboard */
export async function loadTexture(device: GPUDevice, url: string | null): Promise<GPUTexture> {
  if (!url) return makeCheckerboard(device);
  try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const blob   = await resp.blob();
    const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none" });
    const tex = device.createTexture({
      size: [bitmap.width, bitmap.height, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    device.queue.copyExternalImageToTexture(
      { source: bitmap },
      { texture: tex },
      [bitmap.width, bitmap.height],
    );
    return tex;
  } catch (e) {
    console.warn("Texture load failed, using checkerboard:", e);
    return makeCheckerboard(device);
  }
}

/** Create a GPUSampler suitable for texture mapping */
export function createSampler(device: GPUDevice): GPUSampler {
  return device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
    mipmapFilter: "linear",
    addressModeU: "repeat",
    addressModeV: "repeat",
  });
}
