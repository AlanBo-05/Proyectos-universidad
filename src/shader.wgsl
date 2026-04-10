// shader.wgsl — Full pipeline shader
// model_id: 0=Flat, 1=Gouraud, 2=Phong, 3=Wireframe, 4=NormalBuffer, 5=Texture+Phong

struct Uniforms {
  mvp        : mat4x4<f32>,   //  0
  model      : mat4x4<f32>,   // 64
  normalMat  : mat4x4<f32>,   // 128

  lightPos   : vec3<f32>,     // 192
  _p0        : f32,           // 204

  lightColor : vec3<f32>,     // 208
  _p1        : f32,           // 220

  ambient    : f32,           // 224
  diffuse    : f32,           // 228
  specular   : f32,           // 232
  shininess  : f32,           // 236

  camPos     : vec3<f32>,     // 240
  model_id   : u32,           // 252

  objectColor : vec3<f32>,    // 256
  time        : f32,          // 268

  wireThick   : f32,          // 272  wireframe edge thickness
  _p2         : f32,
  _p3         : f32,
  _p4         : f32,
};

@group(0) @binding(0) var<uniform> u : Uniforms;
@group(0) @binding(1) var texSampler : sampler;
@group(0) @binding(2) var texColor   : texture_2d<f32>;

// ── Vertex I/O ─────────────────────────────────────────────────────────────────
struct VSIn {
  @location(0) position  : vec3<f32>,
  @location(1) normal    : vec3<f32>,
  @location(2) uv        : vec2<f32>,
  @location(3) barycentric : vec3<f32>,  // injected as vertex attrib
};

struct VSOut {
  @builtin(position) clipPos : vec4<f32>,
  @location(0) worldPos      : vec3<f32>,
  @location(1) worldNormal   : vec3<f32>,
  @location(2) uv            : vec2<f32>,
  @location(3) gouraudColor  : vec3<f32>,
  @location(4) bary          : vec3<f32>,
};

// ── Lighting helpers ───────────────────────────────────────────────────────────

fn blinnPhongBRDF(N: vec3<f32>, fragWorldPos: vec3<f32>) -> vec3<f32> {
  let L = normalize(u.lightPos - fragWorldPos);
  let V = normalize(u.camPos   - fragWorldPos);
  let H = normalize(L + V);

  let ambientC  = u.ambient  * u.lightColor;
  let NdotL     = max(dot(N, L), 0.0);
  let diffuseC  = u.diffuse  * NdotL * u.lightColor;

  var specularC = vec3<f32>(0.0);
  if NdotL > 0.0 {
    let NdotH = max(dot(N, H), 0.0);
    specularC = u.specular * pow(NdotH, u.shininess) * u.lightColor;
  }
  return (ambientC + diffuseC + specularC) * u.objectColor;
}

// ── Flat shading (derived from screen-space derivatives)
fn flatShading(fragWorldPos: vec3<f32>) -> vec3<f32> {
  let dx    = dpdx(fragWorldPos);
  let dy    = dpdy(fragWorldPos);
  let faceN = normalize(cross(dx, dy));
  return blinnPhongBRDF(faceN, fragWorldPos);
}

// ── Gouraud lighting — computed per vertex, interpolated
fn gouraudLighting(N: vec3<f32>, vertWorldPos: vec3<f32>) -> vec3<f32> {
  let L = normalize(u.lightPos - vertWorldPos);
  let V = normalize(u.camPos   - vertWorldPos);
  let H = normalize(L + V);

  let ambientC  = u.ambient  * u.lightColor;
  let NdotL     = max(dot(N, L), 0.0);
  let diffuseC  = u.diffuse  * NdotL * u.lightColor;

  var specularC = vec3<f32>(0.0);
  if NdotL > 0.0 {
    let NdotH = max(dot(N, H), 0.0);
    specularC = u.specular * pow(NdotH, u.shininess) * u.lightColor;
  }
  return (ambientC + diffuseC + specularC) * u.objectColor;
}

// ── Phong lighting — computed per fragment with smooth interpolated N
fn phongLighting(N: vec3<f32>, fragWorldPos: vec3<f32>) -> vec3<f32> {
  let L = normalize(u.lightPos - fragWorldPos);
  let V = normalize(u.camPos   - fragWorldPos);
  let R = reflect(-L, N);

  let ambientC  = u.ambient  * u.lightColor;
  let NdotL     = max(dot(N, L), 0.0);
  let diffuseC  = u.diffuse  * NdotL * u.lightColor;

  var specularC = vec3<f32>(0.0);
  if NdotL > 0.0 {
    let RdotV = max(dot(R, V), 0.0);
    specularC = u.specular * pow(RdotV, u.shininess) * u.lightColor;
  }
  return (ambientC + diffuseC + specularC) * u.objectColor;
}

// ── Spherical UV mapping (task 10) — maps world-space normal to [0,1]²
fn sphericalUV(worldNormal: vec3<f32>) -> vec2<f32> {
  let n = normalize(worldNormal);
  let u_coord = 0.5 + atan2(n.z, n.x) / (2.0 * 3.14159265);
  let v_coord = 0.5 - asin(clamp(n.y, -1.0, 1.0)) / 3.14159265;
  return vec2<f32>(u_coord, v_coord);
}

// ── Wireframe edge detection (task 11) from barycentric coords
fn wireframeEdge(bary: vec3<f32>, thickness: f32) -> f32 {
  let d = fwidth(bary) * thickness;
  let edge = smoothstep(vec3<f32>(0.0), d, bary);
  return 1.0 - min(edge.x, min(edge.y, edge.z));
}

// ── Vertex shader ──────────────────────────────────────────────────────────────
@vertex
fn vs_main(input: VSIn) -> VSOut {
  var out: VSOut;

  let worldPos4    = u.model    * vec4<f32>(input.position, 1.0);
  let worldNormal4 = u.normalMat * vec4<f32>(input.normal, 0.0);

  out.clipPos     = u.mvp * vec4<f32>(input.position, 1.0);
  out.worldPos    = worldPos4.xyz;
  out.worldNormal = normalize(worldNormal4.xyz);
  out.uv          = input.uv;
  out.bary        = input.barycentric;

  // Gouraud: compute lighting per vertex
  if u.model_id == 1u {
    out.gouraudColor = gouraudLighting(out.worldNormal, out.worldPos);
  } else {
    out.gouraudColor = vec3<f32>(0.0);
  }

  return out;
}

// ── Fragment shader ────────────────────────────────────────────────────────────
@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
  let N = normalize(input.worldNormal);
  var color: vec3<f32>;

  switch u.model_id {
    case 0u: { // Flat
      color = flatShading(input.worldPos);
    }
    case 1u: { // Gouraud
      color = input.gouraudColor;
    }
    case 2u: { // Phong
      color = phongLighting(N, input.worldPos);
    }
    case 3u: { // Wireframe — base Phong + edge highlight
      let baseColor = phongLighting(N, input.worldPos);
      let edge = wireframeEdge(input.bary, u.wireThick);
      color = mix(baseColor, vec3<f32>(0.05), edge);
    }
    case 4u: { // Normal buffer visualization: world normals as RGB
      color = N * 0.5 + vec3<f32>(0.5);
    }
    case 5u: { // Texture + Phong shading (task 10)
      let sphUV = sphericalUV(N);
      let texColor = textureSample(texColor, texSampler, sphUV).rgb;
      // Override objectColor with texture for lighting
      let L = normalize(u.lightPos - input.worldPos);
      let V = normalize(u.camPos   - input.worldPos);
      let R = reflect(-L, N);
      let NdotL = max(dot(N, L), 0.0);
      var spec = vec3<f32>(0.0);
      if NdotL > 0.0 {
        spec = u.specular * pow(max(dot(R, V), 0.0), u.shininess) * u.lightColor;
      }
      color = (u.ambient * u.lightColor + u.diffuse * NdotL * u.lightColor) * texColor + spec;
    }
    default: { // Blinn-Phong
      color = blinnPhongBRDF(N, input.worldPos);
    }
  }

  return vec4<f32>(color, 1.0);
}
