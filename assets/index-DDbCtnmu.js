(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const r of document.querySelectorAll('link[rel="modulepreload"]'))n(r);new MutationObserver(r=>{for(const i of r)if(i.type==="childList")for(const s of i.addedNodes)s.tagName==="LINK"&&s.rel==="modulepreload"&&n(s)}).observe(document,{childList:!0,subtree:!0});function o(r){const i={};return r.integrity&&(i.integrity=r.integrity),r.referrerPolicy&&(i.referrerPolicy=r.referrerPolicy),r.crossOrigin==="use-credentials"?i.credentials="include":r.crossOrigin==="anonymous"?i.credentials="omit":i.credentials="same-origin",i}function n(r){if(r.ep)return;r.ep=!0;const i=o(r);fetch(r.href,i)}})();const Ct=`// shader.wgsl — Full pipeline shader
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
`,V={identity(){return[0,0,0,1]},fromAxisAngle(e,t){const o=Math.sin(t/2),[n,r,i]=f.normalize(e);return[n*o,r*o,i*o,Math.cos(t/2)]},multiply(e,t){const[o,n,r,i]=e,[s,a,u,h]=t;return[i*s+o*h+n*u-r*a,i*a-o*u+n*h+r*s,i*u+o*a-n*s+r*h,i*h-o*s-n*a-r*u]},normalize(e){const t=Math.hypot(...e)||1;return[e[0]/t,e[1]/t,e[2]/t,e[3]/t]},toMat4(e){const[t,o,n,r]=e,i=new Float32Array(16);return i[0]=1-2*(o*o+n*n),i[1]=2*(t*o+n*r),i[2]=2*(t*n-o*r),i[3]=0,i[4]=2*(t*o-n*r),i[5]=1-2*(t*t+n*n),i[6]=2*(o*n+t*r),i[7]=0,i[8]=2*(t*n+o*r),i[9]=2*(o*n-t*r),i[10]=1-2*(t*t+o*o),i[11]=0,i[12]=0,i[13]=0,i[14]=0,i[15]=1,i}},f={add(e,t){return[e[0]+t[0],e[1]+t[1],e[2]+t[2]]},sub(e,t){return[e[0]-t[0],e[1]-t[1],e[2]-t[2]]},scale(e,t){return[e[0]*t,e[1]*t,e[2]*t]},dot(e,t){return e[0]*t[0]+e[1]*t[1]+e[2]*t[2]},cross(e,t){return[e[1]*t[2]-e[2]*t[1],e[2]*t[0]-e[0]*t[2],e[0]*t[1]-e[1]*t[0]]},length(e){return Math.hypot(e[0],e[1],e[2])},normalize(e){const t=Math.hypot(e[0],e[1],e[2])||1;return[e[0]/t,e[1]/t,e[2]/t]},lerp(e,t,o){return[e[0]+(t[0]-e[0])*o,e[1]+(t[1]-e[1])*o,e[2]+(t[2]-e[2])*o]},negate(e){return[-e[0],-e[1],-e[2]]}},I={identity(){const e=new Float32Array(16);return e[0]=1,e[5]=1,e[10]=1,e[15]=1,e},multiply(e,t){const o=new Float32Array(16);for(let n=0;n<4;n++)for(let r=0;r<4;r++)o[n*4+r]=e[0+r]*t[n*4+0]+e[4+r]*t[n*4+1]+e[8+r]*t[n*4+2]+e[12+r]*t[n*4+3];return o},transpose(e){const t=new Float32Array(16);for(let o=0;o<4;o++)for(let n=0;n<4;n++)t[n*4+o]=e[o*4+n];return t},invert(e){const t=new Float32Array(16),o=e[0],n=e[1],r=e[2],i=e[3],s=e[4],a=e[5],u=e[6],h=e[7],l=e[8],d=e[9],g=e[10],E=e[11],w=e[12],p=e[13],x=e[14],B=e[15],L=o*a-n*s,k=o*u-r*s,c=o*h-i*s,M=n*u-r*a,m=n*h-i*a,C=r*h-i*u,T=l*p-d*w,P=l*x-g*w,A=l*B-E*w,z=d*x-g*p,R=d*B-E*p,U=g*B-E*x;let b=L*U-k*R+c*z+M*A-m*P+C*T;return b?(b=1/b,t[0]=(a*U-u*R+h*z)*b,t[1]=(u*A-s*U-h*P)*b,t[2]=(s*R-a*A+h*T)*b,t[3]=(a*P-s*z-u*T)*b,t[4]=(r*R-n*U-i*z)*b,t[5]=(o*U-r*A+i*P)*b,t[6]=(n*A-o*R-i*T)*b,t[7]=(o*z-n*P+r*T)*b,t[8]=(p*C-x*m+B*M)*b,t[9]=(x*c-w*C-B*k)*b,t[10]=(w*m-p*c+B*L)*b,t[11]=(p*k-w*M-x*L)*b,t[12]=(g*m-d*C-E*M)*b,t[13]=(l*C-g*c+E*k)*b,t[14]=(d*c-l*m-E*L)*b,t[15]=(l*M-d*k+g*L)*b,t):I.identity()},normalMatrix(e){return I.transpose(I.invert(e))},translation(e,t,o){const n=I.identity();return n[12]=e,n[13]=t,n[14]=o,n},scaling(e,t,o){const n=I.identity();return n[0]=e,n[5]=t,n[10]=o,n},rotationX(e){const t=Math.cos(e),o=Math.sin(e),n=I.identity();return n[5]=t,n[6]=o,n[9]=-o,n[10]=t,n},rotationY(e){const t=Math.cos(e),o=Math.sin(e),n=I.identity();return n[0]=t,n[2]=-o,n[8]=o,n[10]=t,n},rotationZ(e){const t=Math.cos(e),o=Math.sin(e),n=I.identity();return n[0]=t,n[1]=o,n[4]=-o,n[5]=t,n},perspective(e,t,o,n){const r=1/Math.tan(e/2),i=new Float32Array(16);return i[0]=r/t,i[5]=r,i[10]=n/(o-n),i[11]=-1,i[14]=n*o/(o-n),i},lookAt(e,t,o){const n=f.normalize(f.sub(e,t)),r=f.normalize(f.cross(o,n)),i=f.cross(n,r),s=new Float32Array(16);return s[0]=r[0],s[4]=r[1],s[8]=r[2],s[12]=-f.dot(r,e),s[1]=i[0],s[5]=i[1],s[9]=i[2],s[13]=-f.dot(i,e),s[2]=n[0],s[6]=n[1],s[10]=n[2],s[14]=-f.dot(n,e),s[3]=0,s[7]=0,s[11]=0,s[15]=1,s}};class Mt{target=[0,0,0];radius=5;minRadius=.1;maxRadius=5e3;orientation=V.identity();dragging=!1;lastNDC=[0,0];canvas;moveSpeed=1.5;keys=new Set;bind(t){this.canvas=t,t.addEventListener("mousedown",n=>{n.button===0&&(this.dragging=!0,this.lastNDC=this.toNDC(n.clientX,n.clientY),n.preventDefault())}),window.addEventListener("mouseup",n=>{n.button===0&&(this.dragging=!1)}),window.addEventListener("mousemove",n=>{if(!this.dragging)return;const r=this.toNDC(n.clientX,n.clientY);this.onDrag(this.lastNDC,r),this.lastNDC=r}),t.addEventListener("wheel",n=>{n.preventDefault();const r=1+n.deltaY*.001;this.radius=Math.max(this.minRadius,Math.min(this.maxRadius,this.radius*r))},{passive:!1});let o=null;t.addEventListener("touchstart",n=>{o=n.touches,n.preventDefault()},{passive:!1}),t.addEventListener("touchmove",n=>{if(n.preventDefault(),!!o){if(n.touches.length===1&&o.length===1){const r=this.toNDC(n.touches[0].clientX,n.touches[0].clientY),i=this.toNDC(o[0].clientX,o[0].clientY);this.onDrag(i,r)}else if(n.touches.length===2&&o.length===2){const r=Math.hypot(o[0].clientX-o[1].clientX,o[0].clientY-o[1].clientY),i=Math.hypot(n.touches[0].clientX-n.touches[1].clientX,n.touches[0].clientY-n.touches[1].clientY),s=r/(i||1);this.radius=Math.max(this.minRadius,Math.min(this.maxRadius,this.radius*s))}o=n.touches}},{passive:!1}),t.addEventListener("touchend",()=>{o=null}),window.addEventListener("keydown",n=>this.keys.add(n.key)),window.addEventListener("keyup",n=>this.keys.delete(n.key))}reset(t,o){this.target=[...t],this.radius=o*2.5,this.maxRadius=o*20,this.minRadius=o*.1,this.orientation=V.identity()}update(t){const o=this.moveSpeed*t*this.radius*.3,n=this.getViewMatrix(),r=[n[0],n[4],n[8]],i=[n[1],n[5],n[9]],s=[-n[2],-n[6],-n[10]];(this.keys.has("a")||this.keys.has("ArrowLeft"))&&(this.target=f.add(this.target,f.scale(r,-o))),(this.keys.has("d")||this.keys.has("ArrowRight"))&&(this.target=f.add(this.target,f.scale(r,o))),(this.keys.has("w")||this.keys.has("ArrowUp"))&&(this.target=f.add(this.target,f.scale(s,o))),(this.keys.has("s")||this.keys.has("ArrowDown"))&&(this.target=f.add(this.target,f.scale(s,-o))),this.keys.has("q")&&(this.target=f.add(this.target,f.scale(i,-o))),this.keys.has("e")&&(this.target=f.add(this.target,f.scale(i,o))),this.keys.has("r")&&this.reset(this.target,this.radius/2.5)}getViewMatrix(){const t=V.toMat4(this.orientation),o=[t[8]*this.radius,t[9]*this.radius,t[10]*this.radius],n=f.add(this.target,o),r=[t[4],t[5],t[6]];return I.lookAt(n,this.target,r)}getPosition(){const t=this.getViewMatrix();return[-(t[0]*t[12]+t[1]*t[13]+t[2]*t[14]),-(t[4]*t[12]+t[5]*t[13]+t[6]*t[14]),-(t[8]*t[12]+t[9]*t[13]+t[10]*t[14])]}toNDC(t,o){const n=this.canvas.getBoundingClientRect();return[(t-n.left)/n.width*2-1,-((o-n.top)/n.height*2-1)]}toSphere(t){const[o,n]=t,r=o*o+n*n;return r<=.5?f.normalize([o,n,Math.sqrt(1-r)]):f.normalize([o,n,.5/Math.sqrt(r)])}onDrag(t,o){const n=this.toSphere(t),r=this.toSphere(o),i=f.cross(n,r),s=f.length(i);if(s<1e-8)return;const a=Math.asin(Math.min(1,s))*2,u=V.fromAxisAngle(f.normalize(i),a);this.orientation=V.normalize(V.multiply(u,this.orientation))}}function Lt(e){const t=e.split("/"),o=(parseInt(t[0])||1)-1,n=t.length>1&&t[1]?parseInt(t[1])-1:-1,r=t.length>2&&t[2]?parseInt(t[2])-1:-1;return{pi:o,ti:n,ni:r}}function Pt(e){const t=[],o=[],n=[],r=new Map,i=[],s=[],a=[],u=new Map,h=e.split(/\r?\n/);for(const c of h){const M=c.trim();if(!M||M.startsWith("#"))continue;const m=M.split(/\s+/),C=m[0];if(C==="v")t.push(parseFloat(m[1]),parseFloat(m[2]),parseFloat(m[3]));else if(C==="vt")n.push(parseFloat(m[1]),parseFloat(m[2]??"0"));else if(C==="vn")o.push(parseFloat(m[1]),parseFloat(m[2]),parseFloat(m[3]));else if(C==="f"){const T=[];for(let P=1;P<m.length;P++)T.push(Lt(m[P]));for(let P=1;P<T.length-1;P++){const A=[T[0],T[P],T[P+1]],z=[];for(const y of A){const K=`${y.pi}/${y.ti}/${y.ni}`;let W=r.get(K);if(W===void 0){W=i.length/8,r.set(K,W);const gt=t[y.pi*3+0]??0,pt=t[y.pi*3+1]??0,mt=t[y.pi*3+2]??0,vt=y.ni>=0?o[y.ni*3+0]??0:0,bt=y.ni>=0?o[y.ni*3+1]??0:0,yt=y.ni>=0?o[y.ni*3+2]??0:1,wt=y.ti>=0?n[y.ti*2+0]??0:0,xt=y.ti>=0?n[y.ti*2+1]??0:0;i.push(gt,pt,mt,vt,bt,yt,wt,xt)}z.push(W)}const R=z[0]*8,U=z[1]*8,b=z[2]*8,Z=[i[R],i[R+1],i[R+2]],ut=[i[U],i[U+1],i[U+2]],dt=[i[b],i[b+1],i[b+2]],ft=f.sub(ut,Z),ht=f.sub(dt,Z),H=f.normalize(f.cross(ft,ht));a.push({indices:z,faceNormal:H});for(const y of z)s.push(y);for(const y of z)u.has(y)||u.set(y,[]),u.get(y).push(H)}}}const l=i.length/8;for(let c=0;c<l;c++){const M=u.get(c);if(!M||M.length===0)continue;let m=0,C=0,T=0;for(const A of M)m+=A[0],C+=A[1],T+=A[2];const P=Math.hypot(m,C,T)||1;i[c*8+3]=m/P,i[c*8+4]=C/P,i[c*8+5]=T/P}let d=1/0,g=1/0,E=1/0,w=-1/0,p=-1/0,x=-1/0;for(let c=0;c<i.length;c+=8){const M=i[c],m=i[c+1],C=i[c+2];M<d&&(d=M),m<g&&(g=m),C<E&&(E=C),M>w&&(w=M),m>p&&(p=m),C>x&&(x=C)}const B=[(d+w)/2,(g+p)/2,(E+x)/2];let L=0;for(let c=0;c<i.length;c+=8){const M=i[c]-B[0],m=i[c+1]-B[1],C=i[c+2]-B[2];L=Math.max(L,Math.hypot(M,m,C))}L===0&&(L=1);const k=new Float32Array(a.length*3);for(let c=0;c<a.length;c++)k[c*3+0]=a[c].faceNormal[0],k[c*3+1]=a[c].faceNormal[1],k[c*3+2]=a[c].faceNormal[2];return{vertices:new Float32Array(i),indices:new Uint32Array(s),faceNormals:k,bounds:{min:[d,g,E],max:[w,p,x],center:B,radius:L},triangleCount:s.length/3,vertexCount:l}}async function tt(e){const t=await fetch(e);if(!t.ok)throw new Error(`Failed to load OBJ: ${e} (${t.status})`);const o=await t.text();return Pt(o)}function Et(e){const o=new Uint8Array(65536);for(let r=0;r<128;r++)for(let i=0;i<128;i++){const s=(i>>4)+(r>>4)&1,a=(r*128+i)*4;o[a+0]=s?240:30,o[a+1]=s?240:30,o[a+2]=s?240:30,o[a+3]=255}const n=e.createTexture({size:[128,128,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});return e.queue.writeTexture({texture:n},o,{bytesPerRow:512},[128,128,1]),n}async function Nt(e,t){return Et(e)}function Bt(e){return e.createSampler({magFilter:"linear",minFilter:"linear",mipmapFilter:"linear",addressModeU:"repeat",addressModeV:"repeat"})}const St=[{id:0,label:"Flat"},{id:1,label:"Gouraud"},{id:2,label:"Phong"},{id:3,label:"Wireframe"},{id:4,label:"Normals"},{id:5,label:"Texture"}],It={Beacon:"#e8992e",Teapot:"#4a9eff",Cube:"#cc55ff",Sphere:"#44e896"},_=new Map;function Tt(e){const t=e.gui,o=document.createElement("div");o.id="gui-overlay",o.innerHTML=`
<div class="gui-panel" id="gui-panel">
  <div class="gui-header">
    <span class="gui-title"> 3D Pipeline</span>
    <button class="gui-toggle" id="gui-toggle" title="Collapse">−</button>
  </div>
  <div class="gui-body" id="gui-body">

    <!-- Load models -->
    <section class="gui-section">
      <div class="gui-label">Load Models</div>
      <div class="btn-row">
        <button class="load-btn" id="btn-beacon">Beacon</button>
        <button class="load-btn" id="btn-teapot">Teapot</button>
        <button class="load-btn" id="btn-cube">Cube</button>
        <button class="load-btn" id="btn-sphere">Sphere</button>
      </div>
      <div class="status-line" id="load-status">Beacon loaded</div>
    </section>

    <!-- Per-object controls (injected dynamically) -->
    <section class="gui-section" id="obj-section">
      <div class="gui-label">Objects</div>
      <div id="obj-list"></div>
    </section>

    <!-- Material -->
    <section class="gui-section">
      <div class="gui-label">Material</div>
      ${j("ambient","Ambient  Ka",0,1,.01,t.ambient)}
      ${j("diffuse","Diffuse  Kd",0,1,.01,t.diffuse)}
      ${j("specular","Specular Ks",0,1,.01,t.specular)}
      ${j("shininess","Shininess n",1,256,1,t.shininess)}
    </section>

    <!-- Light -->
    <section class="gui-section">
      <div class="gui-label">Light</div>
      ${j("lightX","X",-10,10,.1,t.lightX)}
      ${j("lightY","Y",-10,10,.1,t.lightY)}
      ${j("lightZ","Z",-10,10,.1,t.lightZ)}
      <label class="check-row">
        <input type="checkbox" id="autoRotLight" ${t.autoRotLight?"checked":""}> Auto-rotate
      </label>
      <div class="color-row">
        <span>Light Color</span>
        <input type="color" id="lightColor" value="${t.lightColor}">
      </div>
    </section>

    <!-- Wireframe -->
    <section class="gui-section">
      <div class="gui-label">Wireframe</div>
      ${j("wireThick","Edge width",.5,4,.1,t.wireThick)}
    </section>

    <!-- Controls hint -->
    <div class="gui-hint">
      🖱 Drag = orbit &nbsp;|&nbsp; Scroll = zoom<br>
      WASD/QE = pan &nbsp;|&nbsp; R = reset camera
    </div>
  </div>
</div>`,document.body.appendChild(o);const n=document.getElementById("gui-toggle"),r=document.getElementById("gui-body");n.addEventListener("click",()=>{const l=r.style.display==="none";r.style.display=l?"":"none",n.textContent=l?"−":"+"});function i(l){document.getElementById("load-status").textContent=l}function s(l,d){i(`Loading ${l}…`),d().then(()=>{i(`${l} loaded ✓`),h()}).catch(g=>{i(`Error: ${g.message}`),console.error(g)})}document.getElementById("btn-beacon").addEventListener("click",()=>s("Beacon",e.onLoadBeacon)),document.getElementById("btn-teapot").addEventListener("click",()=>s("Teapot",e.onLoadTeapot)),document.getElementById("btn-cube").addEventListener("click",()=>{e.onLoadCube(),i("Cube added ✓"),h()}),document.getElementById("btn-sphere").addEventListener("click",()=>{e.onLoadSphere(),i("Sphere added ✓"),h()}),["ambient","diffuse","specular","shininess"].forEach(l=>{const d=document.getElementById(l),g=document.getElementById(`${l}-val`);d.addEventListener("input",()=>{t[l]=parseFloat(d.value),g.textContent=d.value})}),["lightX","lightY","lightZ"].forEach(l=>{const d=document.getElementById(l),g=document.getElementById(`${l}-val`);d.addEventListener("input",()=>{t[l]=parseFloat(d.value),g.textContent=d.value})}),document.getElementById("autoRotLight").addEventListener("change",l=>{t.autoRotLight=l.target.checked}),document.getElementById("lightColor").addEventListener("input",l=>{t.lightColor=l.target.value});const a=document.getElementById("wireThick"),u=document.getElementById("wireThick-val");a.addEventListener("input",()=>{const l=parseFloat(a.value);t.wireThick=l,u.textContent=l.toFixed(1),e.onWireThickChange(l)});function h(){const l=document.getElementById("obj-list"),d=e.getObjects();if(l.innerHTML="",d.length===0){l.innerHTML='<div class="empty-hint">No objects loaded</div>';return}for(const{uid:g,name:E}of d){_.has(g)||_.set(g,2);const w=_.get(g),p=It[E]??"#4a9eff",x=document.createElement("div");x.className="obj-card",x.innerHTML=`
        <div class="obj-card-header">
          <span class="obj-name">${E}</span>
          <button class="obj-remove" title="Remove">✕</button>
        </div>
        <div class="shading-btns">
          ${St.map(L=>`<button class="shade-btn ${w===L.id?"active":""}" data-mode="${L.id}">${L.label}</button>`).join("")}
        </div>
        <div class="color-row">
          <span>Color</span>
          <input type="color" class="obj-color" value="${p}">
        </div>`,l.appendChild(x),x.querySelector(".obj-remove").addEventListener("click",()=>{e.onRemoveObject(g),_.delete(g),h()}),x.querySelectorAll(".shade-btn").forEach(L=>{L.addEventListener("click",()=>{const k=Number(L.dataset.mode);_.set(g,k),e.onShadingChange(g,k),x.querySelectorAll(".shade-btn").forEach(c=>c.classList.remove("active")),L.classList.add("active")})});const B=x.querySelector(".obj-color");B.addEventListener("input",()=>{e.onColorChange(g,B.value)})}}h(),window._rebuildObjList=h}function j(e,t,o,n,r,i){return`
  <div class="slider-row">
    <span class="slider-label">${t}</span>
    <input type="range" id="${e}" min="${o}" max="${n}" step="${r}" value="${i}">
    <span class="slider-val" id="${e}-val">${i}</span>
  </div>`}if(!navigator.gpu)throw new Error("WebGPU not supported");const O=document.querySelector("#gfx-main"),et=await navigator.gpu.requestAdapter();if(!et)throw new Error("No GPU adapter found");const N=await et.requestDevice(),nt=O.getContext("webgpu"),ot=navigator.gpu.getPreferredCanvasFormat();let X=null;function it(){O.width=Math.max(1,Math.floor(window.innerWidth*devicePixelRatio)),O.height=Math.max(1,Math.floor(window.innerHeight*devicePixelRatio)),nt.configure({device:N,format:ot,alphaMode:"premultiplied"}),X?.destroy(),X=N.createTexture({size:[O.width,O.height],format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT})}it();window.addEventListener("resize",it);const kt=Bt(N),zt=await Nt(N),rt=288,J=N.createShaderModule({label:"Main Shader",code:Ct}),st=N.createRenderPipeline({label:"Main Pipeline",layout:"auto",vertex:{module:J,entryPoint:"vs_main",buffers:[{arrayStride:32,attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x3"},{shaderLocation:2,offset:24,format:"float32x2"}]},{arrayStride:12,attributes:[{shaderLocation:3,offset:0,format:"float32x3"}]}]},fragment:{module:J,entryPoint:"fs_main",targets:[{format:ot}]},primitive:{topology:"triangle-list",cullMode:"back"},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}});let S=[],At=0;function Ft(e){const t=e/3,o=new Float32Array(t*9);for(let r=0;r<t;r++){const i=r*9;o[i+0]=1,o[i+1]=0,o[i+2]=0,o[i+3]=0,o[i+4]=1,o[i+5]=0,o[i+6]=0,o[i+7]=0,o[i+8]=1}const n=N.createBuffer({size:o.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST});return N.queue.writeBuffer(n,0,o),n}function Rt(e){const t=N.createBuffer({size:e.vertices.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST});N.queue.writeBuffer(t,0,e.vertices);const o=N.createBuffer({size:Math.max(e.indices.byteLength,4),usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST});N.queue.writeBuffer(o,0,e.indices);const n=Ft(e.indices.length);return{vertBuf:t,idxBuf:o,baryBuf:n}}function Ut(e,t){return N.createBindGroup({layout:st.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:e}},{binding:1,resource:kt},{binding:2,resource:t.createView()}]})}function at(e,t,o=0){const{center:n,radius:r}=e.bounds,i=1/r,s=I.translation(-n[0]*i+t,-n[1]*i+o,-n[2]*i),a=I.scaling(i,i,i);return I.multiply(s,a)}function Y(e,t,o,n,r=zt){const{vertBuf:i,idxBuf:s,baryBuf:a}=Rt(t),u=N.createBuffer({size:rt,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),h=Ut(u,r),l=at(t,o);return{uid:At++,name:e,mesh:t,vertBuf:i,idxBuf:s,baryBuf:a,uniformBuf:u,bindGroup:h,color:n,modelMatrix:l,shadingMode:2,wireThick:1.5}}const $=new Mt;$.bind(O);$.reset([0,0,0],3);const F={ambient:.12,diffuse:.75,specular:.6,shininess:32,lightX:2,lightY:5,lightZ:2,autoRotLight:!0,lightColor:"#ffffff",wireThick:1.5};function jt(){const e=[{n:[0,0,1],verts:[[-1,-1,1,0,1],[1,-1,1,1,1],[1,1,1,1,0],[-1,1,1,0,0]]},{n:[0,0,-1],verts:[[1,-1,-1,0,1],[-1,-1,-1,1,1],[-1,1,-1,1,0],[1,1,-1,0,0]]},{n:[-1,0,0],verts:[[-1,-1,-1,0,1],[-1,-1,1,1,1],[-1,1,1,1,0],[-1,1,-1,0,0]]},{n:[1,0,0],verts:[[1,-1,1,0,1],[1,-1,-1,1,1],[1,1,-1,1,0],[1,1,1,0,0]]},{n:[0,1,0],verts:[[-1,1,1,0,1],[1,1,1,1,1],[1,1,-1,1,0],[-1,1,-1,0,0]]},{n:[0,-1,0],verts:[[-1,-1,-1,0,1],[1,-1,-1,1,1],[1,-1,1,1,0],[-1,-1,1,0,0]]}],t=[],o=[];for(const r of e){const i=t.length/8;for(const s of r.verts)t.push(s[0],s[1],s[2],...r.n,s[3],s[4]);o.push(i,i+1,i+2,i,i+2,i+3)}const n=[0,0,0];return{vertices:new Float32Array(t),indices:new Uint32Array(o),faceNormals:new Float32Array(18),bounds:{min:[-1,-1,-1],max:[1,1,1],center:n,radius:Math.sqrt(3)},triangleCount:o.length/3,vertexCount:t.length/8}}function Ot(e=48,t=64){const o=[],n=[];for(let i=0;i<=e;i++){const s=Math.PI*i/e;for(let a=0;a<=t;a++){const u=2*Math.PI*a/t,h=Math.sin(s)*Math.cos(u),l=Math.cos(s),d=Math.sin(s)*Math.sin(u);o.push(h,l,d,h,l,d,a/t,i/e)}}for(let i=0;i<e;i++)for(let s=0;s<t;s++){const a=i*(t+1)+s,u=a+t+1;n.push(a,u,a+1,a+1,u,u+1)}const r=[0,0,0];return{vertices:new Float32Array(o),indices:new Uint32Array(n),faceNormals:new Float32Array(0),bounds:{min:[-1,-1,-1],max:[1,1,1],center:r,radius:1},triangleCount:n.length/3,vertexCount:o.length/8}}async function lt(){const e=await tt("/models/KAUST_Beacon.obj");return Y("Beacon",e,0,[.9,.6,.2])}async function Vt(){const e=await tt("/models/teapot.obj");return Y("Teapot",e,0,[.3,.7,1])}function Dt(){return Y("Cube",jt(),0,[.8,.3,.9])}function $t(){return Y("Sphere",Ot(),0,[.2,.9,.5])}Tt({onLoadBeacon:async()=>{const e=await lt();S.push(e),G(),D()},onLoadTeapot:async()=>{const e=await Vt();S.push(e),G(),D()},onLoadCube:()=>{S.push(Dt()),G(),D()},onLoadSphere:()=>{S.push($t()),G(),D()},onRemoveObject:e=>{S=S.filter(t=>t.uid!==e),G(),D()},onShadingChange:(e,t)=>{const o=S.find(n=>n.uid===e);o&&(o.shadingMode=t)},onColorChange:(e,t)=>{const o=S.find(n=>n.uid===e);if(o){const n=parseInt(t.slice(1),16);o.color=[(n>>16&255)/255,(n>>8&255)/255,(n&255)/255]}},onWireThickChange:e=>{F.wireThick=e},getObjects:()=>S.map(e=>({uid:e.uid,name:e.name})),gui:F});function G(){const e=S.length;if(e===0)return;const t=Math.ceil(Math.sqrt(e)),o=2.5,n=Math.min(e,t),r=Math.ceil(e/t),i=(n-1)*o,s=(r-1)*o;for(let a=0;a<e;a++){const u=a%t,h=Math.floor(a/t),l=-i/2+u*o,d=s/2-h*o;S[a].modelMatrix=at(S[a].mesh,l,d)}}function D(){if(S.length===0)return;const e=S.length,t=Math.ceil(Math.sqrt(e)),o=Math.ceil(e/t),n=2.5,r=(t-1)*n/2+1.5,i=(o-1)*n/2+1.5,s=Math.sqrt(r*r+i*i)+1;$.reset([0,0,0],s)}const _t=await lt();S.push(_t);D();const q=new ArrayBuffer(rt),v=new Float32Array(q),Gt=new Uint32Array(q);let Q=performance.now();const Wt=performance.now();function Yt(e){const t=parseInt(e.slice(1),16);return[(t>>16&255)/255,(t>>8&255)/255,(t&255)/255]}function ct(e){const t=Math.min(.033,(e-Q)/1e3);Q=e;const o=(e-Wt)/1e3;$.update(t);const n=O.width/O.height,r=I.perspective(60*Math.PI/180,n,.01,1e4),i=$.getViewMatrix(),s=$.getPosition();let a=F.lightX,u=F.lightY,h=F.lightZ;F.autoRotLight&&(a=Math.cos(o*.5)*3,h=Math.sin(o*.5)*3);const[l,d,g]=Yt(F.lightColor),E=N.createCommandEncoder(),w=E.beginRenderPass({colorAttachments:[{view:nt.getCurrentTexture().createView(),clearValue:{r:.06,g:.06,b:.1,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:X.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});w.setPipeline(st);for(const p of S){if(!p.mesh.indices.length)continue;const x=I.normalMatrix(p.modelMatrix),B=I.multiply(I.multiply(r,i),p.modelMatrix);v.set(B,0),v.set(p.modelMatrix,16),v.set(x,32),v[48]=a,v[49]=u,v[50]=h,v[51]=0,v[52]=l,v[53]=d,v[54]=g,v[55]=0,v[56]=F.ambient,v[57]=F.diffuse,v[58]=F.specular,v[59]=F.shininess,v[60]=s[0],v[61]=s[1],v[62]=s[2],Gt[63]=p.shadingMode,v[64]=p.color[0],v[65]=p.color[1],v[66]=p.color[2],v[67]=o,v[68]=p.wireThick,v[69]=0,v[70]=0,v[71]=0,N.queue.writeBuffer(p.uniformBuf,0,q),w.setBindGroup(0,p.bindGroup),w.setVertexBuffer(0,p.vertBuf),w.setVertexBuffer(1,p.baryBuf),w.setIndexBuffer(p.idxBuf,"uint32"),w.drawIndexed(p.mesh.indices.length)}w.end(),N.queue.submit([E.finish()]),requestAnimationFrame(ct)}requestAnimationFrame(ct);
