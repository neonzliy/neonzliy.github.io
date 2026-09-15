import * as T from '../../vendor/three/three.module.min.js';

// A smooth, elliptical loft: sections = [z, half width, half height, center y].
// Partial angular ranges make separate fitted armor panels from the same surface.
export function loft(sections, { start = 0, end = Math.PI * 2, rings = 32, sides = 32, tStart = 0, tEnd = 1 } = {}) {
  const profile = new T.CatmullRomCurve3(sections.map(s => new T.Vector3(s[0], s[1], s[2])));
  const centers = new T.CatmullRomCurve3(sections.map(s => new T.Vector3(s[0], s[3] || 0, 0)));
  const positions = [], indices = [];
  for (let i = 0; i <= rings; i++) {
    const t = tStart + (tEnd - tStart) * i / rings;
    const p = profile.getPoint(t), c = centers.getPoint(t);
    for (let j = 0; j <= sides; j++) {
      const a = start + (end - start) * j / sides;
      positions.push(Math.max(.001, p.y) * Math.cos(a), c.y + Math.max(.001, p.z) * Math.sin(a), p.x);
    }
  }
  for (let i = 0; i < rings; i++) for (let j = 0; j < sides; j++) {
    const a = i * (sides + 1) + j, b = a + sides + 1;
    indices.push(a, a + 1, b, a + 1, b + 1, b);
  }
  return surface(positions, indices);
}

function surface(positions, indices) {
  const g = new T.BufferGeometry();
  g.setAttribute('position', new T.Float32BufferAttribute(positions, 3));
  g.setIndex(indices); g.computeVertexNormals();
  return g;
}

// Closed, cambered feather blade. Rounded shoulders taper into an offset tip.
export function feather(length = 1, width = .23, bend = .08, resolution = 18) {
  const p = [], idx = [], sides = 12;
  for (let i = 0; i <= resolution; i++) {
    const t = i / resolution;
    const w = width * Math.pow(Math.sin(Math.PI * t), .55) * (1 - .4 * t) + .002;
    for (let j = 0; j <= sides; j++) {
      const a = j / sides * Math.PI * 2;
      p.push(Math.cos(a) * w + bend * t * t, Math.sin(a) * (.022 * Math.sin(Math.PI * t) + .002) + .055 * Math.sin(Math.PI * t), -t * length);
    }
  }
  for (let i = 0; i < resolution; i++) for (let j = 0; j < sides; j++) {
    const a = i * (sides + 1) + j, b = a + sides + 1;
    idx.push(a, b, a + 1, a + 1, b, b + 1);
  }
  return surface(p, idx);
}

export function rod(a, b, radius, material) {
  const from = new T.Vector3(...a), to = new T.Vector3(...b);
  const m = new T.Mesh(new T.CylinderGeometry(radius, radius, from.distanceTo(to), 12), material);
  m.position.copy(from).add(to).multiplyScalar(.5);
  m.quaternion.setFromUnitVectors(new T.Vector3(0,1,0), to.sub(from).normalize());
  return m;
}

export function curveTube(points, radius, material, segments = 20) {
  return new T.Mesh(new T.TubeGeometry(new T.CatmullRomCurve3(points.map(p => new T.Vector3(...p))), segments, radius, 6, false), material);
}

export function verticalLoft(sections, options={}) {
  return loft(sections.map(([y,rx,rz,cz=0])=>[y,rx,rz,-cz]),options).rotateX(-Math.PI/2);
}

export function thicken(geometry,depth=.018) {
  const src=geometry.attributes.position,norm=geometry.attributes.normal,n=src.count;
  const p=[],indices=[...geometry.index.array],edges=new Map();
  for(let i=0;i<n;i++)p.push(src.getX(i),src.getY(i),src.getZ(i));
  for(let i=0;i<n;i++)p.push(src.getX(i)-norm.getX(i)*depth,src.getY(i)-norm.getY(i)*depth,src.getZ(i)-norm.getZ(i)*depth);
  const original=[...indices];
  for(let i=0;i<original.length;i+=3){
    const [a,b,c]=original.slice(i,i+3);indices.push(c+n,b+n,a+n);
    for(const [u,v] of [[a,b],[b,c],[c,a]]){const key=u<v?`${u}:${v}`:`${v}:${u}`;if(edges.has(key))edges.delete(key);else edges.set(key,[u,v]);}
  }
  for(const [u,v] of edges.values())indices.push(u,v+n,v,u,u+n,v+n);
  geometry.dispose();return surface(p,indices);
}
