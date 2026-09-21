import * as THREE from '../../vendor/three/three.module.min.js';

const CYAN = 0x73d7d0;
const ICE = 0xc9e7e6;
const BLUE = 0x6e98ad;
const AMBER = 0xe4a85f;
const MODE_KEYS = ['cloud', 'question', 'query', 'boundary', 'crystal', 'system', 'grid', 'monogram'];
const TAU = Math.PI * 2;

function seededRandom(seed = 705) {
  let value = seed >>> 0;
  return () => {
    value = (value * 1664525 + 1013904223) >>> 0;
    return value / 4294967296;
  };
}

function smoothStep(value) {
  const progress = THREE.MathUtils.clamp(value, 0, 1);
  return progress * progress * (3 - 2 * progress);
}

function fract(value) {
  return value - Math.floor(value);
}

function createGlowTexture(inner = '255,255,255', outer = '115,215,208') {
  const canvas = document.createElement('canvas');
  canvas.width = 48;
  canvas.height = 48;
  const context = canvas.getContext('2d');
  const gradient = context.createRadialGradient(24, 24, 0, 24, 24, 24);
  gradient.addColorStop(0, `rgba(${inner},1)`);
  gradient.addColorStop(0.16, `rgba(${inner},.92)`);
  gradient.addColorStop(0.45, `rgba(${outer},.25)`);
  gradient.addColorStop(1, `rgba(${outer},0)`);
  context.fillStyle = gradient;
  context.fillRect(0, 0, 48, 48);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function createRibbonGeometry(lengthSegments, widthSegments) {
  const columns = widthSegments + 1;
  const positions = new Float32Array((lengthSegments + 1) * columns * 3);
  const colors = new Float32Array(positions.length);
  const indices = [];
  for (let uIndex = 0; uIndex <= lengthSegments; uIndex += 1) {
    for (let vIndex = 0; vIndex <= widthSegments; vIndex += 1) {
      const vertex = uIndex * columns + vIndex;
      const edge = Math.abs((vIndex / widthSegments) * 2 - 1);
      colors[vertex * 3] = 0.7 + edge * 0.3;
      colors[vertex * 3 + 1] = 0.7 + edge * 0.3;
      colors[vertex * 3 + 2] = 0.7 + edge * 0.3;
    }
  }
  for (let uIndex = 0; uIndex < lengthSegments; uIndex += 1) {
    for (let vIndex = 0; vIndex < widthSegments; vIndex += 1) {
      const a = uIndex * columns + vIndex;
      const b = a + columns;
      indices.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.attributes.position.setUsage(THREE.DynamicDrawUsage);
  geometry.computeVertexNormals();
  geometry.attributes.normal.setUsage(THREE.DynamicDrawUsage);
  return geometry;
}

function createLineGeometry(count) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 3), 3));
  geometry.attributes.position.setUsage(THREE.DynamicDrawUsage);
  return geometry;
}

export class SignalTopology {
  constructor({ mobile = false } = {}) {
    this.mobile = mobile;
    this.root = new THREE.Group();
    this.content = new THREE.Group();
    this.engine = new THREE.Group();
    this.root.add(this.content);
    this.content.add(this.engine);
    this.random = seededRandom();
    this.temp = new THREE.Vector3();
    this.blend = new THREE.Vector3();
    this.dummy = new THREE.Object3D();
    this.tempColor = new THREE.Color();
    this.cyanColor = new THREE.Color(CYAN);
    this.blueColor = new THREE.Color(BLUE);
    this.amberColor = new THREE.Color(AMBER);
    this.hoverTarget = 0;
    this.hoverAmount = 0;
    this.arrivalDuration = 1.45;
    this.arrivalProgress = 0;
    this.queryActive = false;
    this.queryStartTime = 0;
    this.queryProgress = 0;
    this.buildCore();
    this.buildSheath();
    this.buildParticles();
    this.buildSignal();
    this.buildAtmosphere();
  }

  buildCore() {
    this.core = new THREE.Group();
    this.engine.add(this.core);
    this.parts = [];
    this.coreMaterials = [];
    this.coreMaterial = new THREE.MeshPhysicalMaterial({
      color: 0x728f94, emissive: 0x102a30, emissiveIntensity: 0.5,
      roughness: 0.28, metalness: 0.58, clearcoat: 0.52,
      transparent: true, opacity: 0.8, depthWrite: false,
    });
    this.darkMaterial = new THREE.MeshPhysicalMaterial({
      color: 0x1c2a31, emissive: 0x08171c, emissiveIntensity: 0.4,
      roughness: 0.31, metalness: 0.78, clearcoat: 0.32,
      transparent: true, opacity: 0.94, depthWrite: true,
    });
    this.edgeMaterial = new THREE.MeshBasicMaterial({
      color: ICE, transparent: true, opacity: 0.34, depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.glassMaterial = new THREE.MeshPhysicalMaterial({
      color: 0x75b6b7, emissive: 0x0b3036, emissiveIntensity: 0.52,
      roughness: 0.14, metalness: 0.04, clearcoat: 0.9, transmission: 0.16,
      transparent: true, opacity: 0.14, depthWrite: false, side: THREE.DoubleSide,
    });
    this.bearingMaterial = this.darkMaterial.clone();
    this.bearingMaterial.color.setHex(0x111a20);
    this.bearingMaterial.roughness = 0.2;
    this.bearingMaterial.opacity = 0.98;
    this.mechanismMaterial = this.coreMaterial.clone();
    this.mechanismMaterial.color.setHex(0xa0bec0);
    this.mechanismMaterial.emissive.setHex(0x173a40);
    this.mechanismMaterial.emissiveIntensity = 0.62;
    this.mechanismMaterial.opacity = 0.9;
    this.counterMaterial = this.darkMaterial.clone();
    this.counterMaterial.color.setHex(0x435d63);
    this.counterMaterial.emissive.setHex(0x10272d);
    this.counterMaterial.emissiveIntensity = 0.54;
    this.counterMaterial.opacity = 0.92;
    this.coreMaterials.push(
      this.coreMaterial, this.darkMaterial, this.edgeMaterial, this.glassMaterial, this.bearingMaterial,
      this.mechanismMaterial, this.counterMaterial,
    );

    const v = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
    const registerPart = (group, {
      base = v(), hero = v(), explode = v(), split = v(), calibrate = v(), order = v(), schematic = v(),
      phase = 0, layer = 0,
    } = {}) => {
      group.position.copy(base);
      this.core.add(group);
      const part = {
        group, base: base.clone(), hero: hero.clone(), explode: explode.clone(), split: split.clone(),
        calibrate: calibrate.clone(), order: order.clone(), schematic: schematic.clone(), phase, layer,
        disassemblyStart: 0,
        disassemblyEnd: 1,
        baseRotation: group.rotation.clone(), baseScale: group.scale.clone(),
      };
      this.parts.push(part);
      return part;
    };
    const addBeam = (parent, start, end, radius, material = this.coreMaterial, sides = 8) => {
      const direction = end.clone().sub(start);
      const mesh = new THREE.Mesh(
        new THREE.CylinderGeometry(radius, radius * 0.94, direction.length(), sides), material,
      );
      mesh.position.copy(start).add(end).multiplyScalar(0.5);
      mesh.quaternion.setFromUnitVectors(v(0, 1, 0), direction.normalize());
      parent.add(mesh);
      return mesh;
    };
    const addXAxle = (parent, length, radius, material = this.darkMaterial, x = 0, y = 0, z = 0, sides = 12) => {
      const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, length, sides), material);
      mesh.rotation.z = Math.PI * 0.5;
      mesh.position.set(x, y, z);
      parent.add(mesh);
      return mesh;
    };
    const addOutlineBox = (parent, width, height, depth, material = this.edgeMaterial) => {
      const edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(width, height, depth));
      const lines = new THREE.LineSegments(edges, material);
      parent.add(lines);
      return lines;
    };
    const createChamferedPlate = (width, height, depth, cut = 0.08) => {
      const halfW = width * 0.5;
      const halfH = height * 0.5;
      const shape = new THREE.Shape();
      shape.moveTo(-halfW + cut, -halfH);
      shape.lineTo(halfW - cut, -halfH);
      shape.lineTo(halfW, -halfH + cut);
      shape.lineTo(halfW, halfH - cut);
      shape.lineTo(halfW - cut, halfH);
      shape.lineTo(-halfW + cut, halfH);
      shape.lineTo(-halfW, halfH - cut);
      shape.lineTo(-halfW, -halfH + cut);
      shape.closePath();
      const geometry = new THREE.ExtrudeGeometry(shape, {
        depth, steps: 1, bevelEnabled: true, bevelSegments: 1,
        bevelSize: Math.min(0.025, cut * 0.35), bevelThickness: 0.018,
      });
      geometry.translate(0, 0, -depth * 0.5);
      return geometry;
    };

    // The open block is the visual anchor. Four vertical cylinder sleeves expose the bore rhythm.
    const block = new THREE.Group();
    const blockShellMaterial = this.coreMaterial.clone();
    blockShellMaterial.color.setHex(0x526f76);
    blockShellMaterial.opacity = 0.68;
    this.coreMaterials.push(blockShellMaterial);
    const boreXs = [-0.66, -0.22, 0.22, 0.66];
    boreXs.forEach((x, index) => {
      const sleeve = new THREE.Mesh(
        new THREE.CylinderGeometry(0.19, 0.21, 0.54, this.mobile ? 14 : 20, 1, true),
        blockShellMaterial,
      );
      sleeve.position.set(x, 0.12, 0);
      const lip = new THREE.Mesh(
        new THREE.TorusGeometry(0.205, 0.03, 7, this.mobile ? 18 : 28), this.mechanismMaterial,
      );
      lip.rotation.x = Math.PI * 0.5;
      lip.position.set(x, 0.39, 0);
      const lipTrace = new THREE.Mesh(
        new THREE.TorusGeometry(0.238, 0.012, 5, this.mobile ? 18 : 28), this.edgeMaterial,
      );
      lipTrace.rotation.x = Math.PI * 0.5;
      lipTrace.position.set(x, 0.402, 0);
      const bore = new THREE.Mesh(
        new THREE.CylinderGeometry(0.145, 0.145, 0.49, this.mobile ? 12 : 18, 1, true),
        this.bearingMaterial,
      );
      bore.position.set(x, 0.13, 0);
      block.add(sleeve, lip, lipTrace, bore);
      if (index < boreXs.length - 1) {
        addBeam(block, v(x + 0.2, 0.35, 0.22), v(boreXs[index + 1] - 0.2, 0.35, 0.22), 0.026, this.coreMaterial, 6);
        addBeam(block, v(x + 0.2, -0.18, -0.21), v(boreXs[index + 1] - 0.2, -0.18, -0.21), 0.03, this.darkMaterial, 6);
      }
    });
    [-0.3, 0.32].forEach((y) => [-0.28, 0.28].forEach((z) => addXAxle(block, 1.68, 0.034, this.darkMaterial, 0, y, z, 7)));
    [-0.88, 0.88].forEach((x) => {
      addBeam(block, v(x, -0.31, -0.27), v(x, 0.38, -0.27), 0.045, blockShellMaterial, 7);
      addBeam(block, v(x, -0.31, 0.27), v(x, 0.38, 0.27), 0.045, blockShellMaterial, 7);
    });
    addOutlineBox(block, 1.84, 0.77, 0.62, this.edgeMaterial);
    this.blockPart = registerPart(block, {
      hero: v(0, -0.015, -0.02), explode: v(0, -0.08, -0.05), split: v(0, -0.04, -0.28),
      calibrate: v(0.06, -0.03, -0.08), order: v(0, -0.12, 0), schematic: v(0, -0.04, 0),
      phase: 0.04, layer: -0.18,
    });
    this.blockPart.disassemblyStart = 0.68;
    this.blockPart.disassemblyEnd = 0.94;

    // A true articulated crankshaft sits below the bores, with offset journals and counterweights.
    const crank = new THREE.Group();
    this.crankRotor = new THREE.Group();
    crank.add(this.crankRotor);
    addXAxle(this.crankRotor, 1.94, 0.055, this.bearingMaterial, 0, 0, 0, 14);
    boreXs.forEach((x, index) => {
      const angle = index % 2 ? Math.PI : 0;
      const journalY = Math.sin(angle + Math.PI * 0.5) * 0.1;
      const journalZ = Math.cos(angle) * 0.1;
      addXAxle(this.crankRotor, 0.22, 0.075, this.mechanismMaterial, x, journalY, journalZ, 12);
      [-0.115, 0.115].forEach((side) => {
        const weight = new THREE.Mesh(
          new THREE.CylinderGeometry(0.18, 0.18, 0.048, this.mobile ? 10 : 14), this.counterMaterial,
        );
        weight.rotation.z = Math.PI * 0.5;
        weight.scale.set(1, 0.72, 1);
        weight.position.set(x + side, -journalY * 0.42, -journalZ * 0.52);
        this.crankRotor.add(weight);
      });
      addBeam(this.crankRotor, v(x - 0.08, 0, 0), v(x - 0.08, journalY, journalZ), 0.035, this.mechanismMaterial, 7);
      addBeam(this.crankRotor, v(x + 0.08, 0, 0), v(x + 0.08, journalY, journalZ), 0.035, this.mechanismMaterial, 7);
    });
    this.crankPart = registerPart(crank, {
      base: v(0, -0.47, 0.03), hero: v(0, -0.09, 0.08), explode: v(0, -0.52, 0.42),
      split: v(0.1, -0.28, 0.42), calibrate: v(-0.04, -0.18, 0.24), order: v(0, -0.36, 0.02),
      schematic: v(0, -0.28, 0), phase: 0.12, layer: 0.6,
    });
    this.crankPart.disassemblyStart = 0.72;
    this.crankPart.disassemblyEnd = 0.98;

    // Each piston is an independently staged assembly: crown, wrist pin, and connecting rod.
    this.pistons = boreXs.map((x, index) => {
      const group = new THREE.Group();
      const moving = new THREE.Group();
      const crownMaterial = (index % 2 ? this.coreMaterial : this.darkMaterial).clone();
      this.coreMaterials.push(crownMaterial);
      const crown = new THREE.Mesh(
        new THREE.CylinderGeometry(0.145, 0.14, 0.17, this.mobile ? 12 : 18),
        crownMaterial,
      );
      crown.position.y = 0.02;
      const ringA = new THREE.Mesh(new THREE.TorusGeometry(0.143, 0.012, 5, 20), this.edgeMaterial);
      ringA.rotation.x = Math.PI * 0.5;
      ringA.position.y = 0.075;
      const ringB = ringA.clone();
      ringB.position.y = 0.035;
      const wrist = addXAxle(moving, 0.25, 0.025, this.edgeMaterial, 0, -0.01, 0, 7);
      const crankOffset = index % 2 ? -0.1 : 0.1;
      addBeam(moving, v(0, -0.08, 0), v(0, -0.51, crankOffset), 0.035, this.mechanismMaterial, 7);
      const bigEnd = new THREE.Mesh(new THREE.TorusGeometry(0.075, 0.025, 6, 16), this.darkMaterial);
      bigEnd.rotation.x = Math.PI * 0.5;
      bigEnd.position.set(0, -0.51, crankOffset);
      moving.add(crown, ringA, ringB, wrist, bigEnd);
      group.add(moving);
      const sign = index % 2 ? 1 : -1;
      const part = registerPart(group, {
        base: v(x, 0.34, 0), hero: v((index - 1.5) * 0.035, 0.05 + index * 0.012, sign * (0.08 + index * 0.012)),
        explode: v((index - 1.5) * 0.12, 0.46 + index * 0.045, sign * 0.38),
        split: v(index < 2 ? -0.4 : 0.4, 0.18, sign * 0.28),
        calibrate: v((index - 1.5) * 0.1, 0.28, sign * 0.18),
        order: v(0, (index - 1.5) * 0.12, sign * 0.04),
        schematic: v(0, (index - 1.5) * 0.18, sign * 0.02), phase: 0.2 + index * 0.1, layer: 0.25 + index * 0.08,
      });
      part.disassemblyStart = 0.22 + index * 0.045;
      part.disassemblyEnd = 0.5 + index * 0.045;
      return { ...part, moving, crown, index };
    });

    // Twin overhead cams and valve followers read as a separate precision layer.
    const valveTrain = new THREE.Group();
    this.camRotors = [];
    [-0.17, 0.17].forEach((z, railIndex) => {
      const rotor = new THREE.Group();
      addXAxle(rotor, 1.72, 0.034, this.bearingMaterial, 0, 0, z, 12);
      boreXs.forEach((x, index) => {
        const lobe = new THREE.Mesh(
          new THREE.SphereGeometry(0.095, this.mobile ? 8 : 12, this.mobile ? 6 : 8), this.coreMaterial,
        );
        lobe.scale.set(0.32, 1, 0.72);
        lobe.position.set(x, index % 2 ? 0.035 : -0.035, z);
        lobe.rotation.x = (index * 0.78) + railIndex * 0.4;
        rotor.add(lobe);
      });
      valveTrain.add(rotor);
      this.camRotors.push(rotor);
    });
    boreXs.forEach((x) => [-0.17, 0.17].forEach((z) => {
      const follower = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.04, 0.22, 7), this.darkMaterial);
      follower.position.set(x, -0.18, z);
      valveTrain.add(follower);
    }));
    [-0.92, 0.92].forEach((x) => addBeam(valveTrain, v(x, -0.22, -0.27), v(x, 0.12, 0.27), 0.036, this.coreMaterial, 7));
    this.valvePart = registerPart(valveTrain, {
      base: v(0, 0.66, 0), hero: v(0, 0.16, -0.05), explode: v(0.08, 0.72, -0.32),
      split: v(0, 0.56, -0.32), calibrate: v(0.16, 0.42, -0.2), order: v(0, 0.42, 0),
      schematic: v(0, 0.4, 0), phase: 0.62, layer: -0.45,
    });
    this.valvePart.disassemblyStart = 0.06;
    this.valvePart.disassemblyEnd = 0.32;

    // End case and accessory rotor intentionally sit on different depth planes.
    const endCase = new THREE.Group();
    const endPlate = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.78, 0.64), blockShellMaterial);
    endPlate.position.x = 0.02;
    addOutlineBox(endCase, 0.22, 0.8, 0.66, this.edgeMaterial);
    const endBearing = new THREE.Mesh(new THREE.TorusGeometry(0.23, 0.055, 8, 24), this.bearingMaterial);
    endBearing.rotation.y = Math.PI * 0.5;
    endBearing.position.x = 0.13;
    endCase.add(endPlate, endBearing);
    this.endCasePart = registerPart(endCase, {
      base: v(1.08, -0.02, -0.04), hero: v(0.11, 0.02, -0.12), explode: v(0.68, 0.05, -0.46),
      split: v(0.6, -0.08, -0.3), calibrate: v(0.38, 0.18, -0.3), order: v(0.34, 0.1, -0.08),
      schematic: v(0.34, 0, 0), phase: 0.48, layer: -0.72,
    });
    this.endCasePart.disassemblyStart = 0.46;
    this.endCasePart.disassemblyEnd = 0.7;
    const accessory = new THREE.Group();
    [0.3, 0.22, 0.13].forEach((radius, index) => {
      const rotor = new THREE.Mesh(
        new THREE.TorusGeometry(radius, index === 0 ? 0.045 : 0.02, 7, this.mobile ? 18 : 28),
        index === 1 ? this.edgeMaterial : this.darkMaterial,
      );
      rotor.rotation.y = Math.PI * 0.5;
      rotor.position.x = index * -0.045;
      accessory.add(rotor);
    });
    this.accessoryRotor = accessory;
    this.accessoryPart = registerPart(accessory, {
      base: v(-1.12, -0.14, 0.06), hero: v(-0.18, 0.04, 0.2), explode: v(-0.76, 0.08, 0.58),
      split: v(-0.66, -0.04, 0.44), calibrate: v(-0.45, 0.28, 0.34), order: v(-0.4, 0.08, 0.08),
      schematic: v(-0.4, 0, 0), phase: 0.34, layer: 0.92,
    });
    this.accessoryPart.disassemblyStart = 0.42;
    this.accessoryPart.disassemblyEnd = 0.66;

    // Cutaway covers and pan retain the exploded-diagram silhouette without hiding the mechanism.
    const coverSpecs = [
      { name: 'head', base: v(0, 0.88, -0.03), size: [1.88, 0.16, 0.64], hero: v(0, 0.2, -0.1), explode: v(0, 0.88, -0.55), split: v(0, 0.65, -0.42), order: v(0, 0.62, 0), schematic: v(0, 0.62, 0), layer: -0.7 },
      { name: 'pan', base: v(0, -0.78, 0.02), size: [1.72, 0.18, 0.56], hero: v(0, -0.1, 0.12), explode: v(0, -0.75, 0.55), split: v(0, -0.58, 0.4), order: v(0, -0.54, 0), schematic: v(0, -0.52, 0), layer: 0.72 },
      { name: 'side', base: v(0.08, -0.08, 0.38), size: [1.58, 0.54, 0.1], hero: v(0, 0, 0.18), explode: v(0.14, 0.06, 0.82), split: v(0.18, 0, 0.68), order: v(0.08, 0, 0.38), schematic: v(0, 0, 0.42), layer: 1.2 },
    ];
    this.plates = coverSpecs.map((spec, index) => {
      const group = new THREE.Group();
      const material = this.glassMaterial.clone();
      material.color.setHex(index === 1 ? 0x617f85 : 0x76aaa9);
      this.coreMaterials.push(material);
      const plateGeometry = createChamferedPlate(
        spec.size[0], spec.size[1], spec.size[2], index === 2 ? 0.12 : 0.08,
      );
      const shell = new THREE.Mesh(plateGeometry, material);
      const wireMaterial = this.edgeMaterial.clone();
      wireMaterial.opacity = 0.14;
      this.coreMaterials.push(wireMaterial);
      const wire = new THREE.LineSegments(new THREE.EdgesGeometry(plateGeometry), wireMaterial);
      group.add(shell, wire);
      if (index < 2) {
        const railY = index === 0 ? -spec.size[1] * 0.16 : spec.size[1] * 0.16;
        [-0.78, 0.78].forEach((x) => {
          const rail = new THREE.Mesh(
            new THREE.BoxGeometry(0.18, spec.size[1] * 0.72, spec.size[2] * 1.04),
            this.darkMaterial,
          );
          rail.position.set(x, railY, 0);
          group.add(rail);
        });
      }
      const part = registerPart(group, {
        base: spec.base, hero: spec.hero, explode: spec.explode, split: spec.split,
        calibrate: spec.explode.clone().multiplyScalar(0.52), order: spec.order, schematic: spec.schematic,
        phase: 0.05 + index * 0.23, layer: spec.layer,
      });
      const timings = [[0, 0.24], [0.76, 0.99], [0.5, 0.74]];
      [part.disassemblyStart, part.disassemblyEnd] = timings[index];
      return { ...part, shell, wire, material, wireMaterial, index };
    });

    // A near-side manifold gives the silhouette branching rather than barrel-like continuity.
    const manifold = new THREE.Group();
    boreXs.forEach((x, index) => {
      const pipe = new THREE.Mesh(
        new THREE.TorusGeometry(0.2 + index * 0.008, 0.025, 7, this.mobile ? 16 : 22, Math.PI * 0.72),
        index === 3 ? this.coreMaterial : this.darkMaterial,
      );
      pipe.position.set(x, 0.04, 0);
      pipe.rotation.set(Math.PI * 0.5, 0, index % 2 ? 0.22 : -0.22);
      manifold.add(pipe);
    });
    addXAxle(manifold, 1.52, 0.045, this.coreMaterial, 0.12, -0.15, 0.16, 9);
    this.manifoldPart = registerPart(manifold, {
      base: v(0, 0.04, 0.32), hero: v(-0.04, 0.03, 0.13), explode: v(-0.18, 0.28, 0.72),
      split: v(-0.28, 0.16, 0.6), calibrate: v(-0.2, 0.34, 0.48), order: v(0.02, 0.18, 0.28),
      schematic: v(0, 0.18, 0.36), phase: 0.53, layer: 1.05,
    });
    this.manifoldPart.disassemblyStart = 0.5;
    this.manifoldPart.disassemblyEnd = 0.74;

    // Fasteners and sensor pins provide the final small scale and stay tied to the block family.
    const nodeSpecs = [
      [-0.86, 0.39, 0.27], [-0.44, 0.42, 0.28], [0, 0.42, 0.28], [0.44, 0.42, 0.28], [0.86, 0.39, 0.27],
      [-0.82, -0.28, 0.28], [-0.28, -0.31, 0.29], [0.28, -0.31, 0.29], [0.82, -0.28, 0.28],
    ];
    this.nodes = nodeSpecs.map((position, index) => {
      const group = new THREE.Group();
      const nodeMaterial = index % 4 === 0 ? this.coreMaterial.clone() : this.darkMaterial.clone();
      nodeMaterial.opacity = 0.92;
      this.coreMaterials.push(nodeMaterial);
      const node = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.052, 0.09, 6), nodeMaterial);
      node.rotation.x = Math.PI * 0.5;
      const pin = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.15, 5), this.edgeMaterial);
      pin.rotation.x = Math.PI * 0.5;
      pin.position.z = 0.08;
      group.add(node, pin);
      const base = v(...position);
      const side = index % 2 ? -1 : 1;
      const part = registerPart(group, {
        base, hero: v((index - 4) * 0.012, side * 0.025, 0.05),
        explode: v((index - 4) * 0.055, side * 0.25, 0.52 + (index % 3) * 0.08),
        split: v(index < 5 ? 0 : (index - 6.5) * 0.08, side * 0.2, 0.45),
        calibrate: v((index - 4) * 0.04, 0.2, 0.34), order: v(0, (index - 4) * 0.07, 0.16),
        schematic: v(0, (index - 4) * 0.09, 0.08), phase: 0.08 + (index % 5) * 0.13, layer: 0.85,
      });
      part.disassemblyStart = 0.54 + (index % 3) * 0.035;
      part.disassemblyEnd = 0.78 + (index % 3) * 0.035;
      return { ...part, node, pin, nodeMaterial, index };
    });

    // Compatibility groups retained for the existing update contract.
    this.stages = [this.blockPart, this.crankPart, this.valvePart, this.endCasePart];
    this.chambers = this.pistons;
    this.couplings = [this.endCasePart, this.accessoryPart, this.manifoldPart];
    this.intakeMaterial = this.edgeMaterial.clone();
    this.intake = new THREE.Mesh(new THREE.TorusGeometry(0.36, 0.015, 5, 34, 4.5), this.intakeMaterial);
    this.intake.rotation.y = Math.PI * 0.5;
    this.intake.rotation.x = -0.62;
    this.intake.position.set(-1.58, 0.18, 0.16);
    this.core.add(this.intake);
    this.outletMaterial = this.edgeMaterial.clone();
    this.outletMaterial.color.setHex(AMBER);
    this.outlet = new THREE.Mesh(new THREE.TorusGeometry(0.25, 0.017, 6, 30, 3.5), this.outletMaterial);
    this.outlet.rotation.y = Math.PI * 0.5;
    this.outlet.rotation.x = 1.05;
    this.outlet.position.set(1.48, -0.12, -0.06);
    this.core.add(this.outlet);
    this.coreMaterials.forEach((material) => {
      material.userData.engineOpacity = material.opacity;
    });
  }

  buildSheath() {
    this.sheath = new THREE.Group();
    this.engine.add(this.sheath);
    const filamentCount = this.mobile ? 9 : 12;
    const lengthSegments = this.mobile ? 38 : 52;
    this.filaments = [];
    for (let index = 0; index < filamentCount; index += 1) {
      const geometry = createLineGeometry(lengthSegments + 1);
      const material = new THREE.LineBasicMaterial({
        color: index % 4 === 0 ? 0xa6dcda : (index % 3 === 0 ? 0x6cb9bd : 0x5c879d),
        transparent: true,
        opacity: this.mobile ? 0.21 : 0.19,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      const line = new THREE.Line(geometry, material);
      line.frustumCulled = false;
      line.renderOrder = 2 + index * 0.01;
      this.sheath.add(line);
      this.filaments.push({
        geometry, material, line, index, count: filamentCount,
        phase: (index / filamentCount) * TAU,
        lengthSegments, direction: index % 2 ? -1 : 1,
        lane: index - (filamentCount - 1) * 0.5,
      });
    }
  }

  buildParticles() {
    this.particleCount = this.mobile ? 68 : 104;
    this.samples = [];
    for (let index = 0; index < this.particleCount; index += 1) {
      const lane = index % 9;
      this.samples.push({
        phase: this.random(), angle: this.random() * TAU,
        speed: 0.038 + this.random() * 0.052, lane: lane - 4,
        radius: 0.22 + this.random() * 0.78, jitter: 0.08 + this.random() * 0.42,
        size: 0.38 + this.random() * 0.7,
        relevant: index % 5 === 0 || index % 13 === 0,
      });
    }
    const geometry = new THREE.IcosahedronGeometry(this.mobile ? 0.022 : 0.026, 1);
    this.particleMaterial = new THREE.MeshPhysicalMaterial({
      color: 0xffffff, vertexColors: true, emissive: 0x123d43, emissiveIntensity: 0.78,
      roughness: 0.32, metalness: 0.02, clearcoat: 0.28,
      transparent: true, opacity: 0.76, depthWrite: false,
    });
    this.particles = new THREE.InstancedMesh(geometry, this.particleMaterial, this.particleCount);
    this.particles.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.particles.frustumCulled = false;
    this.particles.renderOrder = 4;
    this.engine.add(this.particles);
  }

  buildSignal() {
    this.signal = new THREE.Group();
    this.engine.add(this.signal);
    this.signalCore = new THREE.Mesh(
      new THREE.IcosahedronGeometry(this.mobile ? 0.07 : 0.085, 2),
      new THREE.MeshBasicMaterial({
        color: 0xffbd68, transparent: true, opacity: 0, depthWrite: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    this.signalHalo = new THREE.Sprite(new THREE.SpriteMaterial({
      color: AMBER, map: createGlowTexture('255,225,171', '228,168,95'),
      transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending,
    }));
    this.signalHalo.scale.setScalar(this.mobile ? 0.74 : 0.92);
    this.signal.add(this.signalCore, this.signalHalo);
    this.signalTrailGeometry = createLineGeometry(18);
    this.signalTrailMaterial = new THREE.LineBasicMaterial({
      color: AMBER, transparent: true, opacity: 0, depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.signalTrail = new THREE.Line(this.signalTrailGeometry, this.signalTrailMaterial);
    this.engine.add(this.signalTrail);
  }

  buildAtmosphere() {
    const count = this.mobile ? 18 : 30;
    const positions = new Float32Array(count * 3);
    for (let index = 0; index < count; index += 1) {
      positions[index * 3] = (this.random() - 0.5) * 6.8;
      positions[index * 3 + 1] = (this.random() - 0.5) * 4.6;
      positions[index * 3 + 2] = (this.random() - 0.5) * 3.8 - 0.8;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.ambientPoints = new THREE.Points(geometry, new THREE.PointsMaterial({
      color: 0x6ca6aa, map: createGlowTexture(), size: this.mobile ? 0.034 : 0.044,
      transparent: true, opacity: 0.08, depthWrite: false,
      blending: THREE.AdditiveBlending, sizeAttenuation: true,
    }));
    this.content.add(this.ambientPoints);
  }

  sampleFilamentMode(target, mode, filament, u, time, motion) {
    const phase = filament.phase;
    const normalizedLane = filament.lane / Math.max(1, (filament.count - 1) * 0.5);
    const xBase = (u - 0.5) * 3.78;
    const centerEnvelope = Math.sin(Math.PI * u);
    const inputTurbulence = Math.pow(1 - u, 2.4);
    let x = xBase;
    let y = 0;
    let z = 0;
    let angle = phase;
    let radius = 0.54;
    switch (mode) {
      case 0: // Attract: turbulent threads bend into the first chamber.
        angle = phase + u * (1.3 + filament.direction * 0.18);
        y = normalizedLane * 0.42
          + Math.sin(u * Math.PI * (1.4 + (filament.index % 3) * 0.22) + phase) * 0.11 * centerEnvelope
          + inputTurbulence * Math.sin(phase * 2.3 + time * 0.38) * 0.18 * motion;
        z = (filament.index % 3 - 1) * 0.2
          + Math.cos(angle) * 0.08 * centerEnvelope
          + inputTurbulence * Math.cos(phase * 1.6 + time * 0.31) * 0.14 * motion;
        if (filament.index % 4 === 0) y -= centerEnvelope * 0.38;
        if (filament.index % 5 === 0) y += centerEnvelope * 0.34;
        break;
      case 1: // Converge: the bundle narrows sharply at the measurement chamber.
        angle = phase + u * 1.2 + Math.sin(time * 0.18 + phase) * 0.05 * motion;
        radius = 0.54 - Math.pow(centerEnvelope, 1.15) * 0.22;
        y = normalizedLane * radius + Math.sin(u * Math.PI * 1.6 + phase) * 0.08;
        z = (filament.index % 4 - 1.5) * 0.18 + Math.cos(angle) * 0.055;
        break;
      case 2: // Transmit: coherent streams travel through each stage.
        angle = phase + u * 1.1 + time * 0.04 * filament.direction * motion;
        y = normalizedLane * 0.3 + Math.sin(u * Math.PI * 2 + phase) * 0.045;
        z = (filament.index % 3 - 1) * 0.13 + Math.cos(angle) * 0.045;
        x += Math.sin(Math.PI * u) * (filament.index % 2 ? -0.04 : 0.05);
        break;
      case 3: { // Split/evaluate: two separated evidence banks.
        const bank = filament.index % 2 ? -1 : 1;
        y = bank * (0.25 + centerEnvelope * 0.3) + normalizedLane * 0.055;
        z = normalizedLane * 0.24 + Math.sin(Math.PI * u + phase) * 0.065;
        break;
      }
      case 4: // Calibrate: a measured feedback loop returns toward the center.
        angle = phase + u * 2.65 + Math.sin(time * 0.14) * 0.05 * motion;
        radius = 0.37 + Math.sin(u * Math.PI * 4 + phase) * 0.045;
        x = (u - 0.5) * 3.25 - Math.sin(Math.PI * u) * (filament.index % 3 === 0 ? 0.28 : 0);
        y = Math.sin(angle) * radius + Math.sin(u * Math.PI * 2) * 0.12;
        z = Math.cos(angle) * radius;
        break;
      case 5: // Order: stable service lanes with measured spacing.
        y = normalizedLane * 0.46;
        z = (filament.index % 3 - 1) * 0.11 + Math.sin(Math.PI * u + phase) * 0.026;
        break;
      case 6: // Resolve: flattened traces retain a gentle organic cadence.
        y = normalizedLane * 0.27 + Math.sin(u * Math.PI * 2 + phase) * 0.018;
        z = normalizedLane * 0.025;
        x = (u - 0.5) * 3.95;
        break;
      case 7: // Condense: the field folds into a compact seed.
      default:
        x = (u - 0.5) * 1.08;
        angle = phase + u * 3.4;
        radius = 0.2 + centerEnvelope * 0.16;
        y = Math.sin(angle) * radius;
        z = Math.cos(angle) * radius;
        break;
    }
    const ripple = Math.sin(u * Math.PI * 3 + phase + time * 0.25) * 0.012 * motion * centerEnvelope;
    target.set(x, y, z + ripple);
    return target;
  }

  updateSheath(time, state, reducedMotion) {
    const motion = reducedMotion ? 0 : (state.motion ?? 1);
    const grid = state.grid ?? 0;
    const monogram = state.monogram ?? 0;
    const gridVisibility = this.mobile ? 1 - grid : 1 - grid * 0.82;
    const arrival = THREE.MathUtils.lerp(0.44, 1, this.arrivalProgress);
    this.filaments.forEach((filament) => {
      const positions = filament.geometry.attributes.position.array;
      let offset = 0;
      for (let uIndex = 0; uIndex <= filament.lengthSegments; uIndex += 1) {
        const u = uIndex / filament.lengthSegments;
        this.blend.set(0, 0, 0);
        let total = 0;
        MODE_KEYS.forEach((key, mode) => {
          const weight = state[key] ?? 0;
          if (weight <= 0.001) return;
          this.sampleFilamentMode(this.temp, mode, filament, u, time, motion);
          this.blend.addScaledVector(this.temp, weight);
          total += weight;
        });
        if (total <= 0.001) {
          this.sampleFilamentMode(this.blend, 0, filament, u, time, motion);
        } else if (Math.abs(total - 1) > 0.001) {
          this.blend.multiplyScalar(1 / total);
        }
        if (arrival < 1) {
          this.blend.x -= (1 - arrival) * (1.2 + filament.index * 0.045);
          this.blend.y *= arrival;
          this.blend.z *= arrival;
        }
        positions[offset] = this.blend.x;
        positions[offset + 1] = this.blend.y;
        positions[offset + 2] = this.blend.z;
        offset += 3;
      }
      filament.geometry.attributes.position.needsUpdate = true;
      filament.material.opacity = (this.mobile ? 0.22 : 0.19) * arrival
        * gridVisibility * (1 - monogram * 0.04) * (1 + this.hoverAmount * 0.08);
    });
  }

  sampleParticleMode(target, mode, sample, progress, time, motion) {
    let journeyX;
    if (progress < 0.48) {
      const intake = progress / 0.48;
      journeyX = -2.08 + Math.pow(intake, 1.75) * 1.05;
    } else if (progress < 0.82) {
      const processing = (progress - 0.48) / 0.34;
      journeyX = -1.03 + processing * 1.64;
    } else {
      const release = (progress - 0.82) / 0.18;
      journeyX = 0.61 + release * 1.28;
    }
    const x = journeyX;
    const center = Math.sin(Math.PI * progress);
    const organized = smoothStep((progress - 0.35) / 0.4);
    let radius;
    let angle;
    switch (mode) {
      case 0: // Attract.
        radius = THREE.MathUtils.lerp(sample.radius * 1.22, 0.27 + Math.abs(sample.lane) * 0.026, organized);
        angle = sample.angle + progress * 3.3 + time * 0.13 * motion;
        target.set(
          x,
          Math.sin(angle) * radius + (1 - organized) * Math.sin(time * 0.42 + sample.angle) * sample.jitter * motion + Math.sin(progress * Math.PI * 1.3) * 0.1,
          Math.cos(angle) * radius + (1 - organized) * Math.cos(time * 0.36 + sample.angle) * sample.jitter * motion,
        );
        break;
      case 1: // Converge.
        radius = 0.78 - Math.pow(center, 1.15) * 0.59;
        angle = sample.angle + progress * 2.15;
        target.set(x, Math.sin(angle) * radius + Math.sin(progress * Math.PI * 1.3) * 0.09, Math.cos(angle) * radius);
        break;
      case 2: // Transmit.
        radius = 0.16 + Math.abs(sample.lane) * 0.038;
        angle = sample.angle + progress * 1.45;
        target.set(x, Math.sin(angle) * radius + Math.sin(progress * Math.PI * 1.3) * 0.08, Math.cos(angle) * radius);
        break;
      case 3: // Split/evaluate.
        target.set(x, (sample.lane < 0 ? -1 : 1) * (0.22 + center * 0.32), sample.lane * 0.085);
        break;
      case 4: // Calibrate/feedback.
        radius = 0.23 + (Math.abs(sample.lane) % 3) * 0.072;
        angle = sample.angle + progress * 3.7;
        target.set(x - Math.sin(progress * Math.PI) * (sample.relevant ? 0.24 : 0), Math.sin(angle) * radius, Math.cos(angle) * radius);
        break;
      case 5: // Order lanes.
        target.set(x, sample.lane * 0.085, (sample.lane % 3 - 1) * 0.082);
        break;
      case 6: // Resolve traces.
        target.set(x, sample.lane * 0.072, sample.lane * 0.009);
        break;
      case 7: // Condense seed.
      default:
        angle = sample.angle + progress * TAU;
        radius = 0.13 + progress * 0.18;
        target.set((progress - 0.5) * 0.88, Math.sin(angle) * radius, Math.cos(angle) * radius);
        break;
    }
    return target;
  }

  updateParticles(time, state, reducedMotion) {
    const motion = reducedMotion ? 0 : (state.motion ?? 1);
    const query = state.queryGlow ?? 0;
    const boundary = state.boundary ?? 0;
    const system = state.system ?? 0;
    const grid = state.grid ?? 0;
    const monogram = state.monogram ?? 0;
    const arrival = THREE.MathUtils.lerp(0.2, 1, this.arrivalProgress);
    this.samples.forEach((sample, index) => {
      const progress = reducedMotion ? sample.phase : fract(sample.phase + time * sample.speed * (0.35 + motion * 0.65));
      this.blend.set(0, 0, 0);
      let total = 0;
      MODE_KEYS.forEach((key, mode) => {
        const weight = state[key] ?? 0;
        if (weight <= 0.001) return;
        this.sampleParticleMode(this.temp, mode, sample, progress, time, motion);
        this.blend.addScaledVector(this.temp, weight);
        total += weight;
      });
      if (total <= 0.001) {
        this.sampleParticleMode(this.blend, 0, sample, progress, time, motion);
      } else if (Math.abs(total - 1) > 0.001) {
        this.blend.multiplyScalar(1 / total);
      }
      if (arrival < 1) this.blend.x -= (1 - arrival) * (0.8 + sample.jitter);
      this.dummy.position.copy(this.blend);
      const speedPulse = reducedMotion ? 1 : 1 + Math.sin(time * 1.1 + sample.angle) * 0.08 * motion;
      const focus = sample.relevant ? 1 + query * 0.45 + boundary * 0.16 : 1;
      const traceScale = 1 - grid * 0.18 - monogram * 0.08;
      const output = smoothStep((progress - 0.72) / 0.2);
      const sparseRelease = sample.relevant ? 1 : THREE.MathUtils.lerp(1, 0.16, output);
      const baseScale = sample.size * speedPulse * focus * traceScale * arrival * sparseRelease;
      const velocityStretch = 1 + smoothStep((progress - 0.4) / 0.46) * 2.2;
      this.dummy.rotation.set(0, 0, 0);
      this.dummy.scale.set(baseScale * velocityStretch, baseScale, baseScale);
      this.dummy.updateMatrix();
      this.particles.setMatrixAt(index, this.dummy.matrix);
      const amber = output * (sample.relevant ? 0.38 + query * 0.6 + boundary * 0.18 : 0.02 + system * 0.04);
      this.tempColor.copy(index % 3 === 0 ? this.blueColor : this.cyanColor).lerp(this.amberColor, amber);
      if (!sample.relevant && output > 0) this.tempColor.multiplyScalar(1 - output * 0.52);
      this.particles.setColorAt(index, this.tempColor);
    });
    this.particles.instanceMatrix.needsUpdate = true;
    this.particles.instanceColor.needsUpdate = true;
    this.particleMaterial.opacity = 0.68 * arrival * (this.mobile ? 1 - grid : 1 - grid * 0.82);
  }

  updateCore(time, state, reducedMotion, pointer) {
    const motion = reducedMotion ? 0 : (state.motion ?? 1);
    const cloud = state.cloud ?? 0;
    const boundary = state.boundary ?? 0;
    const crystal = state.crystal ?? 0;
    const system = state.system ?? 0;
    const grid = state.grid ?? 0;
    const monogram = state.monogram ?? 0;
    const question = state.questionFocus ?? 0;
    const query = state.queryGlow ?? 0;
    const gridVisibility = this.mobile ? 1 - grid : 1 - grid * 0.82;
    const arrival = THREE.MathUtils.lerp(0.28, 1, this.arrivalProgress);
    const partEase = reducedMotion ? 1 : 0.07;
    const spread = this.mobile ? 0.58 : 1;
    const pointerX = reducedMotion ? 0 : pointer.x;
    const pointerY = reducedMotion ? 0 : pointer.y;
    const engineYaw = (cloud * 0.12 + question * 0.16 + query * (1 - this.queryProgress) * 0.07) * spread;
    const engineTilt = (cloud * -0.025 + question * -0.045) * spread;
    this.engine.rotation.y = THREE.MathUtils.lerp(this.engine.rotation.y, engineYaw, partEase);
    this.engine.rotation.x = THREE.MathUtils.lerp(this.engine.rotation.x, engineTilt, partEase);

    this.parts.forEach((part, index) => {
      const stagedAssembly = smoothStep((this.queryProgress - part.phase) / 0.28);
      const questionDisassembly = smoothStep(
        (question - part.disassemblyStart) / Math.max(0.001, part.disassemblyEnd - part.disassemblyStart),
      );
      const exploded = questionDisassembly + query * (1 - stagedAssembly) * 0.86;
      this.temp.copy(part.base)
        .addScaledVector(part.hero, cloud * spread)
        .addScaledVector(part.explode, exploded * spread)
        .addScaledVector(part.split, boundary * spread)
        .addScaledVector(part.calibrate, crystal * spread)
        .addScaledVector(part.order, system * spread * 1.12)
        .addScaledVector(part.schematic, grid * spread);

      // Pointer motion separates depth planes locally, subordinate to the authored scene pose.
      this.temp.x += pointerX * part.layer * 0.045;
      this.temp.y += pointerY * Math.abs(part.layer) * 0.035;
      this.temp.z += (pointerX * 0.06 - pointerY * 0.035) * part.layer;

      // Diagnostics breathe around the selected mechanism without collapsing into a ring.
      if (crystal > 0.001) {
        const calibrationPhase = part.phase * TAU + time * 0.16 * motion;
        this.temp.y += Math.sin(calibrationPhase) * 0.045 * crystal * motion;
        this.temp.z += Math.cos(calibrationPhase * 0.83) * 0.065 * crystal * motion;
        this.temp.x += Math.sin(calibrationPhase * 0.47) * 0.025 * crystal * motion;
      }
      part.group.position.lerp(this.temp, partEase);

      const stagger = Math.sin(time * (0.13 + part.phase * 0.04) + part.phase * 8.2) * 0.025 * motion;
      const targetRotationX = part.baseRotation.x
        + exploded * part.explode.z * 0.2 + boundary * part.split.y * 0.12
        + crystal * stagger + pointerY * part.layer * 0.018;
      const targetRotationY = part.baseRotation.y
        + exploded * part.explode.y * 0.14 + system * (index % 2 ? -0.045 : 0.045)
        + crystal * stagger * 0.7 + pointerX * part.layer * 0.025;
      const targetRotationZ = part.baseRotation.z
        + exploded * part.explode.x * 0.12 + grid * (index % 2 ? -0.025 : 0.025);
      part.group.rotation.x = THREE.MathUtils.lerp(part.group.rotation.x, targetRotationX, partEase);
      part.group.rotation.y = THREE.MathUtils.lerp(part.group.rotation.y, targetRotationY, partEase);
      part.group.rotation.z = THREE.MathUtils.lerp(part.group.rotation.z, targetRotationZ, partEase);

      const schematicScale = 1 - grid * 0.12;
      const finalSeat = 1 + monogram * (index % 5 === 0 ? 0.025 : 0);
      this.temp.set(
        part.baseScale.x * schematicScale * finalSeat,
        part.baseScale.y * schematicScale * finalSeat,
        part.baseScale.z * schematicScale * finalSeat,
      );
      part.group.scale.lerp(this.temp, partEase);
    });

    this.coreMaterials.forEach((material) => {
      material.opacity = material.userData.engineOpacity * arrival * gridVisibility;
    });
    this.coreMaterial.opacity = (0.78 + query * 0.07 + system * 0.05) * arrival * gridVisibility;
    this.darkMaterial.opacity = (0.92 + boundary * 0.04) * arrival * gridVisibility;
    this.bearingMaterial.opacity = 0.96 * arrival * gridVisibility;
    this.edgeMaterial.opacity = (0.29 + question * 0.05 + query * 0.04) * arrival * gridVisibility;
    this.glassMaterial.opacity = (0.12 + query * 0.03 + crystal * 0.035) * arrival * gridVisibility;
    const internalReveal = THREE.MathUtils.clamp(cloud + question + query, 0, 1);
    this.mechanismMaterial.opacity = (0.88 + internalReveal * 0.1) * arrival * gridVisibility;
    this.mechanismMaterial.emissiveIntensity = 0.58 + internalReveal * 0.28 + crystal * 0.12;
    this.counterMaterial.opacity = (0.88 + internalReveal * 0.08) * arrival * gridVisibility;
    this.counterMaterial.emissiveIntensity = 0.5 + internalReveal * 0.24;

    // The machine remains alive at rest: crank, piston, and cams move as one restrained mechanism.
    const mechanicalTime = reducedMotion ? 0.72 : time * (0.42 + motion * 0.16);
    this.crankRotor.rotation.x = mechanicalTime;
    this.pistons.forEach((piston, index) => {
      const stroke = Math.sin(mechanicalTime + index * Math.PI) * 0.045 * motion;
      piston.moving.position.y = stroke;
      piston.moving.rotation.x = Math.sin(mechanicalTime + index * Math.PI) * 0.035 * motion;
      piston.crown.material.emissiveIntensity = 0.42
        + query * smoothStep((this.queryProgress - piston.phase) / 0.22) * 0.46
        + crystal * (index === 1 ? 0.32 : 0.08);
    });
    this.camRotors.forEach((rotor, index) => {
      rotor.rotation.x = mechanicalTime * (index ? -0.52 : 0.52) + index * 0.6;
    });
    this.accessoryRotor.rotation.x = mechanicalTime * -0.72;

    this.plates.forEach((plate, index) => {
      const reveal = 0.1 + question * 0.08 + boundary * 0.04 + crystal * 0.03 + system * 0.025;
      plate.material.opacity = reveal * arrival * gridVisibility;
      plate.wireMaterial.opacity = (0.1 + question * 0.08 + grid * 0.1) * arrival * gridVisibility;
      plate.shell.rotation.z = Math.sin(time * 0.12 + index) * 0.008 * motion;
    });
    this.nodes.forEach((node, index) => {
      node.nodeMaterial.emissiveIntensity = 0.46 + query * smoothStep((this.queryProgress - node.phase) / 0.2) * 0.5;
      node.node.rotation.z = (index % 2 ? -1 : 1) * time * 0.08 * motion;
    });
    this.intake.scale.setScalar(1 + cloud * 0.08 + question * 0.04);
    this.intakeMaterial.opacity = (0.17 + cloud * 0.17 + question * 0.1) * arrival * gridVisibility;
    this.outletMaterial.opacity = (0.11 + query * 0.3 + boundary * 0.07 + crystal * 0.05 + monogram * 0.08)
      * arrival * gridVisibility;
  }

  sampleEnginePath(target, phase) {
    const p = THREE.MathUtils.clamp(phase, 0, 1);
    let local;
    if (p < 0.18) {
      local = smoothStep(p / 0.18);
      target.set(
        THREE.MathUtils.lerp(-1.58, -0.66, local),
        THREE.MathUtils.lerp(0.18, 0.34, local) + Math.sin(local * Math.PI) * 0.08,
        THREE.MathUtils.lerp(0.16, 0.02, local),
      );
    } else if (p < 0.34) {
      local = smoothStep((p - 0.18) / 0.16);
      target.set(-0.66, THREE.MathUtils.lerp(0.34, 0.72, local), 0.02 - Math.sin(local * Math.PI) * 0.12);
    } else if (p < 0.5) {
      local = smoothStep((p - 0.34) / 0.16);
      target.set(
        THREE.MathUtils.lerp(-0.66, 0.24, local),
        0.72 + Math.sin(local * Math.PI) * 0.06,
        THREE.MathUtils.lerp(-0.1, 0.12, local),
      );
    } else if (p < 0.66) {
      local = smoothStep((p - 0.5) / 0.16);
      target.set(0.24 + Math.sin(local * Math.PI) * 0.08, THREE.MathUtils.lerp(0.72, -0.47, local), 0.12 + Math.sin(local * TAU) * 0.08);
    } else if (p < 0.84) {
      local = smoothStep((p - 0.66) / 0.18);
      target.set(
        THREE.MathUtils.lerp(0.24, 1.08, local),
        -0.47 + Math.sin(local * Math.PI) * 0.06,
        THREE.MathUtils.lerp(0.12, -0.05, local),
      );
    } else {
      local = smoothStep((p - 0.84) / 0.16);
      target.set(
        THREE.MathUtils.lerp(1.08, 1.58, local),
        THREE.MathUtils.lerp(-0.47, -0.12, local),
        THREE.MathUtils.lerp(-0.05, -0.06, local) - Math.sin(local * Math.PI) * 0.08,
      );
    }
    return target;
  }

  updateSignal(time, state, reducedMotion) {
    const query = state.queryGlow ?? 0;
    const boundary = state.boundary ?? 0;
    const crystal = state.crystal ?? 0;
    const system = state.system ?? 0;
    const grid = state.grid ?? 0;
    const monogram = state.monogram ?? 0;
    const gridVisibility = this.mobile ? 1 - grid : 1 - grid * 0.82;
    const presence = (0.04 + query * 0.94 + boundary * 0.3 + crystal * 0.18
      + system * 0.1 + grid * 0.06 + monogram * 0.03) * gridVisibility;
    const phase = query > 0.1 ? this.queryProgress : (reducedMotion ? 0.72 : fract(time * 0.075 + 0.18));
    this.sampleEnginePath(this.temp, phase);
    this.signal.position.copy(this.temp);
    this.signalCore.material.opacity = presence * 0.82;
    this.signalHalo.material.opacity = presence * 0.24;
    const pulse = reducedMotion ? 1 : 1 + Math.sin(time * 1.6) * 0.08;
    this.signal.scale.setScalar(pulse * (0.84 + this.hoverAmount * 0.06));
    const positions = this.signalTrailGeometry.attributes.position.array;
    for (let index = 0; index < 18; index += 1) {
      const trail = index / 17;
      const trailPhase = THREE.MathUtils.clamp(phase - trail * 0.19, 0, 1);
      this.sampleEnginePath(this.temp, trailPhase);
      positions[index * 3] = this.temp.x;
      positions[index * 3 + 1] = this.temp.y;
      positions[index * 3 + 2] = this.temp.z;
    }
    this.signalTrailGeometry.attributes.position.needsUpdate = true;
    this.signalTrailMaterial.opacity = presence * 0.2;
  }

  updateTimelines(time, state, reducedMotion) {
    this.arrivalProgress = reducedMotion ? 1 : smoothStep(time / this.arrivalDuration);
    const query = state.queryGlow ?? 0;
    if (reducedMotion) {
      this.queryProgress = query > 0.05 ? 0.76 : 0;
      return;
    }
    if (query > 0.1 && !this.queryActive) {
      this.queryActive = true;
      this.queryStartTime = time;
    } else if (query < 0.045) {
      this.queryActive = false;
      this.queryProgress = 0;
    }
    if (this.queryActive) this.queryProgress = smoothStep((time - this.queryStartTime) / 2.3);
  }

  setMobile(mobile) {
    this.mobile = mobile;
    this.ambientPoints.material.size = mobile ? 0.034 : 0.044;
  }

  setProjectFocus(active) {
    this.hoverTarget = active ? 1 : 0;
  }

  update(time, pointer, state, reducedMotion = false) {
    const easing = reducedMotion ? 1 : 0.045;
    const reactionEase = reducedMotion ? 1 : 0.075;
    this.hoverAmount = THREE.MathUtils.lerp(this.hoverAmount, this.hoverTarget, reactionEase);
    this.updateTimelines(time, state, reducedMotion);
    this.root.position.x = THREE.MathUtils.lerp(this.root.position.x, state.x, easing);
    this.root.position.y = THREE.MathUtils.lerp(this.root.position.y, state.y, easing);
    this.root.position.z = THREE.MathUtils.lerp(this.root.position.z, state.z, easing);
    const requestedScale = state.scale * (1 + this.hoverAmount * 0.018);
    const scale = THREE.MathUtils.lerp(this.root.scale.x, requestedScale, reactionEase);
    this.root.scale.setScalar(scale);
    const motion = reducedMotion ? 0 : (state.motion ?? 1);
    const floatX = Math.sin(time * 0.19) * 0.035 * motion;
    const floatY = Math.cos(time * 0.16 + 0.7) * 0.042 * motion;
    const floatZ = Math.sin(time * 0.12 + 1.4) * 0.025 * motion;
    this.content.position.x = THREE.MathUtils.lerp(this.content.position.x, floatX, easing);
    this.content.position.y = THREE.MathUtils.lerp(this.content.position.y, floatY, easing);
    this.content.position.z = THREE.MathUtils.lerp(this.content.position.z, floatZ, easing);
    const pointerX = reducedMotion ? 0 : pointer.x * 0.075;
    const pointerY = reducedMotion ? 0 : pointer.y * 0.052;
    const idleTurn = Math.sin(time * 0.13 + 0.5) * 0.025 * motion;
    const idleTilt = Math.cos(time * 0.11) * 0.012 * motion;
    this.content.rotation.x = THREE.MathUtils.lerp(this.content.rotation.x, state.rx + pointerY + idleTilt, easing);
    this.content.rotation.y = THREE.MathUtils.lerp(this.content.rotation.y, state.ry + pointerX + idleTurn, easing);
    this.content.rotation.z = THREE.MathUtils.lerp(this.content.rotation.z, state.rz - pointer.x * 0.012, easing);
    this.updateCore(time, state, reducedMotion, pointer);
    this.updateSheath(time, state, reducedMotion);
    this.updateParticles(time, state, reducedMotion);
    this.updateSignal(time, state, reducedMotion);
    this.ambientPoints.material.opacity = 0.08
      * (this.mobile ? 1 - (state.grid ?? 0) : 1 - (state.grid ?? 0) * 0.85)
      * (1 - (state.monogram ?? 0) * 0.72);
    if (!reducedMotion) {
      this.ambientPoints.rotation.y = Math.sin(time * 0.07) * 0.025;
      this.ambientPoints.rotation.z = Math.cos(time * 0.05) * 0.018;
    }
  }
}
