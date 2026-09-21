import * as THREE from '../../vendor/three/three.module.min.js';

// Original procedural sculpture: a fictional machine that turns questions into decisions.
// It deliberately references no biological anatomy or external model data.

const CYAN = 0x67e1da;
const ICE = 0xe3f2ee;
const GRAPHITE = 0x182329;
const CERAMIC = 0xc8d2cf;
const VIOLET = 0x8f7ac2;
const AMBER = 0xf0aa52;
const TAU = Math.PI * 2;
const PARTICLE_COUNT = 6000;
const MOBILE_PARTICLE_COUNT = 2100;
const SEMANTIC_COLORS = [
  0x8db4b5, // input
  0xc7ded9, // evidence
  VIOLET, // memory
  CYAN, // inference
  0xb9ebe4, // decision
  AMBER, // output
];

function seededRandom(seed = 5102026) {
  let value = seed >>> 0;
  return () => {
    value = (value * 1664525 + 1013904223) >>> 0;
    return value / 4294967296;
  };
}

function fract(value) {
  return value - Math.floor(value);
}

function smoothStep(value) {
  const amount = THREE.MathUtils.clamp(value, 0, 1);
  return amount * amount * (3 - 2 * amount);
}

function cylinderBetween(start, end, radius, material, radialSegments = 10) {
  const direction = new THREE.Vector3().subVectors(end, start);
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, direction.length(), radialSegments),
    material,
  );
  mesh.position.copy(start).add(end).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
  return mesh;
}

function makeLine(points, material) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  return new THREE.Line(geometry, material);
}

function makeSegments(points, material) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  return new THREE.LineSegments(geometry, material);
}

function approach(current, target, speed, delta = 1 / 60) {
  return THREE.MathUtils.lerp(current, target, 1 - Math.exp(-speed * Math.min(delta, 0.08)));
}

export class SyntheticCognitionEngine {
  constructor({ mobile = false } = {}) {
    this.mobile = mobile;
    this.random = seededRandom();
    this.root = new THREE.Group();
    this.root.name = 'Synthetic Cognition Engine';
    this.content = new THREE.Group();
    this.root.add(this.content);

    this.pointer = new THREE.Vector2();
    this.temp = new THREE.Vector3();
    this.dummy = new THREE.Object3D();
    this.hoverTarget = 0;
    this.hoverAmount = 0;
    this.arrival = 0;
    this.lastTime = 0;
    this.activeScene = 0;

    this.hemisphereSegments = [];
    this.shellPlates = [];
    this.sensorNodes = [];
    this.rings = [];
    this.routes = [];
    this.prototypeModules = [];
    this.disposables = [];

    this.buildMaterials();
    this.buildHemispheres();
    this.buildReasoningCore();
    this.buildMechanicalBridge();
    this.buildShellPlates();
    this.buildSignalNetwork();
    this.buildParticleAtlas();
    this.buildPrototypeModules();
    this.buildBlueprint();
    this.buildAtmosphere();
    this.setMobile(mobile);
  }

  track(value) {
    this.disposables.push(value);
    return value;
  }

  buildMaterials() {
    this.graphiteMaterial = this.track(new THREE.MeshStandardMaterial({
      color: GRAPHITE,
      metalness: 0.88,
      roughness: 0.28,
      emissive: 0x071216,
      emissiveIntensity: 0.55,
    }));
    this.ceramicMaterial = this.track(new THREE.MeshPhysicalMaterial({
      color: CERAMIC,
      metalness: 0.12,
      roughness: 0.3,
      clearcoat: 0.76,
      clearcoatRoughness: 0.2,
      emissive: 0x101c1e,
      emissiveIntensity: 0.28,
    }));
    this.darkCeramicMaterial = this.track(this.ceramicMaterial.clone());
    this.darkCeramicMaterial.color.setHex(0x708386);
    this.darkCeramicMaterial.emissive.setHex(0x0a1a20);
    this.shellMaterial = this.track(new THREE.MeshPhysicalMaterial({
      color: 0x6f9299,
      metalness: 0.08,
      roughness: 0.12,
      transmission: 0.18,
      thickness: 0.34,
      clearcoat: 1,
      transparent: true,
      opacity: 0.19,
      depthWrite: false,
      side: THREE.DoubleSide,
    }));
    this.shellEdgeMaterial = this.track(new THREE.MeshBasicMaterial({
      color: CYAN,
      wireframe: true,
      transparent: true,
      opacity: 0.11,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }));
    this.cyanMaterial = this.track(new THREE.MeshStandardMaterial({
      color: 0x2d8687,
      emissive: CYAN,
      emissiveIntensity: 1.25,
      metalness: 0.34,
      roughness: 0.2,
    }));
    this.violetMaterial = this.track(new THREE.MeshStandardMaterial({
      color: 0x4d416f,
      emissive: VIOLET,
      emissiveIntensity: 0.72,
      metalness: 0.42,
      roughness: 0.28,
    }));
    this.amberMaterial = this.track(new THREE.MeshBasicMaterial({
      color: AMBER,
      transparent: true,
      opacity: 0.94,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }));
    this.coreMaterial = this.track(new THREE.MeshPhysicalMaterial({
      color: 0x6dc6c2,
      emissive: CYAN,
      emissiveIntensity: 1.8,
      roughness: 0.12,
      metalness: 0.2,
      clearcoat: 1,
      transparent: true,
      opacity: 0.72,
      depthWrite: false,
    }));
    this.coreHaloMaterial = this.track(new THREE.MeshBasicMaterial({
      color: CYAN,
      transparent: true,
      opacity: 0.09,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
    }));
    this.blueprintMaterial = this.track(new THREE.LineBasicMaterial({
      color: 0x75aeb5,
      transparent: true,
      opacity: 0,
      depthWrite: false,
    }));
    this.measureMaterial = this.track(new THREE.LineBasicMaterial({
      color: CYAN,
      transparent: true,
      opacity: 0,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }));
  }

  // Six unequal instruments orbit the reasoning core. Their different construction
  // keeps the silhouette from reading as two mirrored machine halves.
  buildHemispheres() {
    this.hemispheres = new THREE.Group();
    this.content.add(this.hemispheres);
    const layouts = [
      { type: 'lens', p: [-1.28, 0.5, -0.24], r: [0.18, -0.38, -0.24], s: 0.92, orbit: 0.2 },
      { type: 'frame', p: [-0.72, -0.74, 0.34], r: [0.62, 0.2, 0.34], s: 0.75, orbit: 1.25 },
      { type: 'plates', p: [0.18, 0.94, 0.42], r: [-0.34, 0.48, -0.18], s: 0.82, orbit: 2.4 },
      { type: 'node', p: [1.18, 0.42, -0.38], r: [0.2, -0.48, 0.2], s: 0.72, orbit: 3.6 },
      { type: 'membrane', p: [1.04, -0.66, 0.28], r: [-0.46, 0.38, -0.4], s: 1.02, orbit: 4.7 },
      { type: 'relay', p: [-0.12, -0.92, -0.42], r: [0.42, -0.24, 0.12], s: 0.64, orbit: 5.65 },
    ];
    const dodecahedron = this.track(new THREE.DodecahedronGeometry(0.38, 0));
    const octahedron = this.track(new THREE.OctahedronGeometry(0.34, 0));
    const openFrame = this.track(new THREE.TorusGeometry(0.34, 0.035, 7, 30));
    const lens = this.track(new THREE.SphereGeometry(0.38, 18, 12));
    const plate = this.track(new THREE.BoxGeometry(0.58, 0.08, 0.42));
    const membrane = this.track(new THREE.IcosahedronGeometry(0.42, 1));
    const relay = this.track(new THREE.CylinderGeometry(0.2, 0.29, 0.48, 8));
    const jointGeometry = this.track(new THREE.OctahedronGeometry(0.07, 0));

    layouts.forEach((spec, index) => {
      const segment = new THREE.Group();
      segment.position.set(...spec.p);
      segment.rotation.set(...spec.r);
      segment.scale.setScalar(spec.s);
      segment.userData.base = segment.position.clone();
      segment.userData.baseRotation = segment.rotation.clone();
      segment.userData.baseScale = spec.s;
      segment.userData.index = index;
      segment.userData.orbit = spec.orbit;
      segment.userData.radial = segment.position.clone().sub(new THREE.Vector3(-0.12, 0.06, 0.08)).normalize();

      if (spec.type === 'lens') {
        const glass = new THREE.Mesh(lens, this.shellMaterial);
        glass.scale.z = 0.58;
        const iris = new THREE.Mesh(this.track(new THREE.TorusGeometry(0.2, 0.055, 8, 26)), this.cyanMaterial);
        iris.rotation.x = Math.PI / 2;
        segment.add(glass, iris);
      } else if (spec.type === 'frame') {
        const frameA = new THREE.Mesh(openFrame, this.graphiteMaterial);
        const frameB = new THREE.Mesh(openFrame, this.violetMaterial);
        frameB.scale.setScalar(0.66);
        frameB.rotation.set(0.62, 0.38, 0.2);
        segment.add(frameA, frameB);
      } else if (spec.type === 'plates') {
        for (let layer = -2; layer <= 2; layer += 1) {
          const slab = new THREE.Mesh(plate, layer === 0 ? this.cyanMaterial : this.ceramicMaterial);
          slab.position.y = layer * 0.11;
          slab.position.x = Math.abs(layer) * -0.035;
          slab.rotation.z = layer * 0.035;
          slab.scale.set(1 - Math.abs(layer) * 0.08, 1, 1 - Math.abs(layer) * 0.06);
          segment.add(slab);
        }
      } else if (spec.type === 'node') {
        const shell = new THREE.Mesh(dodecahedron, this.darkCeramicMaterial);
        const nucleus = new THREE.Mesh(octahedron, this.amberMaterial);
        nucleus.scale.setScalar(0.48);
        segment.add(shell, nucleus);
      } else if (spec.type === 'membrane') {
        const skin = new THREE.Mesh(membrane, this.shellEdgeMaterial);
        skin.scale.set(1.16, 0.76, 0.58);
        const seed = new THREE.Mesh(octahedron, this.violetMaterial);
        seed.scale.set(0.42, 0.6, 0.42);
        segment.add(skin, seed);
      } else {
        const body = new THREE.Mesh(relay, this.graphiteMaterial);
        body.rotation.z = Math.PI / 2;
        const cap = new THREE.Mesh(octahedron, this.cyanMaterial);
        cap.position.x = 0.28;
        cap.scale.setScalar(0.42);
        segment.add(body, cap);
      }

      const joint = new THREE.Mesh(jointGeometry, index === 3 ? this.amberMaterial : this.cyanMaterial);
      joint.position.set(0.32, index % 2 ? -0.08 : 0.08, 0.16);
      segment.add(joint);
      this.hemispheres.add(segment);
      this.hemisphereSegments.push(segment);
      this.sensorNodes.push(joint);
    });
  }

  buildReasoningCore() {
    this.coreGroup = new THREE.Group();
    this.coreGroup.position.set(-0.12, 0.06, 0.08);
    this.content.add(this.coreGroup);
    const coreGeometry = this.track(new THREE.OctahedronGeometry(0.51, 2));
    this.core = new THREE.Mesh(coreGeometry, this.coreMaterial);
    this.core.scale.set(0.86, 1.08, 0.86);
    this.coreGroup.add(this.core);

    const inner = new THREE.Mesh(this.track(new THREE.IcosahedronGeometry(0.29, 1)), this.amberMaterial);
    inner.scale.set(0.68, 0.9, 0.68);
    this.coreGroup.add(inner);
    this.coreInner = inner;

    this.coreHalo = new THREE.Mesh(this.track(new THREE.SphereGeometry(0.72, 20, 14)), this.coreHaloMaterial);
    this.coreGroup.add(this.coreHalo);

    const ringSpecs = [
      [0.66, 0.04, 0.62, 0.24, -0.28, Math.PI * 1.42],
      [0.82, 0.028, 1.08, -0.42, 0.36, Math.PI * 1.18],
      [0.98, 0.024, -0.48, 0.82, 0.66, Math.PI * 1.34],
      [1.13, 0.02, 0.72, -0.66, 1.04, Math.PI * 1.06],
    ];
    ringSpecs.forEach((spec, index) => {
      const assembly = new THREE.Group();
      const arc = new THREE.Mesh(
        this.track(new THREE.TorusGeometry(spec[0], spec[1], 8, this.mobile ? 34 : 56, spec[5])),
        index === 2 ? this.violetMaterial : (index === 3 ? this.graphiteMaterial : this.cyanMaterial),
      );
      arc.rotation.z = -spec[5] * 0.5;
      assembly.add(arc);

      const bearingGeometry = this.track(new THREE.CylinderGeometry(spec[1] * 2.5, spec[1] * 2.5, 0.12, 8));
      [-1, 1].forEach((end) => {
        const angle = end * spec[5] * 0.5;
        const anchor = new THREE.Vector3(Math.cos(angle) * spec[0], Math.sin(angle) * spec[0], 0);
        const bearing = new THREE.Mesh(bearingGeometry, this.ceramicMaterial);
        bearing.position.copy(anchor);
        bearing.rotation.x = Math.PI / 2;
        assembly.add(bearing);
        assembly.add(cylinderBetween(anchor.clone().multiplyScalar(0.56), anchor, spec[1] * 0.65, this.graphiteMaterial, 6));
      });
      assembly.rotation.set(spec[2], spec[3], spec[4]);
      assembly.userData.baseRotation = assembly.rotation.clone();
      assembly.userData.speed = index === 0 ? 0.13 : 0;
      assembly.userData.phase = index * 1.7;
      this.coreGroup.add(assembly);
      this.rings.push(assembly);
    });
  }

  buildMechanicalBridge() {
    this.bridge = new THREE.Group();
    this.content.add(this.bridge);
    this.connectorMaterial = this.track(new THREE.LineBasicMaterial({
      color: CYAN, transparent: true, opacity: 0.42, depthWrite: false,
    }));
    const connectorSpecs = [
      [[-1.05, 0.39, -0.18], [-0.82, 0.18, 0.12], [-0.55, 0.18, 0.12]],
      [[-0.5, -0.54, 0.28], [-0.36, -0.28, 0.06], [-0.24, -0.2, 0.04]],
      [[0.28, 0.62, 0.34], [0.38, 0.4, 0.08], [0.56, 0.31, -0.08]],
      [[0.72, -0.4, 0.24], [0.56, -0.22, 0.04], [0.45, -0.12, 0.02]],
      [[0.62, 0.17, -0.08], [0.82, 0.28, -0.2], [0.96, 0.34, -0.28]],
    ];
    connectorSpecs.forEach((points, index) => {
      const controls = points.map((point) => new THREE.Vector3(...point));
      const curve = new THREE.CatmullRomCurve3(controls, false, 'centripetal', 0.5);
      const connector = makeLine(curve.getPoints(this.mobile ? 12 : 20), this.connectorMaterial);
      connector.userData.index = index;
      this.bridge.add(connector);
    });

    // One aperture receives the selected decision during the action chapter.
    this.outputAperture = new THREE.Group();
    this.outputAperture.position.set(1.72, -0.32, 0.22);
    this.outputAperture.rotation.set(0.26, 0.84, -0.28);
    const apertureRing = new THREE.Mesh(
      this.track(new THREE.TorusGeometry(0.2, 0.035, 8, 30)),
      this.amberMaterial,
    );
    const apertureIris = new THREE.Mesh(
      this.track(new THREE.CircleGeometry(0.125, 24)),
      this.track(new THREE.MeshBasicMaterial({
        color: AMBER, transparent: true, opacity: 0.2, depthWrite: false, side: THREE.DoubleSide,
      })),
    );
    this.apertureIrisMaterial = apertureIris.material;
    this.outputAperture.add(apertureRing, apertureIris);
    this.outputAperture.scale.setScalar(0.001);
    this.bridge.add(this.outputAperture);
  }

  buildShellPlates() {
    this.shellGroup = new THREE.Group();
    this.content.add(this.shellGroup);
    const shellGeometry = this.track(new THREE.SphereGeometry(1, 24, 14, 0.14, Math.PI * 0.72, 0.28, Math.PI * 0.52));
    const plateSpecs = [
      { p: [-1.05, 0.52, -0.2], r: [0.1, -1.34, -0.18], s: [0.7, 0.52, 0.38], peel: [-0.46, 0.34, -0.3] },
      { p: [0.22, 0.82, 0.35], r: [-0.42, 0.78, 0.38], s: [0.62, 0.45, 0.34], peel: [0.12, 0.62, 0.4] },
      { p: [0.96, -0.56, 0.22], r: [0.36, 1.08, -0.44], s: [0.76, 0.5, 0.42], peel: [0.54, -0.38, 0.26] },
    ];
    plateSpecs.forEach((spec, index) => {
      const plateGroup = new THREE.Group();
      const base = new THREE.Vector3(...spec.p);
      plateGroup.position.copy(base);
      plateGroup.rotation.set(...spec.r);
      plateGroup.userData.base = base.clone();
      plateGroup.userData.baseRotation = plateGroup.rotation.clone();
      plateGroup.userData.peel = new THREE.Vector3(...spec.peel);
      plateGroup.userData.index = index;

      const plateMesh = new THREE.Mesh(shellGeometry, this.shellMaterial);
      plateMesh.scale.set(...spec.s);
      const edge = new THREE.Mesh(shellGeometry, this.shellEdgeMaterial);
      edge.scale.copy(plateMesh.scale).multiplyScalar(1.015);
      plateGroup.add(plateMesh, edge);
      this.shellGroup.add(plateGroup);
      this.shellPlates.push(plateGroup);
    });
  }

  routeMaterial(color) {
    return this.track(new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: 0.34,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }));
  }

  // Product, evidence, and behavior enter independently, then converge at the core.
  buildSignalNetwork() {
    this.signalGroup = new THREE.Group();
    this.content.add(this.signalGroup);
    this.routeMaterials = [this.routeMaterial(CYAN), this.routeMaterial(VIOLET), this.routeMaterial(ICE)];
    const routes = [];
    const routeSpecs = [
      [[-1.28, 0.5, -0.24], [-0.92, 0.72, -0.4], [-0.48, 0.42, 0.12], [-0.12, 0.06, 0.08]],
      [[-0.72, -0.74, 0.34], [-0.56, -0.3, 0.5], [-0.28, -0.18, 0.18], [-0.12, 0.06, 0.08]],
      [[0.18, 0.94, 0.42], [-0.05, 0.68, 0.55], [0.1, 0.32, 0.22], [-0.12, 0.06, 0.08]],
      [[-0.12, 0.06, 0.08], [0.32, 0.3, -0.15], [0.72, 0.58, -0.28], [1.18, 0.42, -0.38]],
      [[-0.12, 0.06, 0.08], [0.2, -0.25, 0.35], [0.62, -0.42, 0.5], [1.04, -0.66, 0.28]],
      [[-0.12, 0.06, 0.08], [-0.42, -0.35, -0.2], [-0.28, -0.7, -0.45], [-0.12, -0.92, -0.42]],
    ];
    routeSpecs.forEach((spec, index) => {
      const controls = spec.map((point) => new THREE.Vector3(...point));
      const curve = new THREE.CatmullRomCurve3(controls, false, 'centripetal', 0.5);
      const samples = curve.getPoints(this.mobile ? 24 : 42);
      const line = makeLine(samples, this.routeMaterials[index % 3]);
      line.userData.category = index % 3;
      this.signalGroup.add(line);
      routes.push({ curve, line, category: index % 3, phase: this.random() });
    });
    this.routes = routes;

  }

  // One GPU point cloud carries six semantic populations through all eight chapters.
  // Target buffers are generated once, then reused without per-frame allocations.
  buildParticleAtlas() {
    const count = PARTICLE_COUNT;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    this.particleTargets = Array.from({ length: 8 }, () => new Float32Array(count * 3));
    this.particleSemantic = new Uint8Array(count);
    this.particleHypothesis = new Uint8Array(count);
    this.particlePhase = new Float32Array(count);
    this.particleSceneWeights = new Float32Array(8);
    this.particleColor = new THREE.Color();
    this.particleAmberColor = new THREE.Color(AMBER);
    this.lastParticleColorMix = -1;

    const assembledCenters = [
      [-1.28, 0.5, -0.24], [-0.72, -0.74, 0.34], [0.18, 0.94, 0.42],
      [-0.12, 0.06, 0.08], [1.04, -0.66, 0.28], [1.18, 0.42, -0.38],
    ];
    const resolvedCenters = [
      [-0.96, 0.66, 0.22], [-0.56, -0.5, -0.2], [0.02, 0.78, -0.34],
      [-0.18, 0.12, 0.18], [0.78, -0.34, 0.42], [1.02, 0.36, 0.08],
    ];
    const atlasCenters = [
      [-2.04, 0.92, -0.2], [-1.28, -1.0, 0.16], [-0.48, 1.18, -0.34],
      [0.36, 0.34, 0.35], [1.18, -0.88, -0.12], [2.05, 0.74, 0.14],
    ];
    const writeCluster = (buffer, offset, center, rx, ry, rz, u, v, w) => {
      const radius = Math.cbrt(u);
      const theta = TAU * v;
      const z = w * 2 - 1;
      const radial = Math.sqrt(Math.max(0, 1 - z * z));
      buffer[offset] = center[0] + Math.cos(theta) * radial * radius * rx;
      buffer[offset + 1] = center[1] + Math.sin(theta) * radial * radius * ry;
      buffer[offset + 2] = center[2] + z * radius * rz;
    };

    for (let index = 0; index < count; index += 1) {
      const offset = index * 3;
      const semantic = index % 6;
      const hypothesis = Math.floor(index / 6) % 3;
      const u = this.random();
      const v = this.random();
      const w = this.random();
      const phase = this.random() * TAU;
      this.particleSemantic[index] = semantic;
      this.particleHypothesis[index] = hypothesis;
      this.particlePhase[index] = phase;

      // 0: compact asymmetric orbital assembly.
      const assembled = assembledCenters[semantic];
      writeCluster(this.particleTargets[0], offset, assembled, 0.31, 0.25, 0.26, u, v, w);

      // 1: open, modules peel along their own radial vectors at varied depth.
      const openCenter = [
        (assembled[0] + 0.12) * 1.24 - 0.12,
        (assembled[1] - 0.06) * 1.18 + 0.06,
        assembled[2] * 1.36 + (semantic % 2 ? 0.08 : -0.06),
      ];
      writeCluster(this.particleTargets[1], offset, openCenter, 0.34, 0.27, 0.29, u, v, w);

      // 2: three pathways converge through the reasoning core before branching out.
      const path = semantic % 3;
      const progress = fract(u + semantic * 0.137);
      const incoming = semantic < 3;
      const pathY = (path - 1) * 0.5;
      const routeX = incoming ? -1.82 + progress * 1.7 : -0.12 + progress * 1.86;
      const convergence = Math.sin(progress * Math.PI);
      this.particleTargets[2][offset] = routeX;
      this.particleTargets[2][offset + 1] = 0.06 + pathY * (1 - convergence * 0.82) + (v - 0.5) * 0.1;
      this.particleTargets[2][offset + 2] = 0.08 + (w - 0.5) * 0.18 + Math.sin(progress * Math.PI * 2 + path) * 0.08;

      // 3: the orbit redistributes into two non-mirrored constellations.
      const bridgeParticle = index % 13 === 0;
      const splitCenters = [
        [-1.62, 0.54, -0.28], [-1.22, -0.62, 0.44], [-0.58, 0.92, 0.22],
        [0.32, 0.12, -0.12], [1.18, -0.72, 0.48], [1.54, 0.46, -0.36],
      ];
      const splitCenter = bridgeParticle
        ? [-0.68 + u * 1.56, -0.28 + v * 0.48, -0.08 + (w - 0.5) * 0.28]
        : splitCenters[semantic];
      writeCluster(this.particleTargets[3], offset, splitCenter, bridgeParticle ? 0.1 : 0.28, 0.22, 0.24, u, v, w);

      // 4: three hypotheses fan out; the center hypothesis is denser and locks to output.
      if ((semantic === 4 || semantic === 5) && hypothesis === 1) {
        const lockProgress = fract(u * 0.72 + 0.18);
        this.particleTargets[4][offset] = lockProgress * 1.72;
        this.particleTargets[4][offset + 1] = 1.02 - lockProgress * 0.84 + (v - 0.5) * 0.07;
        this.particleTargets[4][offset + 2] = 0.62 + (w - 0.5) * 0.12;
      } else {
        const prototypeCenter = [
          (hypothesis - 1) * 1.04 - 0.08,
          0.9 - Math.abs(hypothesis - 1) * 0.2 + hypothesis * 0.05,
          0.62 + (hypothesis - 1) * 0.4,
        ];
        const winnerScale = hypothesis === 1 ? 0.2 : 0.35;
        writeCluster(this.particleTargets[4], offset, prototypeCenter, winnerScale, winnerScale * 0.8, winnerScale, u, v, w);
      }

      // 5: one selected decision leaves through a deliberate curved output aperture.
      if (semantic >= 4 || (semantic === 3 && hypothesis === 1)) {
        const actionProgress = fract(u + semantic * 0.19);
        const arc = Math.sin(actionProgress * Math.PI);
        this.particleTargets[5][offset] = -0.04 + actionProgress * 1.76;
        this.particleTargets[5][offset + 1] = 0.08 - actionProgress * 0.4 + arc * 0.48 + (v - 0.5) * 0.09;
        this.particleTargets[5][offset + 2] = 0.08 + actionProgress * 0.14 + arc * 0.28 + (w - 0.5) * 0.1;
      } else {
        writeCluster(this.particleTargets[5], offset, assembled, 0.23, 0.2, 0.21, u, v, w);
      }

      // 6: measured atlas, with six populations separated into a readable constellation.
      writeCluster(this.particleTargets[6], offset, atlasCenters[semantic], 0.3, 0.24, 0.2, u, v, w);

      // 7: resolution is a newly learned, tighter orbital topology rather than a reset.
      const resolved = resolvedCenters[semantic];
      writeCluster(this.particleTargets[7], offset, resolved, 0.21, 0.18, 0.18, u, v, w);

      positions[offset] = this.particleTargets[0][offset];
      positions[offset + 1] = this.particleTargets[0][offset + 1];
      positions[offset + 2] = this.particleTargets[0][offset + 2];
      this.particleColor.setHex(SEMANTIC_COLORS[semantic]);
      colors[offset] = this.particleColor.r;
      colors[offset + 1] = this.particleColor.g;
      colors[offset + 2] = this.particleColor.b;
    }

    const geometry = this.track(new THREE.BufferGeometry());
    const positionAttribute = new THREE.BufferAttribute(positions, 3);
    positionAttribute.setUsage(THREE.DynamicDrawUsage);
    const colorAttribute = new THREE.BufferAttribute(colors, 3);
    colorAttribute.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('position', positionAttribute);
    geometry.setAttribute('color', colorAttribute);
    this.particleMaterial = this.track(new THREE.PointsMaterial({
      size: this.mobile ? 0.034 : 0.028,
      vertexColors: true,
      transparent: true,
      opacity: 0.84,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    }));
    this.particleAtlas = new THREE.Points(geometry, this.particleMaterial);
    this.particleAtlas.frustumCulled = false;
    this.particleAtlas.renderOrder = 2;
    this.content.add(this.particleAtlas);
  }

  // Temporary parallel modules appear only during the experimentation chapter.
  buildPrototypeModules() {
    this.prototypes = new THREE.Group();
    this.content.add(this.prototypes);
    const housingGeometry = this.track(new THREE.BoxGeometry(0.52, 0.3, 0.38));
    const loopGeometry = this.track(new THREE.TorusGeometry(0.22, 0.028, 7, 24));
    const prototypeLayout = [
      [-0.92, 0.08, 0.18],
      [-0.04, 0.3, 0.72],
      [0.88, -0.02, -0.18],
    ];
    for (let index = 0; index < 3; index += 1) {
      const module = new THREE.Group();
      module.position.set(...prototypeLayout[index]);
      module.userData.base = module.position.clone();
      module.userData.index = index;
      const housing = new THREE.Mesh(housingGeometry, index === 1 ? this.ceramicMaterial : this.graphiteMaterial);
      const loop = new THREE.Mesh(loopGeometry, index === 1 ? this.amberMaterial : this.violetMaterial);
      loop.rotation.x = Math.PI / 2;
      module.add(housing, loop);
      this.prototypes.add(module);
      this.prototypeModules.push(module);
    }
    this.prototypes.scale.setScalar(0.001);
  }

  buildAtlasLabels() {
    const glyphs = {
      A: ['01110', '10001', '10001', '11111', '10001', '10001', '10001'],
      C: ['01111', '10000', '10000', '10000', '10000', '10000', '01111'],
      D: ['11110', '10001', '10001', '10001', '10001', '10001', '11110'],
      E: ['11111', '10000', '10000', '11110', '10000', '10000', '11111'],
      F: ['11111', '10000', '10000', '11110', '10000', '10000', '10000'],
      I: ['11111', '00100', '00100', '00100', '00100', '00100', '11111'],
      M: ['10001', '11011', '10101', '10101', '10001', '10001', '10001'],
      N: ['10001', '11001', '10101', '10011', '10001', '10001', '10001'],
      O: ['01110', '10001', '10001', '10001', '10001', '10001', '01110'],
      P: ['11110', '10001', '10001', '11110', '10000', '10000', '10000'],
      Q: ['01110', '10001', '10001', '10001', '10101', '10010', '01101'],
      R: ['11110', '10001', '10001', '11110', '10100', '10010', '10001'],
      S: ['01111', '10000', '10000', '01110', '00001', '00001', '11110'],
      T: ['11111', '00100', '00100', '00100', '00100', '00100', '00100'],
      U: ['10001', '10001', '10001', '10001', '10001', '10001', '01110'],
      V: ['10001', '10001', '10001', '10001', '10001', '01010', '00100'],
      Y: ['10001', '10001', '01010', '00100', '00100', '00100', '00100'],
    };
    const labels = [
      ['QUESTION', -2.42, 1.28],
      ['EVIDENCE', -1.69, -1.35],
      ['MEMORY', -0.78, 1.55],
      ['INFERENCE', 0.03, 0.75],
      ['DECISION', 0.87, -1.23],
      ['OUTPUT', 1.77, 1.15],
    ];
    const points = [];
    const size = 0.022;
    labels.forEach(([word, startX, startY]) => {
      [...word].forEach((letter, letterIndex) => {
        const glyph = glyphs[letter];
        if (!glyph) return;
        glyph.forEach((row, y) => {
          [...row].forEach((pixel, x) => {
            if (pixel !== '1') return;
            const px = startX + (letterIndex * 6 + x) * size;
            const py = startY - y * size;
            points.push(
              new THREE.Vector3(px, py, -0.12),
              new THREE.Vector3(px + size * 0.72, py, -0.12),
            );
          });
        });
      });
    });
    this.atlasLabelMaterial = this.track(new THREE.LineBasicMaterial({
      color: ICE,
      transparent: true,
      opacity: 0,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }));
    this.atlasLabels = makeSegments(points, this.atlasLabelMaterial);
    this.blueprint.add(this.atlasLabels);
  }

  buildBlueprint() {
    this.blueprint = new THREE.Group();
    this.content.add(this.blueprint);
    const grid = [];
    for (let index = -5; index <= 5; index += 1) {
      grid.push(
        new THREE.Vector3(index * 0.42, -2.35, -0.7), new THREE.Vector3(index * 0.42, 1.35, -0.7),
        new THREE.Vector3(-2.2, index * 0.34 - 0.5, -0.7), new THREE.Vector3(2.2, index * 0.34 - 0.5, -0.7),
      );
    }
    this.gridLines = makeSegments(grid, this.blueprintMaterial);
    this.blueprint.add(this.gridLines);

    const measures = [];
    const anchors = [
      [-1.95, 0.7, -0.2, -2.45, 1.0, -0.2],
      [1.94, 0.55, -0.2, 2.42, 0.86, -0.2],
      [-0.22, -1.38, -0.2, -1.05, -2.05, -0.2],
      [0.25, 0.18, 0.55, 1.32, 1.08, 0.55],
    ];
    anchors.forEach((a) => {
      const from = new THREE.Vector3(a[0], a[1], a[2]);
      const to = new THREE.Vector3(a[3], a[4], a[5]);
      measures.push(from, to);
      measures.push(to.clone().add(new THREE.Vector3(-0.08, 0.04, 0)), to.clone().add(new THREE.Vector3(0.26, 0.04, 0)));
      measures.push(to.clone().add(new THREE.Vector3(-0.08, -0.04, 0)), to.clone().add(new THREE.Vector3(0.16, -0.04, 0)));
    });
    this.measureLines = makeSegments(measures, this.measureMaterial);
    this.blueprint.add(this.measureLines);
    this.buildAtlasLabels();

    const lzPoints = [
      new THREE.Vector3(-0.16, 0.17, 0.52), new THREE.Vector3(-0.16, -0.14, 0.52),
      new THREE.Vector3(-0.16, -0.14, 0.52), new THREE.Vector3(0, -0.14, 0.52),
      new THREE.Vector3(0.05, 0.16, 0.52), new THREE.Vector3(0.22, 0.16, 0.52),
      new THREE.Vector3(0.22, 0.16, 0.52), new THREE.Vector3(0.05, -0.14, 0.52),
      new THREE.Vector3(0.05, -0.14, 0.52), new THREE.Vector3(0.23, -0.14, 0.52),
    ];
    this.monogramMaterial = this.track(new THREE.LineBasicMaterial({
      color: AMBER, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending,
    }));
    this.monogram = makeSegments(lzPoints, this.monogramMaterial);
    this.content.add(this.monogram);
  }

  buildAtmosphere() {
    const count = 120;
    const positions = new Float32Array(count * 3);
    for (let index = 0; index < count; index += 1) {
      const radius = 1.2 + this.random() * 1.7;
      const angle = this.random() * TAU;
      positions[index * 3] = Math.cos(angle) * radius;
      positions[index * 3 + 1] = (this.random() - 0.55) * 4.1;
      positions[index * 3 + 2] = (this.random() - 0.5) * 1.7;
    }
    const geometry = this.track(new THREE.BufferGeometry());
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.atmosphereMaterial = this.track(new THREE.PointsMaterial({
      color: CYAN,
      size: this.mobile ? 0.025 : 0.034,
      transparent: true,
      opacity: 0.1,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    }));
    this.atmosphere = new THREE.Points(geometry, this.atmosphereMaterial);
    this.content.add(this.atmosphere);
  }

  sceneWeights(state) {
    return {
      open: state.open ?? state.questionFocus ?? state.question ?? 0,
      pathways: state.operating ?? state.queryGlow ?? state.query ?? 0,
      split: state.structure ?? state.boundaryReveal ?? state.boundary ?? 0,
      prototypes: state.prototype ?? state.crystalReveal ?? state.crystal ?? 0,
      action: state.action ?? state.system ?? 0,
      blueprint: state.blueprint ?? state.grid ?? 0,
      resolved: state.resolution ?? state.monogram ?? 0,
    };
  }

  determineScene(state, weights) {
    const scores = [state.assembled ?? state.cloud ?? 0.35, weights.open, weights.pathways, weights.split,
      weights.prototypes, weights.action, weights.blueprint, weights.resolved];
    let bestIndex = 0;
    scores.forEach((score, index) => {
      if (score > scores[bestIndex]) bestIndex = index;
    });
    return bestIndex;
  }

  updateStructure(time, delta, weights, reducedMotion) {
    const ease = reducedMotion ? 1000 : 5.5;
    const motion = reducedMotion ? 0 : 1;
    this.hemisphereSegments.forEach((segment) => {
      const {
        base, baseRotation, baseScale, index, radial,
      } = segment.userData;
      const splitOffset = [
        [-0.38, 0.12, -0.12], [-0.16, -0.34, 0.28], [0.22, 0.3, 0.34],
        [0.42, 0.08, -0.28], [0.24, -0.3, 0.2], [-0.28, -0.15, -0.32],
      ][index];
      const resolvedX = [0.38, 0.27, -0.13, -0.48, -0.49, 0.17][index];
      const resolvedY = [0.02, 0.89, -0.22, -0.07, 0.46, 0.37][index];
      const resolvedZ = [0.36, -0.42, -0.48, 0.38, 0.18, 0.5][index];
      const prototypeDepth = [-0.28, 0.46, 0.18, -0.42, 0.56, -0.12][index];
      const targetX = base.x
        + radial.x * weights.open * (0.28 + index * 0.035)
        - radial.x * weights.pathways * 0.14
        + splitOffset[0] * weights.split
        + (index - 2.5) * weights.blueprint * 0.14
        + resolvedX * weights.resolved;
      const targetY = base.y
        + radial.y * weights.open * (0.24 + index * 0.025)
        - radial.y * weights.pathways * 0.12
        + splitOffset[1] * weights.split
        + (index - 2.5) * weights.blueprint * 0.13
        + resolvedY * weights.resolved;
      const targetZ = base.z
        + radial.z * weights.open * 0.4
        + splitOffset[2] * weights.split
        + prototypeDepth * weights.prototypes
        + (index % 2 ? 0.48 : -0.38) * weights.blueprint
        + resolvedZ * weights.resolved;
      segment.position.x = approach(segment.position.x, targetX, ease, delta);
      segment.position.y = approach(segment.position.y, targetY, ease, delta);
      segment.position.z = approach(segment.position.z, targetZ, ease, delta);
      segment.rotation.x = approach(segment.rotation.x, baseRotation.x
        + radial.y * weights.open * 0.28 + weights.blueprint * (index - 2.5) * 0.1, ease, delta);
      segment.rotation.y = approach(segment.rotation.y, baseRotation.y
        + splitOffset[0] * weights.split * 0.58 + weights.resolved * (index - 2.5) * 0.08, ease, delta);
      segment.rotation.z = approach(segment.rotation.z, baseRotation.z
        + (index % 2 ? 1 : -1) * weights.prototypes * 0.22, ease, delta);
      const resolvedScale = index === 3 ? 0.28 : -0.12;
      const targetScale = baseScale * (1 - weights.pathways * 0.06 + weights.resolved * resolvedScale);
      const moduleScale = approach(segment.scale.x, targetScale, ease, delta);
      segment.scale.setScalar(moduleScale);
      const recalibration = Math.sin(time * 0.55 + index * 0.8 + segment.userData.orbit) * 0.012 * motion;
      segment.position.y += recalibration;
    });

    this.shellPlates.forEach((plate, index) => {
      const {
        base, baseRotation, peel,
      } = plate.userData;
      const targetX = base.x + peel.x * weights.open + peel.x * weights.blueprint * 0.8;
      const targetY = base.y + peel.y * weights.open + peel.y * weights.blueprint * 0.8;
      const targetZ = base.z + peel.z * weights.open + (index - 1) * weights.blueprint * 0.42;
      plate.position.x = approach(plate.position.x, targetX, ease, delta);
      plate.position.y = approach(plate.position.y, targetY, ease, delta);
      plate.position.z = approach(plate.position.z, targetZ, ease, delta);
      plate.rotation.x = approach(plate.rotation.x, baseRotation.x + peel.y * weights.open * 0.36, ease, delta);
      plate.rotation.y = approach(plate.rotation.y, baseRotation.y + peel.x * weights.open * 0.3, ease, delta);
      plate.rotation.z = approach(plate.rotation.z, baseRotation.z + (index - 1) * weights.blueprint * 0.2, ease, delta);
    });

    const bridgeScale = 1 - weights.open * 0.08 + weights.pathways * 0.08;
    const nextBridgeScale = approach(this.bridge.scale.x, bridgeScale, ease, delta);
    this.bridge.scale.setScalar(nextBridgeScale);
    this.bridge.position.z = approach(this.bridge.position.z, weights.split * 0.12, ease, delta);
    this.connectorMaterial.opacity = (0.24 + weights.pathways * 0.46 + weights.split * 0.14)
      * (1 - weights.blueprint * 0.72);
    const apertureScaleTarget = Math.max(0.001, weights.action * 1.18);
    const apertureScale = approach(this.outputAperture.scale.x, apertureScaleTarget, ease, delta);
    this.outputAperture.scale.setScalar(apertureScale);
    this.outputAperture.rotation.z = -0.28 + weights.action * 0.22;
    this.apertureIrisMaterial.opacity = 0.08 + weights.action * 0.48 + weights.resolved * 0.16;

    const prototypeScale = Math.max(0.001, weights.prototypes + weights.blueprint * 0.65);
    this.prototypes.scale.setScalar(approach(this.prototypes.scale.x, prototypeScale, ease, delta));
    this.prototypeModules.forEach((module, index) => {
      const base = module.userData.base;
      module.position.x = approach(module.position.x, base.x + weights.prototypes * (index - 1) * 0.16, ease, delta);
      module.position.y = approach(module.position.y, base.y + weights.prototypes * (0.62 - Math.abs(index - 1) * 0.14), ease, delta);
      module.position.z = approach(module.position.z, base.z + weights.prototypes * (index === 1 ? 0.34 : -0.06), ease, delta);
      module.rotation.y = time * (index === 1 ? 0.2 : -0.11) * motion;
    });
  }

  updateSignals(time, weights, reducedMotion, delta) {
    const categoryBoost = weights.pathways * 0.56;
    const blueprintQuiet = 1 - weights.blueprint * 0.74;
    this.routeMaterials.forEach((material, index) => {
      const prototypeWinner = weights.prototypes * (index === 1 ? 0.58 : -0.12);
      material.opacity = THREE.MathUtils.clamp(
        (0.22 + categoryBoost + prototypeWinner + weights.action * 0.2) * blueprintQuiet,
        0.05,
        0.92,
      );
      material.color.setHex(index === 0 ? CYAN : (index === 1 ? VIOLET : ICE));
      if (weights.prototypes > 0.55 && index === 1) material.color.lerp(this.particleAmberColor, weights.prototypes * 0.72);
    });

    this.amberMaterial.opacity = (0.68 + weights.pathways * 0.14 + weights.prototypes * 0.12 + weights.action * 0.16)
      * blueprintQuiet;
    this.updateParticleAtlas(time, weights, reducedMotion, delta);
  }

  updateParticleAtlas(time, weights, reducedMotion, delta) {
    const sceneWeights = this.particleSceneWeights;
    sceneWeights[1] = Math.max(0, weights.open);
    sceneWeights[2] = Math.max(0, weights.pathways);
    sceneWeights[3] = Math.max(0, weights.split);
    sceneWeights[4] = Math.max(0, weights.prototypes);
    sceneWeights[5] = Math.max(0, weights.action);
    sceneWeights[6] = Math.max(0, weights.blueprint);
    sceneWeights[7] = Math.max(0, weights.resolved);
    let activeWeight = 0;
    for (let scene = 1; scene < 8; scene += 1) activeWeight += sceneWeights[scene];
    sceneWeights[0] = Math.max(0, 1 - Math.min(1, activeWeight));
    let totalWeight = sceneWeights[0];
    for (let scene = 1; scene < 8; scene += 1) totalWeight += sceneWeights[scene];
    const inverseWeight = totalWeight > 0 ? 1 / totalWeight : 1;
    const positionAttribute = this.particleAtlas.geometry.getAttribute('position');
    const positions = positionAttribute.array;
    const count = this.mobile ? MOBILE_PARTICLE_COUNT : PARTICLE_COUNT;
    const transitionEase = reducedMotion ? 1 : 1 - Math.exp(-Math.min(delta, 0.05) * 5.2);
    const idleStrength = reducedMotion ? 0 : (1 - Math.min(1, activeWeight * 0.34));
    const pointerX = this.pointer.x * 2.05;
    const pointerY = this.pointer.y * 1.55;

    for (let index = 0; index < count; index += 1) {
      const offset = index * 3;
      let tx = 0;
      let ty = 0;
      let tz = 0;
      for (let scene = 0; scene < 8; scene += 1) {
        const influence = sceneWeights[scene] * inverseWeight;
        if (influence <= 0) continue;
        const target = this.particleTargets[scene];
        tx += target[offset] * influence;
        ty += target[offset + 1] * influence;
        tz += target[offset + 2] * influence;
      }

      // Only a sparse subset circulates; the main population settles into a precise silhouette.
      if (index % 29 === 0 && !reducedMotion) {
        const phase = this.particlePhase[index];
        const flow = (0.015 + weights.pathways * 0.025 + weights.action * 0.018) * idleStrength;
        tx += Math.cos(time * 0.74 + phase) * flow;
        ty += Math.sin(time * 0.62 + phase) * flow;
        tz += Math.sin(time * 0.51 + phase) * flow * 0.7;
      }

      // Pointer attraction is local to nearby particles and never rotates the full sculpture.
      if (!reducedMotion) {
        const pointerDx = pointerX - tx;
        const pointerDy = pointerY - ty;
        const pointerDistance = pointerDx * pointerDx + pointerDy * pointerDy;
        if (pointerDistance < 0.62) {
          const attraction = (1 - pointerDistance / 0.62) * 0.055;
          tx += pointerDx * attraction;
          ty += pointerDy * attraction;
          tz += 0.025 * attraction;
        }
      }

      const dx = tx - positions[offset];
      const dy = ty - positions[offset + 1];
      const dz = tz - positions[offset + 2];
      if (reducedMotion) {
        positions[offset] = tx;
        positions[offset + 1] = ty;
        positions[offset + 2] = tz;
      } else {
        // A small phase-stable perpendicular term turns straight interpolation into curved travel.
        const distance = Math.min(1, Math.sqrt(dx * dx + dy * dy + dz * dz));
        const curve = Math.sin(this.particlePhase[index]) * distance * 0.035 * transitionEase;
        positions[offset] += dx * transitionEase - dy * curve;
        positions[offset + 1] += dy * transitionEase + dx * curve;
        positions[offset + 2] += dz * transitionEase + Math.cos(this.particlePhase[index]) * curve * 0.42;
      }
    }
    positionAttribute.needsUpdate = true;

    // Prototype comparison dims losing hypotheses while the selected route turns amber.
    const prototypeMix = THREE.MathUtils.clamp(weights.prototypes, 0, 1);
    if (Math.abs(prototypeMix - this.lastParticleColorMix) > 0.01) {
      const colorAttribute = this.particleAtlas.geometry.getAttribute('color');
      const colors = colorAttribute.array;
      for (let index = 0; index < PARTICLE_COUNT; index += 1) {
        const offset = index * 3;
        const semantic = this.particleSemantic[index];
        const hypothesis = this.particleHypothesis[index];
        this.particleColor.setHex(SEMANTIC_COLORS[semantic]);
        if (prototypeMix > 0) {
          if (hypothesis === 1) {
            this.particleColor.lerp(this.particleAmberColor, prototypeMix * 0.88);
          } else {
            this.particleColor.multiplyScalar(1 - prototypeMix * 0.62);
          }
        }
        colors[offset] = this.particleColor.r;
        colors[offset + 1] = this.particleColor.g;
        colors[offset + 2] = this.particleColor.b;
      }
      colorAttribute.needsUpdate = true;
      this.lastParticleColorMix = prototypeMix;
    }
    this.particleMaterial.opacity = (0.72 + weights.pathways * 0.15 + weights.prototypes * 0.14 + weights.action * 0.1)
      * (1 - weights.blueprint * 0.12);
    this.particleMaterial.size = (this.mobile ? 0.034 : 0.028) + weights.prototypes * (this.mobile ? 0.005 : 0.007);
  }

  updateCore(time, delta, weights, reducedMotion) {
    const motion = reducedMotion ? 0 : 1;
    const breath = 1 + Math.sin(time * 0.82) * 0.035 * motion;
    this.core.scale.set(0.86, 1.08, 0.86).multiplyScalar(breath);
    this.core.rotation.y = time * 0.12 * motion;
    this.core.rotation.x = Math.sin(time * 0.19) * 0.08 * motion;
    this.coreInner.rotation.y = -time * 0.28 * motion;
    this.coreInner.rotation.z = time * 0.17 * motion;
    this.coreHalo.scale.setScalar(1 + Math.sin(time * 0.63) * 0.08 * motion);
    this.coreHaloMaterial.opacity = 0.065 + weights.pathways * 0.07 + weights.action * 0.1 + this.hoverAmount * 0.035;
    this.coreMaterial.emissiveIntensity = (1.36 + weights.pathways * 0.75 + weights.action * 0.55 + this.hoverAmount * 0.35)
      * (1 - weights.resolved * 0.28);
    this.rings.forEach((ring, index) => {
      const base = ring.userData.baseRotation;
      const explode = weights.open * index * 0.035 + weights.blueprint * index * 0.16;
      ring.position.z = approach(ring.position.z, explode, reducedMotion ? 1000 : 5.5, delta);
      if (index === 0) {
        ring.rotation.x = base.x + Math.sin(time * 0.12) * 0.025 * motion;
        ring.rotation.y = base.y + Math.cos(time * 0.1) * 0.018 * motion;
        ring.rotation.z = base.z + time * ring.userData.speed * motion + weights.action * 0.16;
      } else {
        const recalibration = Math.sin(time * 0.22 + ring.userData.phase) * (0.026 + index * 0.006) * motion;
        ring.rotation.x = base.x + recalibration;
        ring.rotation.y = base.y - recalibration * 0.72;
        ring.rotation.z = base.z + recalibration * 0.46 + weights.action * index * 0.12;
      }
      const ringScale = 1 + weights.open * index * 0.055 + weights.blueprint * index * 0.11;
      ring.scale.setScalar(ringScale);
    });
  }

  updateAppearance(weights) {
    const blueprint = weights.blueprint;
    const resolved = weights.resolved;
    this.shellMaterial.opacity = (0.19 + weights.open * 0.05) * (1 - blueprint * 0.62);
    this.shellEdgeMaterial.opacity = (0.1 + weights.open * 0.08) * (1 - blueprint * 0.5);
    this.ceramicMaterial.opacity = 1 - blueprint * 0.92;
    this.darkCeramicMaterial.opacity = 1 - blueprint * 0.93;
    this.graphiteMaterial.opacity = 1 - blueprint * 0.95;
    this.cyanMaterial.opacity = 1 - blueprint * 0.82;
    this.violetMaterial.opacity = 1 - blueprint * 0.82;
    [this.ceramicMaterial, this.darkCeramicMaterial, this.graphiteMaterial, this.cyanMaterial, this.violetMaterial].forEach((material) => {
      material.transparent = blueprint > 0.01;
    });
    this.coreMaterial.opacity = 0.72 * (1 - blueprint * 0.84) * (1 - resolved * 0.34);
    this.coreHaloMaterial.opacity *= 1 - blueprint * 0.9;
    this.blueprintMaterial.opacity = blueprint * 0.14;
    this.measureMaterial.opacity = blueprint * 0.62;
    this.atlasLabelMaterial.opacity = blueprint * 0.72;
    this.monogramMaterial.opacity = resolved * 0.96;
    this.atmosphereMaterial.opacity = 0.085 * (1 - blueprint * 0.75) * (1 - resolved * 0.4);
  }

  setMobile(mobile) {
    this.mobile = mobile;
    if (!this.particleAtlas) return;
    this.particleAtlas.geometry.setDrawRange(0, mobile ? MOBILE_PARTICLE_COUNT : PARTICLE_COUNT);
    this.particleMaterial.size = mobile ? 0.034 : 0.028;
    this.routes.forEach((route, index) => {
      route.line.visible = !mobile || index < 7;
    });
    this.atmosphere.geometry.setDrawRange(0, mobile ? 70 : 120);
    this.atmosphereMaterial.size = mobile ? 0.025 : 0.034;
    this.shellPlates.forEach((plate, index) => {
      plate.children[1].visible = !mobile || index < 2;
    });
    this.atlasLabels.visible = !mobile;
  }

  setProjectFocus(active) {
    this.hoverTarget = active ? 1 : 0;
  }

  update(time, pointer, state, reducedMotion = false) {
    const delta = Math.min(Math.max(time - this.lastTime, 0), 0.05) || 1 / 60;
    this.lastTime = time;
    this.pointer.copy(pointer || new THREE.Vector2());
    const weights = this.sceneWeights(state || {});
    this.activeScene = this.determineScene(state || {}, weights);
    this.arrival = reducedMotion ? 1 : smoothStep(Math.min(1, time / 1.35));
    this.hoverAmount = approach(this.hoverAmount, this.hoverTarget, reducedMotion ? 1000 : 5.5, delta);

    const rootEase = reducedMotion ? 1000 : 4.6;
    this.root.position.x = approach(this.root.position.x, state.x ?? 0, rootEase, delta);
    this.root.position.y = approach(this.root.position.y, state.y ?? 0, rootEase, delta);
    this.root.position.z = approach(this.root.position.z, state.z ?? 0, rootEase, delta);
    const targetScale = (state.scale ?? 1) * this.arrival * (1 + this.hoverAmount * 0.025);
    const scale = approach(this.root.scale.x, targetScale, reducedMotion ? 1000 : 5.5, delta);
    this.root.scale.setScalar(scale);

    const motion = reducedMotion ? 0 : (state.motion ?? 1);
    this.content.position.x = Math.sin(time * 0.16) * 0.025 * motion;
    this.content.position.y = Math.cos(time * 0.13 + 0.8) * 0.035 * motion;
    this.content.position.z = Math.sin(time * 0.11) * 0.018 * motion;
    this.content.rotation.x = (state.rx ?? 0) + Math.cos(time * 0.1) * 0.012 * motion;
    this.content.rotation.y = (state.ry ?? 0) + Math.sin(time * 0.085) * 0.025 * motion;
    this.content.rotation.z = state.rz ?? 0;

    this.updateStructure(time, delta, weights, reducedMotion);
    this.updateSignals(time, weights, reducedMotion, delta);
    this.updateCore(time, delta, weights, reducedMotion);
    this.updateAppearance(weights);
    this.atmosphere.rotation.y = Math.sin(time * 0.05) * 0.05 * motion;
    this.atmosphere.rotation.z = Math.cos(time * 0.04) * 0.025 * motion;
  }

  dispose() {
    this.root.removeFromParent();
    this.disposables.forEach((resource) => resource.dispose?.());
  }
}
