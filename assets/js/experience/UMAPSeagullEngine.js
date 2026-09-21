import * as THREE from '../../vendor/three/three.module.min.js';

const POINT_COUNT = 6000;
const MOBILE_POINT_COUNT = 2100;
const TARGET_COUNT = 8;
const TAU = Math.PI * 2;
const LABEL_INDICES = [0, 1, 2, 4, 5, 7, 10];
const MOBILE_LABEL_COUNT = 3;
const SIGNATURE_SEGMENTS = [
  { start: [-1.48, 1.34], end: [-1.48, -1.32], length: 2.66, letter: 0 },
  { start: [-1.48, -1.32], end: [-0.28, -1.32], length: 1.2, letter: 0 },
  { start: [0.08, 1.34], end: [1.58, 1.34], length: 1.5, letter: 1 },
  { start: [1.58, 1.34], end: [0.08, -1.32], length: 3.05, letter: 1 },
  { start: [0.08, -1.32], end: [1.58, -1.32], length: 1.5, letter: 1 },
];
const SIGNATURE_LENGTH = SIGNATURE_SEGMENTS.reduce((total, segment) => total + segment.length, 0);
const BRANCH_CENTERS = [
  [-0.18, 0.02, 0.02],
  [-0.52, 0.24, 0.08],
  [-0.8, 0.56, 0.16],
  [-1.55, 1.2, -0.05],
  [0.16, 0.3, -0.1],
  [0.46, 0.64, 0.13],
  [1.42, 1.42, 0.38],
  [0.18, -0.28, 0.15],
  [0.5, -0.6, -0.1],
  [1.34, -1.38, -0.34],
  [-0.5, -0.28, -0.18],
  [-1.3, -1.18, -0.4],
  [0.7, 0.14, 0.45],
  [1.56, 0.42, 0.75],
];

function hash(value) {
  let number = value >>> 0;
  number = Math.imul(number ^ (number >>> 16), 0x7feb352d);
  number = Math.imul(number ^ (number >>> 15), 0x846ca68b);
  return ((number ^ (number >>> 16)) >>> 0) / 4294967296;
}

function approach(current, target, speed, delta) {
  return THREE.MathUtils.lerp(current, target, 1 - Math.exp(-speed * Math.min(delta, 0.08)));
}

function smoothStep(value) {
  const amount = THREE.MathUtils.clamp(value, 0, 1);
  return amount * amount * (3 - 2 * amount);
}

function makeTextTexture(text, color = '#d9e3df', small = false) {
  const canvas = document.createElement('canvas');
  const fontSize = small ? 27 : 30;
  const font = `${fontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
  let context = canvas.getContext('2d');
  context.font = font;
  canvas.width = Math.ceil(context.measureText(text.toUpperCase()).width + 24);
  canvas.height = small ? 72 : 96;
  context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.font = font;
  context.textBaseline = 'middle';
  context.fillStyle = color;
  context.fillText(text.toUpperCase(), 8, canvas.height * 0.5);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;
  texture.userData.aspect = canvas.width / canvas.height;
  return texture;
}

function fillEllipsoid(target, offset, randomA, randomB, randomC, center, radii) {
  const theta = randomA * TAU;
  const cosPhi = randomB * 2 - 1;
  const sinPhi = Math.sqrt(Math.max(0, 1 - cosPhi * cosPhi));
  const radius = Math.cbrt(randomC);
  target[offset] = center[0] + Math.cos(theta) * sinPhi * radii[0] * radius;
  target[offset + 1] = center[1] + cosPhi * radii[1] * radius;
  target[offset + 2] = center[2] + Math.sin(theta) * sinPhi * radii[2] * radius;
}

function setSignaturePoint(target, offset, stableId, randoms) {
  const [selector, progress, depth, spread] = randoms;
  let cursor = selector * SIGNATURE_LENGTH;
  let segment = SIGNATURE_SEGMENTS[SIGNATURE_SEGMENTS.length - 1];
  for (let index = 0; index < SIGNATURE_SEGMENTS.length; index += 1) {
    const candidate = SIGNATURE_SEGMENTS[index];
    if (cursor <= candidate.length) {
      segment = candidate;
      break;
    }
    cursor -= candidate.length;
  }

  const dx = segment.end[0] - segment.start[0];
  const dy = segment.end[1] - segment.start[1];
  const magnitude = Math.max(0.001, Math.hypot(dx, dy));
  const normalX = -dy / magnitude;
  const normalY = dx / magnitude;
  const normalOffset = (spread + hash(stableId + 2333) - 1) * 0.105;
  const tangentOffset = (hash(stableId + 2371) - 0.5) * 0.035;

  target[offset] = THREE.MathUtils.lerp(segment.start[0], segment.end[0], progress)
    + normalX * normalOffset + (dx / magnitude) * tangentOffset;
  target[offset + 1] = THREE.MathUtils.lerp(segment.start[1], segment.end[1], progress)
    + normalY * normalOffset + (dy / magnitude) * tangentOffset;
  target[offset + 2] = (depth + hash(stableId + 2411) - 1) * 0.17
    + Math.sin(progress * Math.PI) * (segment.letter === 0 ? -0.025 : 0.025);

  return segment.letter;
}

export class UMAPSeagullEngine {
  constructor({ data, mobile = false } = {}) {
    this.data = data;
    this.mobile = mobile;
    this.root = new THREE.Group();
    this.root.name = 'V6 UMAP Seagull Atlas';
    this.content = new THREE.Group();
    this.root.add(this.content);
    this.lastTime = 0;
    this.arrival = 0;
    this.hoverTarget = 0;
    this.hoverAmount = 0;
    this.pointer = new THREE.Vector2();
    this.sceneWeights = new Float32Array(TARGET_COUNT);
    this.targets = Array.from({ length: TARGET_COUNT }, () => new Float32Array(POINT_COUNT * 3));
    this.positions = new Float32Array(POINT_COUNT * 3);
    this.birdColors = new Float32Array(POINT_COUNT * 3);
    this.bioColors = new Float32Array(POINT_COUNT * 3);
    this.atlasColors = new Float32Array(POINT_COUNT * 3);
    this.prototypeColors = new Float32Array(POINT_COUNT * 3);
    this.studyColors = new Float32Array(POINT_COUNT * 3);
    this.signatureColors = new Float32Array(POINT_COUNT * 3);
    this.studyTargets = Array.from({ length: 3 }, () => new Float32Array(POINT_COUNT * 3));
    this.studyColorTargets = Array.from({ length: 3 }, () => new Float32Array(POINT_COUNT * 3));
    this.colors = new Float32Array(POINT_COUNT * 3);
    this.phases = new Float32Array(POINT_COUNT);
    this.stagger = new Float32Array(POINT_COUNT);
    this.pointSizes = new Float32Array(POINT_COUNT);
    this.populations = new Uint8Array(POINT_COUNT);
    this.birdParts = new Uint8Array(POINT_COUNT);
    this.wingPanels = new Uint8Array(POINT_COUNT);
    this.prototypeRoles = new Uint8Array(POINT_COUNT);
    this.disposables = [];
    this.labelEntries = [];
    this.studyLabelEntries = [];
    this.atlasCentroids = [];
    this.studyPhase = 2;
    this.studyStartTime = 0;
    this.wasStudyActive = false;
    this.white = new THREE.Color(0xe7ece8);
    this.grey = new THREE.Color(0x8f9ca2);
    this.darkGrey = new THREE.Color(0x485760);
    this.yellow = new THREE.Color(0xd6ad55);
    this.red = new THREE.Color(0xb94f53);
    this.black = new THREE.Color(0x182025);
    this.signatureTeal = new THREE.Color(0x8fd5c5);
    this.signatureGold = new THREE.Color(0xe0ad68);
    this.wingGraphite = new THREE.Color(0x465c5d);
    this.featherTeal = new THREE.Color(0x739b96);
    this.outerFeatherTeal = new THREE.Color(0xa6d3ca);
    this.auctionBone = new THREE.Color(0xd9d4c8);
    this.auctionGraphite = new THREE.Color(0x526164);
    this.auctionSlate = new THREE.Color(0x82949b);
    this.auctionTeal = new THREE.Color(0x6fa89d);
    this.auctionViolet = new THREE.Color(0x9186b1);
    this.auctionAmber = new THREE.Color(0xe0ad68);
    this.tempColor = new THREE.Color();

    this.buildTargets();
    this.buildPoints();
    this.buildPlinth();
    this.buildLabels();
    this.setMobile(mobile);
  }

  track(resource) {
    this.disposables.push(resource);
    return resource;
  }

  setBirdPoint(index, target, offset, randoms) {
    const [a, b, c, d] = randoms;
    let color = this.white;
    let part = 0;

    if (a < 0.42) {
      fillEllipsoid(target, offset, b, c, d, [-0.1, -0.2, 0], [1.04, 1.06, 0.59]);
      target[offset] -= Math.max(0, -target[offset + 1] - 0.72) * 0.18;
    } else if (a < 0.58) {
      part = 1;
      if (b < 0.52) {
        fillEllipsoid(target, offset, c, d, hash(index + 73), [0.15, 0.64, 0.03], [0.47, 0.54, 0.4]);
      } else {
        fillEllipsoid(target, offset, c, d, hash(index + 79), [0.31, 1.13, 0.02], [0.57, 0.54, 0.48]);
      }
    } else if (a < 0.76) {
      part = 2;
      fillEllipsoid(target, offset, b, c, d, [-0.48, -0.1, 0.24], [0.77, 0.86, 0.24]);
      target[offset] -= (target[offset + 1] + 0.15) * 0.18;
      color = b > 0.42 ? this.darkGrey : this.grey;
    } else if (a < 0.83) {
      part = 3;
      const length = b;
      target[offset] = 0.58 + length * 0.68;
      target[offset + 1] = 1.19 + (c - 0.5) * (0.24 * (1 - length * 0.65));
      target[offset + 2] = (d - 0.5) * 0.2 * (1 - length * 0.5);
      if (length > 0.72 && c < 0.18) color = this.red;
      else color = this.yellow;
    } else if (a < 0.915) {
      part = 4;
      const rightLeg = b > 0.5;
      const t = c;
      target[offset] = (rightLeg ? 0.16 : -0.28) + (d - 0.5) * 0.06;
      target[offset + 1] = -0.9 - t * 0.92;
      target[offset + 2] = (rightLeg ? 0.08 : -0.06) + (hash(index + 91) - 0.5) * 0.07;
      color = this.yellow;
    } else if (a < 0.975) {
      part = 5;
      const rightFoot = b > 0.5;
      const toe = Math.floor(c * 3);
      const t = d;
      target[offset] = (rightFoot ? 0.16 : -0.28) + (t - 0.25) * 0.46;
      target[offset + 1] = -1.82 + (hash(index + 97) - 0.5) * 0.045;
      target[offset + 2] = (toe - 1) * 0.11 + (hash(index + 101) - 0.5) * 0.035;
      color = this.yellow;
    } else if (a < 0.989) {
      part = 6;
      fillEllipsoid(target, offset, b, c, d, [0.55, 1.31, 0.34], [0.08, 0.08, 0.045]);
      color = b < 0.45 ? this.red : this.black;
    } else {
      part = 7;
      const t = b;
      target[offset] = -0.9 - t * 0.32;
      target[offset + 1] = -0.34 + (c - 0.5) * 0.28;
      target[offset + 2] = (d - 0.5) * 0.34;
      color = this.grey;
    }

    this.birdColors[offset] = color.r;
    this.birdColors[offset + 1] = color.g;
    this.birdColors[offset + 2] = color.b;
    this.birdParts[index] = part;
  }

  buildTargets() {
    const points = this.data.points;
    const populations = this.data.populations;
    const atlasTarget = this.targets[3];
    const birdTarget = this.targets[0];
    const collapsedTarget = this.targets[1];
    const sphereTarget = this.targets[2];
    const exploreTarget = this.targets[4];
    const studyTarget = this.targets[5];
    const rebuildTarget = this.targets[6];
    const finalTarget = this.targets[7];
    const populationColors = populations.map((population) => new THREE.Color(population.color));
    const populationStarts = new Uint16Array(populations.length);
    let populationStart = 0;
    populations.forEach((population, index) => {
      populationStarts[index] = populationStart;
      populationStart += population.count;
      this.atlasCentroids.push(new THREE.Vector3(...BRANCH_CENTERS[index]));
    });

    for (let index = 0; index < POINT_COUNT; index += 1) {
      const offset = index * 3;
      const source = points[index];
      const stableId = source[0];
      const population = source[1];
      const a = hash(stableId + 11);
      const b = hash(stableId + 211);
      const c = hash(stableId + 419);
      const d = hash(stableId + 827);
      this.phases[index] = a * TAU;
      this.stagger[index] = hash(stableId + 1229);
      this.pointSizes[index] = 0.95 + hash(stableId + 1877) * 0.1;
      this.populations[index] = population;

      this.setBirdPoint(index, birdTarget, offset, [a, b, c, d]);

      const birdPart = this.birdParts[index];

      // Every non-wing point begins at its original bird position. This keeps
      // the full anatomy intact while target 1 articulates only wings and tail.
      collapsedTarget[offset] = birdTarget[offset];
      collapsedTarget[offset + 1] = birdTarget[offset + 1];
      collapsedTarget[offset + 2] = birdTarget[offset + 2];

      let wingEdge = 0;
      let wingRootWeight = 0;
      let nearWing = false;

      if (birdPart === 2) {
        nearWing = hash(stableId + 3521) < 0.56;
        this.wingPanels[index] = nearWing ? 1 : 2;

        const spanSeed = hash(stableId + 3581);
        const chordSeed = hash(stableId + 3643);
        const depthSeed = hash(stableId + 3691);
        const span = spanSeed * spanSeed;
        const chord = chordSeed * 2 - 1;
        const viewportFactor = this.mobile ? 0.82 : 1;
        const shoulder = nearWing
          ? [-0.34, 0.34, 0.17]
          : [0.02, 0.45, -0.24];
        const direction = nearWing ? -1 : 1;
        const fullSpan = (nearWing ? 2.02 : 1.7) * viewportFactor;
        const rise = (nearWing ? 1.06 : 0.94) * viewportFactor;
        const sweep = Math.sin(span * Math.PI) * (nearWing ? 0.24 : 0.18);
        const centerX = shoulder[0] + direction * fullSpan * span;
        const centerY = shoulder[1] + rise * span + sweep;
        const centerZ = shoulder[2] + direction * 0.035 * span;
        const taper = Math.pow(Math.sin(Math.PI * span), 0.62) * (1 - span * 0.26);
        const rootBridge = (1 - span) * 0.075;
        const halfChord = ((nearWing ? 0.55 : 0.44) * taper + rootBridge) * viewportFactor;
        const featherNotch = Math.sin((span * 7.5 + chordSeed * 1.8) * Math.PI)
          * 0.035 * span * span;

        collapsedTarget[offset] = centerX + direction * chord * halfChord * 0.16;
        collapsedTarget[offset + 1] = centerY + chord * halfChord + featherNotch;
        collapsedTarget[offset + 2] = centerZ
          + (depthSeed - 0.5) * (nearWing ? 0.18 : 0.12)
          + chord * (nearWing ? 0.035 : -0.025);
        wingRootWeight = smoothStep(span / 0.16);
        wingEdge = Math.max(span, Math.abs(chord));
      } else if (birdPart === 7) {
        collapsedTarget[offset] -= 0.05;
        collapsedTarget[offset + 2] -= 0.02;
      }

      this.tempColor.setRGB(
        this.birdColors[offset], this.birdColors[offset + 1], this.birdColors[offset + 2],
      );
      if (birdPart === 2) {
        const panelColor = nearWing ? this.featherTeal : this.wingGraphite;
        this.tempColor.copy(this.white).lerp(
          panelColor,
          wingRootWeight * (nearWing ? 0.96 : 0.86),
        );
        this.tempColor.lerp(
          this.outerFeatherTeal,
          smoothStep((wingEdge - 0.58) / 0.42) * wingRootWeight * (nearWing ? 0.74 : 0.38),
        );
      }
      this.bioColors[offset] = this.tempColor.r;
      this.bioColors[offset + 1] = this.tempColor.g;
      this.bioColors[offset + 2] = this.tempColor.b;

      const sphereTheta = b * TAU;
      const sphereY = c * 2 - 1;
      const sphereRadius = 1.34 * Math.cbrt(0.18 + d * 0.82);
      const spherePlanar = Math.sqrt(Math.max(0, 1 - sphereY * sphereY));
      sphereTarget[offset] = Math.cos(sphereTheta) * spherePlanar * sphereRadius;
      sphereTarget[offset + 1] = sphereY * sphereRadius;
      sphereTarget[offset + 2] = Math.sin(sphereTheta) * spherePlanar * sphereRadius;

      const sourceCentroid = this.data.centroids[population];
      const branchCenter = BRANCH_CENTERS[population];
      const parentIndex = populations[population].parent;
      const localIndex = stableId - populationStarts[population];
      const isBridge = parentIndex !== null && localIndex % 9 === 0;
      let centerX = branchCenter[0];
      let centerY = branchCenter[1];
      let centerZ = branchCenter[2];
      let localScale = 0.88;
      if (isBridge) {
        const parentCenter = BRANCH_CENTERS[parentIndex];
        const bridgeProgress = 0.08 + hash(stableId + 1601) * 0.84;
        centerX = THREE.MathUtils.lerp(parentCenter[0], branchCenter[0], bridgeProgress);
        centerY = THREE.MathUtils.lerp(parentCenter[1], branchCenter[1], bridgeProgress);
        centerZ = THREE.MathUtils.lerp(parentCenter[2], branchCenter[2], bridgeProgress);
        localScale = 0.18;
      }
      const atlasX = centerX + (source[2] - sourceCentroid[0]) * localScale;
      const atlasY = centerY + (source[3] - sourceCentroid[1]) * localScale;
      const atlasZ = centerZ + (source[4] - sourceCentroid[2]) * localScale;
      atlasTarget[offset] = atlasX;
      atlasTarget[offset + 1] = atlasY;
      atlasTarget[offset + 2] = atlasZ;
      const signatureLetter = setSignaturePoint(finalTarget, offset, stableId, [a, b, c, d]);

      // Advertising is represented as an auction iris: audience signals enter
      // from the left, fourteen competing blades evaluate them, and a restrained
      // amber winner signal resolves through the aperture on the right.
      const auctionRole = hash(stableId + 4051);
      if (auctionRole < 0.065) {
        const progress = hash(stableId + 4111);
        const lane = Math.floor(hash(stableId + 4153) * 3);
        const laneY = (lane - 1) * 0.64;
        exploreTarget[offset] = -3.05 + progress * 2.25 + (a - 0.5) * 0.08;
        exploreTarget[offset + 1] = laneY * (1 - progress * 0.72)
          + Math.sin(progress * Math.PI) * 0.16
          + (c - 0.5) * 0.12;
        exploreTarget[offset + 2] = (d - 0.5) * 0.3 + (lane - 1) * 0.07;
        this.prototypeRoles[index] = 1;
        this.tempColor.copy(lane === 1 ? this.auctionBone : this.auctionTeal)
          .lerp(this.auctionSlate, progress * 0.28);
      } else if (auctionRole > 0.955) {
        const progress = hash(stableId + 4211);
        exploreTarget[offset] = 0.18 + progress * 3.02;
        exploreTarget[offset + 1] = Math.sin(progress * Math.PI * 2 + this.phases[index]) * 0.045
          + (c - 0.5) * 0.055;
        exploreTarget[offset + 2] = (d - 0.5) * 0.14;
        this.prototypeRoles[index] = 2;
        this.tempColor.copy(this.auctionBone).lerp(this.auctionAmber, 0.58 + progress * 0.42);
      } else {
        const bladeAngle = (population / populations.length) * TAU - 0.38;
        const bladeProgress = Math.pow(hash(stableId + 4271), 0.76);
        const bladeAcross = hash(stableId + 4337) * 2 - 1;
        const bladeWidth = (0.18 * (1 - bladeProgress) + 0.045) * bladeAcross;
        const angle = bladeAngle + bladeProgress * 0.62 + bladeWidth;
        const radius = 0.64 + bladeProgress * 1.72;
        const camber = Math.sin(bladeProgress * Math.PI)
          * (0.13 + Math.sin(bladeAngle * 2) * 0.025);
        exploreTarget[offset] = Math.cos(angle) * radius * 1.3 - 0.08;
        exploreTarget[offset + 1] = Math.sin(angle) * radius * 0.9 + 0.06;
        exploreTarget[offset + 2] = Math.sin(bladeAngle * 1.5) * 0.2
          + bladeAcross * (0.07 + (1 - bladeProgress) * 0.055)
          + camber;

        const bladeFamily = population % 5;
        if (bladeFamily === 0) this.tempColor.copy(this.auctionBone);
        else if (bladeFamily === 1) this.tempColor.copy(this.auctionGraphite);
        else if (bladeFamily === 2) this.tempColor.copy(this.auctionTeal);
        else if (bladeFamily === 3) this.tempColor.copy(this.auctionSlate);
        else this.tempColor.copy(this.auctionViolet);
        const innerEdge = 1 - smoothStep((bladeProgress - 0.02) / 0.34);
        this.tempColor.lerp(this.white, innerEdge * 0.28);
      }
      this.prototypeColors[offset] = this.tempColor.r;
      this.prototypeColors[offset + 1] = this.tempColor.g;
      this.prototypeColors[offset + 2] = this.tempColor.b;

      const localX = source[2] - sourceCentroid[0];
      const localY = source[3] - sourceCentroid[1];
      const localZ = source[4] - sourceCentroid[2];
      const peripheryAngle = (population / populations.length) * TAU + a * 0.22;
      const peripheryRadius = 2.22 + (population % 3) * 0.18;
      for (let studyIndex = 0; studyIndex < 3; studyIndex += 1) {
        const study = this.studyTargets[studyIndex];
        const studyColors = this.studyColorTargets[studyIndex];
        const focusedPopulation = studyIndex === 0 ? 5 : 3;
        const isCompositeFocus = population === 5 || population === 3 || population === 7;
        const isFocused = studyIndex < 2 ? population === focusedPopulation : isCompositeFocus;
        let studyCenterX = Math.cos(peripheryAngle) * peripheryRadius;
        let studyCenterY = Math.sin(peripheryAngle) * peripheryRadius * 0.62;
        let studyCenterZ = (population % 4 - 1.5) * 0.28;
        let studyScaleX = 0.2;
        let studyScaleY = 0.2;
        let studyScaleZ = 0.2;
        if (studyIndex < 2 && isFocused) {
          studyCenterX = 0;
          studyCenterY = 0.06;
          studyCenterZ = 0.18;
          studyScaleX = 1.95;
          studyScaleY = 3.45;
          studyScaleZ = 3.15;
        } else if (studyIndex === 2 && isFocused) {
          studyCenterX = population === 5 ? -0.78 : (population === 3 ? 0 : 0.78);
          studyCenterY = population === 5 ? 0.5 : (population === 3 ? 0.05 : -0.52);
          studyCenterZ = population === 3 ? 0.24 : -0.04;
          studyScaleX = 1.25;
          studyScaleY = 1.6;
          studyScaleZ = 1.55;
        }
        const studyVolume = isFocused ? (studyIndex < 2 ? 0.34 : 0.16) : 0;
        study[offset] = studyCenterX + localX * studyScaleX;
        study[offset + 1] = studyCenterY + localY * studyScaleY
          + Math.sin(this.phases[index] * 1.37 + studyIndex) * studyVolume;
        study[offset + 2] = studyCenterZ + localZ * studyScaleZ
          + Math.cos(this.phases[index] * 1.73 + studyIndex) * studyVolume * 1.45;

        this.tempColor.copy(populationColors[population]);
        if (!isFocused) this.tempColor.multiplyScalar(studyIndex === 2 ? 0.13 : 0.1);
        else if (studyIndex === 0) this.tempColor.setHex(0x74ca8e);
        else if (studyIndex === 1) this.tempColor.setHex(0x70a7d6);
        studyColors[offset] = this.tempColor.r;
        studyColors[offset + 1] = this.tempColor.g;
        studyColors[offset + 2] = this.tempColor.b;
      }
      studyTarget[offset] = this.studyTargets[2][offset];
      studyTarget[offset + 1] = this.studyTargets[2][offset + 1];
      studyTarget[offset + 2] = this.studyTargets[2][offset + 2];

      rebuildTarget[offset] = atlasX;
      rebuildTarget[offset + 1] = atlasY;
      rebuildTarget[offset + 2] = atlasZ;

      const atlasColor = populationColors[population];
      this.atlasColors[offset] = atlasColor.r;
      this.atlasColors[offset + 1] = atlasColor.g;
      this.atlasColors[offset + 2] = atlasColor.b;
      const signatureColor = signatureLetter === 0 ? this.signatureTeal : this.signatureGold;
      this.tempColor.copy(atlasColor).lerp(signatureColor, 0.74);
      this.signatureColors[offset] = this.tempColor.r;
      this.signatureColors[offset + 1] = this.tempColor.g;
      this.signatureColors[offset + 2] = this.tempColor.b;
      this.studyColors[offset] = this.studyColorTargets[2][offset];
      this.studyColors[offset + 1] = this.studyColorTargets[2][offset + 1];
      this.studyColors[offset + 2] = this.studyColorTargets[2][offset + 2];
    }

    this.positions.set(birdTarget);
    this.colors.set(this.birdColors);
  }

  buildPoints() {
    const geometry = this.track(new THREE.BufferGeometry());
    geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(this.colors, 3));
    geometry.setAttribute('aSize', new THREE.BufferAttribute(this.pointSizes, 1));
    geometry.setDrawRange(0, POINT_COUNT);
    const material = this.track(new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      depthTest: true,
      vertexColors: true,
      uniforms: {
        uPointSize: { value: 0.304 },
        uOpacity: { value: 0.88 },
      },
      vertexShader: `
        uniform float uPointSize;
        attribute float aSize;
        varying vec3 vColor;
        void main() {
          vColor = color;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = clamp(uPointSize * aSize * (115.0 / max(1.0, -mvPosition.z)), 1.65, 8.5);
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        uniform float uOpacity;
        varying vec3 vColor;
        void main() {
          vec2 centered = gl_PointCoord - vec2(0.5);
          float distanceToCenter = length(centered) * 2.0;
          float alpha = 1.0 - smoothstep(0.54, 1.0, distanceToCenter);
          alpha *= 0.82 + (1.0 - distanceToCenter) * 0.18;
          if (alpha < 0.015) discard;
          float sphereZ = sqrt(max(0.0, 1.0 - dot(centered * 2.0, centered * 2.0)));
          vec3 normal = normalize(vec3(centered.x * 2.0, -centered.y * 2.0, sphereZ));
          float diffuse = 0.72 + max(0.0, dot(normal, normalize(vec3(-0.42, 0.58, 0.7)))) * 0.4;
          float highlight = pow(max(0.0, sphereZ), 7.0) * 0.22;
          gl_FragColor = vec4(vColor * diffuse + highlight, alpha * uOpacity);
        }
      `,
    }));
    this.pointsMaterial = material;
    this.points = new THREE.Points(geometry, material);
    this.points.frustumCulled = false;
    this.content.add(this.points);
  }

  buildPlinth() {
    this.plinth = new THREE.Group();
    const ringGeometry = this.track(new THREE.RingGeometry(0.92, 0.94, 80));
    const ringMaterial = this.track(new THREE.MeshBasicMaterial({
      color: 0x8ea29d,
      transparent: true,
      opacity: 0.18,
      depthWrite: false,
      side: THREE.DoubleSide,
    }));
    const ring = new THREE.Mesh(ringGeometry, ringMaterial);
    ring.rotation.x = -Math.PI / 2;
    ring.scale.x = 1.3;
    const lineGeometry = this.track(new THREE.BufferGeometry());
    lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute([
      -1.18, 0, 0, 1.18, 0, 0,
      0, 0, -0.72, 0, 0, 0.72,
      -0.48, 0, -0.48, 0.48, 0, 0.48,
      -0.48, 0, 0.48, 0.48, 0, -0.48,
    ], 3));
    const lineMaterial = this.track(new THREE.LineBasicMaterial({
      color: 0x7faaa2,
      transparent: true,
      opacity: 0.12,
      depthWrite: false,
    }));
    const guides = new THREE.LineSegments(lineGeometry, lineMaterial);
    this.plinth.position.set(-0.1, -1.91, 0);
    this.plinth.scale.set(1.06, 1, 0.76);
    this.plinth.add(ring, guides);
    this.plinthRingMaterial = ringMaterial;
    this.plinthLineMaterial = lineMaterial;
    this.content.add(this.plinth);
  }

  buildLabels() {
    this.labels = new THREE.Group();
    LABEL_INDICES.forEach((populationIndex, labelIndex) => {
      const centroid = this.atlasCentroids[populationIndex];
      const direction = centroid.x >= -0.15 ? 1 : -1;
      const endpoint = new THREE.Vector3(
        centroid.x + direction * (0.48 + (labelIndex % 3) * 0.08),
        centroid.y + (labelIndex % 3 - 1) * 0.18,
        centroid.z + 0.04,
      );
      const lineGeometry = this.track(new THREE.BufferGeometry().setFromPoints([centroid, endpoint]));
      const lineMaterial = this.track(new THREE.LineBasicMaterial({
        color: this.data.populations[populationIndex].color,
        transparent: true,
        opacity: 0,
        depthWrite: false,
      }));
      const line = new THREE.Line(lineGeometry, lineMaterial);
      const texture = this.track(makeTextTexture(this.data.populations[populationIndex].label, '#cfdbd7', true));
      const spriteMaterial = this.track(new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        opacity: 0,
        depthWrite: false,
        depthTest: false,
      }));
      const sprite = new THREE.Sprite(spriteMaterial);
      sprite.position.copy(endpoint);
      const labelHeight = 0.22;
      sprite.position.x += direction * labelHeight * texture.userData.aspect * 0.46;
      sprite.scale.set(labelHeight * texture.userData.aspect, labelHeight, 1);
      this.labels.add(line, sprite);
      this.labelEntries.push({
        line,
        lineMaterial,
        sprite,
        spriteMaterial,
        basePosition: sprite.position.clone(),
        baseScaleX: sprite.scale.x,
        baseScaleY: sprite.scale.y,
      });
    });

    this.studyLabels = new THREE.Group();
    [
      { text: 'EMBEDDING SPACE', color: '#7fc995' },
      { text: 'EXPERIMENT COHORTS', color: '#78a9d4' },
      { text: 'EMBEDDING · COHORTS · PRODUCT', color: '#b8aaa3' },
    ].forEach((specification) => {
      const start = new THREE.Vector3(0, -0.78, 0.12);
      const end = new THREE.Vector3(0, -1.02, 0.12);
      const lineGeometry = this.track(new THREE.BufferGeometry().setFromPoints([start, end]));
      const lineMaterial = this.track(new THREE.LineBasicMaterial({
        color: specification.color,
        transparent: true,
        opacity: 0,
        depthWrite: false,
      }));
      const line = new THREE.Line(lineGeometry, lineMaterial);
      const texture = this.track(makeTextTexture(specification.text, specification.color, true));
      const spriteMaterial = this.track(new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        opacity: 0,
        depthWrite: false,
        depthTest: false,
      }));
      const sprite = new THREE.Sprite(spriteMaterial);
      const labelHeight = 0.21;
      sprite.position.set(0, -1.15, 0.12);
      sprite.scale.set(labelHeight * texture.userData.aspect, labelHeight, 1);
      this.studyLabels.add(line, sprite);
      this.studyLabelEntries.push({
        lineMaterial,
        spriteMaterial,
        sprite,
        baseScaleX: sprite.scale.x,
        baseScaleY: sprite.scale.y,
      });
    });

    this.overview = new THREE.Group();
    const headerTexture = this.track(makeTextTexture('UMAP / SEMANTIC POPULATION ATLAS', '#dce6e1'));
    const headerMaterial = this.track(new THREE.SpriteMaterial({
      map: headerTexture,
      transparent: true,
      opacity: 0,
      depthWrite: false,
      depthTest: false,
    }));
    const header = new THREE.Sprite(headerMaterial);
    header.position.set(0, 2.35, 0.1);
    header.scale.set(0.23 * headerTexture.userData.aspect, 0.23, 1);
    const captionTexture = this.track(makeTextTexture('6,000 POINTS · 14 POPULATIONS · 3D UMAP', '#829a94', true));
    const captionMaterial = this.track(new THREE.SpriteMaterial({
      map: captionTexture,
      transparent: true,
      opacity: 0,
      depthWrite: false,
      depthTest: false,
    }));
    const caption = new THREE.Sprite(captionMaterial);
    caption.position.set(0, -2.12, 0.1);
    caption.scale.set(0.16 * captionTexture.userData.aspect, 0.16, 1);
    this.overview.add(header, caption);
    this.overviewMaterials = [headerMaterial, captionMaterial];
    this.overviewSprites = [
      { sprite: header, baseScaleX: header.scale.x, baseScaleY: header.scale.y },
      { sprite: caption, baseScaleX: caption.scale.x, baseScaleY: caption.scale.y },
    ];
    this.content.add(this.labels, this.studyLabels, this.overview);
  }

  setMobile(mobile) {
    this.mobile = mobile;
    this.compactLegend = mobile && window.matchMedia('(max-width: 430px) and (max-height: 900px)').matches;
    if (!this.points) return;
    this.points.geometry.setDrawRange(0, mobile ? MOBILE_POINT_COUNT : POINT_COUNT);
    this.pointsMaterial.uniforms.uPointSize.value = mobile ? 0.368 : 0.304;
    this.labelEntries.forEach((entry, index) => {
      entry.line.visible = !mobile || index < MOBILE_LABEL_COUNT;
      entry.sprite.visible = !mobile || index < MOBILE_LABEL_COUNT;
    });
    this.overview.visible = !mobile;
    this.studyLabels.visible = true;
  }

  setProjectFocus(active) {
    this.hoverTarget = active ? 1 : 0;
  }

  readWeights(state) {
    const weights = this.sceneWeights;
    weights[0] = Math.max(0, state.assembled ?? 0);
    weights[1] = Math.max(0, state.open ?? 0);
    weights[2] = Math.max(0, state.operating ?? 0);
    weights[3] = Math.max(0, state.structure ?? 0);
    weights[4] = Math.max(0, state.prototype ?? 0);
    weights[5] = Math.max(0, state.action ?? 0);
    weights[6] = Math.max(0, state.blueprint ?? 0);
    weights[7] = Math.max(0, state.resolution ?? 0);
    let total = 0;
    for (let index = 0; index < TARGET_COUNT; index += 1) total += weights[index];
    if (total <= 0) {
      weights[0] = 1;
      total = 1;
    }
    const inverse = 1 / total;
    for (let index = 0; index < TARGET_COUNT; index += 1) weights[index] *= inverse;
  }

  updatePoints(time, reducedMotion, delta) {
    const weights = this.sceneWeights;
    const count = this.mobile ? MOBILE_POINT_COUNT : POINT_COUNT;
    const positionEase = reducedMotion ? 1 : 1 - Math.exp(-Math.min(delta, 0.05) * 5.4);
    const birdMix = THREE.MathUtils.clamp(weights[0] + weights[1], 0, 1);
    const bioWithinBird = weights[1] / Math.max(0.001, weights[0] + weights[1]);
    const birdOnlyTransition = birdMix > 0.999;
    const prototypeMix = weights[4];
    const prototypeMotion = reducedMotion ? 0 : prototypeMix;
    const prototypeTarget = this.targets[4];
    const irisAngle = prototypeMotion > 0.001 ? Math.sin(time * 0.34) * 0.03 : 0;
    const irisBreath = prototypeMotion > 0.001
      ? 1 + Math.sin(time * 0.5 + 0.6) * 0.028
      : 1;
    const irisCos = Math.cos(irisAngle);
    const irisSin = Math.sin(irisAngle);
    const irisCenterX = -0.08;
    const irisCenterY = 0.06;
    const studyMix = weights[5];
    const signatureMix = weights[7];
    const sphereMix = weights[2] * 0.92;
    const activeStudyTarget = this.studyTargets[this.studyPhase];
    const activeStudyColors = this.studyColorTargets[this.studyPhase];
    const pointerX = this.pointer.x * 2.15;
    const pointerY = this.pointer.y * 1.5;

    for (let index = 0; index < count; index += 1) {
      const offset = index * 3;
      let targetX = 0;
      let targetY = 0;
      let targetZ = 0;
      for (let targetIndex = 0; targetIndex < TARGET_COUNT; targetIndex += 1) {
        const influence = weights[targetIndex];
        if (influence <= 0) continue;
        const target = targetIndex === 5 ? activeStudyTarget : this.targets[targetIndex];
        targetX += target[offset] * influence;
        targetY += target[offset + 1] * influence;
        targetZ += target[offset + 2] * influence;
      }

      let pointBioWithinBird = bioWithinBird;
      if (birdOnlyTransition && this.wingPanels[index] === 1) {
        pointBioWithinBird = smoothStep((bioWithinBird - 0.06) / 0.94);
      } else if (birdOnlyTransition && this.wingPanels[index] === 2) {
        pointBioWithinBird = smoothStep(bioWithinBird / 0.94);
      }
      if (birdOnlyTransition && this.wingPanels[index] > 0) {
        const hero = this.targets[0];
        const bio = this.targets[1];
        targetX = THREE.MathUtils.lerp(hero[offset], bio[offset], pointBioWithinBird);
        targetY = THREE.MathUtils.lerp(hero[offset + 1], bio[offset + 1], pointBioWithinBird);
        targetZ = THREE.MathUtils.lerp(hero[offset + 2], bio[offset + 2], pointBioWithinBird);
      }

      if (prototypeMotion > 0.001) {
        const role = this.prototypeRoles[index];
        if (role === 0) {
          const baseX = prototypeTarget[offset] - irisCenterX;
          const baseY = prototypeTarget[offset + 1] - irisCenterY;
          const animatedX = irisCenterX
            + (baseX * irisCos - baseY * irisSin) * irisBreath;
          const animatedY = irisCenterY
            + (baseX * irisSin + baseY * irisCos) * irisBreath;
          targetX += (animatedX - prototypeTarget[offset]) * prototypeMotion;
          targetY += (animatedY - prototypeTarget[offset + 1]) * prototypeMotion;
        } else {
          const signalSpeed = role === 2 ? 1.45 : 0.82;
          const signalX = role === 2 ? 0.062 : 0.05;
          const signalY = role === 2 ? 0.028 : 0.023;
          targetX += Math.sin(time * signalSpeed + this.phases[index]) * signalX * prototypeMotion;
          targetY += Math.cos(time * signalSpeed * 0.72 + this.phases[index]) * signalY * prototypeMotion;
          targetZ += Math.sin(time * signalSpeed * 0.54 + this.phases[index]) * 0.018 * prototypeMotion;
        }
      }

      if (!reducedMotion && index % 17 === 0) {
        const drift = 0.014 + weights[6] * 0.009;
        targetX += Math.cos(time * 0.31 + this.phases[index]) * drift;
        targetY += Math.sin(time * 0.27 + this.phases[index]) * drift;
        targetZ += Math.sin(time * 0.23 + this.phases[index]) * drift * 0.7;
      }

      if (!reducedMotion) {
        const pointerDx = pointerX - targetX;
        const pointerDy = pointerY - targetY;
        const distanceSquared = pointerDx * pointerDx + pointerDy * pointerDy;
        if (distanceSquared < 0.48) {
          const localPull = (1 - distanceSquared / 0.48) * 0.04;
          targetX += pointerDx * localPull;
          targetY += pointerDy * localPull;
          targetZ += localPull * 0.035;
        }
      }

      if (reducedMotion) {
        this.positions[offset] = targetX;
        this.positions[offset + 1] = targetY;
        this.positions[offset + 2] = targetZ;
      } else {
        const pointEase = Math.min(1, positionEase * (0.72 + this.stagger[index] * 0.52));
        this.positions[offset] += (targetX - this.positions[offset]) * pointEase;
        this.positions[offset + 1] += (targetY - this.positions[offset + 1]) * pointEase;
        this.positions[offset + 2] += (targetZ - this.positions[offset + 2]) * pointEase;
      }

      const atlasMix = 1 - birdMix;
      const birdR = THREE.MathUtils.lerp(
        this.birdColors[offset], this.bioColors[offset], pointBioWithinBird,
      );
      const birdG = THREE.MathUtils.lerp(
        this.birdColors[offset + 1], this.bioColors[offset + 1], pointBioWithinBird,
      );
      const birdB = THREE.MathUtils.lerp(
        this.birdColors[offset + 2], this.bioColors[offset + 2], pointBioWithinBird,
      );
      let targetR = birdR * birdMix + this.atlasColors[offset] * atlasMix;
      let targetG = birdG * birdMix + this.atlasColors[offset + 1] * atlasMix;
      let targetB = birdB * birdMix + this.atlasColors[offset + 2] * atlasMix;
      targetR = targetR * (1 - sphereMix) + 0.78 * sphereMix;
      targetG = targetG * (1 - sphereMix) + 0.74 * sphereMix;
      targetB = targetB * (1 - sphereMix) + 0.66 * sphereMix;
      targetR = targetR * (1 - prototypeMix) + this.prototypeColors[offset] * prototypeMix;
      targetG = targetG * (1 - prototypeMix) + this.prototypeColors[offset + 1] * prototypeMix;
      targetB = targetB * (1 - prototypeMix) + this.prototypeColors[offset + 2] * prototypeMix;
      targetR = targetR * (1 - studyMix) + activeStudyColors[offset] * studyMix;
      targetG = targetG * (1 - studyMix) + activeStudyColors[offset + 1] * studyMix;
      targetB = targetB * (1 - studyMix) + activeStudyColors[offset + 2] * studyMix;
      targetR = targetR * (1 - signatureMix) + this.signatureColors[offset] * signatureMix;
      targetG = targetG * (1 - signatureMix) + this.signatureColors[offset + 1] * signatureMix;
      targetB = targetB * (1 - signatureMix) + this.signatureColors[offset + 2] * signatureMix;
      if (reducedMotion) {
        this.colors[offset] = targetR;
        this.colors[offset + 1] = targetG;
        this.colors[offset + 2] = targetB;
      } else {
        const colorEase = Math.min(1, positionEase * (0.5 + this.stagger[index] * 0.75));
        this.colors[offset] += (targetR - this.colors[offset]) * colorEase;
        this.colors[offset + 1] += (targetG - this.colors[offset + 1]) * colorEase;
        this.colors[offset + 2] += (targetB - this.colors[offset + 2]) * colorEase;
      }
    }

    this.points.geometry.getAttribute('position').needsUpdate = true;
    this.points.geometry.getAttribute('color').needsUpdate = true;
  }

  updateAnnotations() {
    const weights = this.sceneWeights;
    const visibility = this.visibility ?? 1;
    const scaleCompensation = THREE.MathUtils.clamp(1 / Math.max(0.12, this.root.scale.x), 0.65, 3.2);
    const atlasOpacity = THREE.MathUtils.clamp(
      weights[3] * 0.86 + weights[6] * 0.82,
      0,
      0.86,
    ) * visibility;
    this.labelEntries.forEach((entry, index) => {
      const mobileFinal = this.mobile && weights[6] > 0.5 && index < MOBILE_LABEL_COUNT;
      if (mobileFinal) {
        if (this.compactLegend) {
          entry.sprite.position.set(-3.25, -1.52 - index * 0.42, entry.basePosition.z);
        } else {
          entry.sprite.position.set(-2.7, 1.02 - index * 0.56, entry.basePosition.z);
        }
      } else {
        entry.sprite.position.copy(entry.basePosition);
      }
      entry.lineMaterial.opacity = atlasOpacity * 0.5 * (mobileFinal ? 0 : 1);
      entry.spriteMaterial.opacity = Math.min(1, atlasOpacity * (mobileFinal ? 1.22 : 1));
      const mobileFinalScale = mobileFinal ? (this.compactLegend ? 1.08 : 1.2) : 1;
      entry.sprite.scale.set(
        entry.baseScaleX * scaleCompensation * mobileFinalScale,
        entry.baseScaleY * scaleCompensation * mobileFinalScale,
        1,
      );
    });
    const studyOpacity = weights[5] * 0.72 * visibility;
    this.studyLabelEntries.forEach((entry, index) => {
      const active = index === this.studyPhase ? 1 : 0;
      entry.lineMaterial.opacity = studyOpacity * active * 0.48;
      entry.spriteMaterial.opacity = studyOpacity * active;
      entry.sprite.scale.set(entry.baseScaleX * scaleCompensation, entry.baseScaleY * scaleCompensation, 1);
    });
    const overviewOpacity = weights[6] * 0.88 * visibility;
    this.overviewMaterials[0].opacity = overviewOpacity;
    this.overviewMaterials[1].opacity = overviewOpacity * 0.75;
    this.overviewSprites.forEach((entry) => {
      entry.sprite.scale.set(entry.baseScaleX * scaleCompensation, entry.baseScaleY * scaleCompensation, 1);
    });
    const specimenOpacity = THREE.MathUtils.clamp(
      weights[0] * 0.9 + weights[1] * 0.42 + weights[2] * 0.64,
      0,
      0.9,
    ) * visibility;
    this.plinthRingMaterial.opacity = specimenOpacity * 0.19;
    this.plinthLineMaterial.opacity = specimenOpacity * 0.13;
    this.plinth.visible = specimenOpacity > 0.01;
  }

  update(time, pointer, state = {}, reducedMotion = false) {
    const delta = Math.min(Math.max(time - this.lastTime, 0), 0.05) || 1 / 60;
    this.lastTime = time;
    this.pointer.copy(pointer || this.pointer);
    this.visibility = state.visibility ?? 1;
    this.readWeights(state);
    const studyActive = this.sceneWeights[5] > 0.5;
    if (studyActive && !this.wasStudyActive) this.studyStartTime = time;
    this.studyPhase = reducedMotion ? 2 : (studyActive ? Math.floor((time - this.studyStartTime) / 3) % 3 : 2);
    this.wasStudyActive = studyActive;
    this.arrival = reducedMotion ? 1 : smoothStep(Math.min(1, time / 1.2));
    this.hoverAmount = approach(this.hoverAmount, this.hoverTarget, reducedMotion ? 1000 : 5.2, delta);

    const rootEase = reducedMotion ? 1000 : 4.6;
    this.root.position.x = approach(this.root.position.x, state.x ?? 0, rootEase, delta);
    this.root.position.y = approach(this.root.position.y, state.y ?? 0, rootEase, delta);
    this.root.position.z = approach(this.root.position.z, state.z ?? 0, rootEase, delta);
    const targetScale = (state.scale ?? 1) * this.arrival * (1 + this.hoverAmount * 0.018);
    const scale = approach(this.root.scale.x, targetScale, reducedMotion ? 1000 : 5.4, delta);
    this.root.scale.setScalar(scale);

    const authoredMotion = state.motion ?? 1;
    const motion = reducedMotion || authoredMotion <= 0 ? 0 : Math.max(0.28, authoredMotion);
    this.content.position.x = Math.sin(time * 0.24) * 0.038 * motion;
    this.content.position.y = Math.cos(time * 0.19 + 0.7) * 0.056 * motion;
    this.content.position.z = Math.sin(time * 0.16) * 0.022 * motion;
    this.content.rotation.x = (state.rx ?? 0) + Math.cos(time * 0.13) * 0.014 * motion;
    this.content.rotation.y = (state.ry ?? 0) + Math.sin(time * 0.11) * 0.026 * motion;
    this.content.rotation.z = (state.rz ?? 0) + Math.sin(time * 0.15 + 1.1) * 0.007 * motion;

    this.updatePoints(time, reducedMotion, delta);
    this.updateAnnotations();
    this.pointsMaterial.uniforms.uOpacity.value = (
      0.82 + this.sceneWeights[2] * 0.08 + this.hoverAmount * 0.04
    ) * this.visibility;
  }

  dispose() {
    this.root.removeFromParent();
    this.disposables.forEach((resource) => resource.dispose?.());
  }
}
