import * as THREE from '../../vendor/three/three.module.min.js';

const DESKTOP_STATES = [
  { x: 1.25, y: 0, z: -0.08, scale: 0.9, rx: -0.06, ry: 0.12, rz: -0.035, cameraX: 0, cameraY: 0, cameraZ: 8.25, assembled: 1, open: 0, operating: 0, structure: 0, prototype: 0, action: 0, blueprint: 0, resolution: 0, motion: 0.72 },
  { x: 1.35, y: 0.15, z: -0.16, scale: 0.84, rx: 0.04, ry: 0.18, rz: 0.02, cameraX: 0.02, cameraY: 0.03, cameraZ: 8.2, assembled: 0, open: 1, operating: 0, structure: 0, prototype: 0, action: 0, blueprint: 0, resolution: 0, motion: 0.48 },
  { x: 1.25, y: 0, z: -0.2, scale: 0.88, rx: 0.02, ry: -0.06, rz: -0.018, cameraX: 0, cameraY: 0.05, cameraZ: 8.16, assembled: 0, open: 0, operating: 1, structure: 0, prototype: 0, action: 0, blueprint: 0, resolution: 0, motion: 0.64 },
  { x: -1.9, y: 0.05, z: -0.24, scale: 0.68, rx: 0.05, ry: 0.16, rz: -0.035, cameraX: -0.04, cameraY: 0, cameraZ: 8.2, assembled: 0, open: 0, operating: 0, structure: 1, prototype: 0, action: 0, blueprint: 0, resolution: 0, motion: 0.46 },
  { x: 1.75, y: 0.22, z: -0.24, scale: 0.74, rx: 0.13, ry: 0.25, rz: 0.05, cameraX: 0.04, cameraY: 0.03, cameraZ: 8.15, assembled: 0, open: 0, operating: 0, structure: 0, prototype: 1, action: 0, blueprint: 0, resolution: 0, motion: 0.58 },
  { x: -2.42, y: -0.16, z: -0.38, scale: 0.68, rx: 0.08, ry: 0.16, rz: -0.045, cameraX: -0.05, cameraY: 0, cameraZ: 8.28, assembled: 0, open: 0, operating: 0, structure: 0, prototype: 0, action: 1, blueprint: 0, resolution: 0, motion: 0.46 },
  { x: 2.25, y: -2.14, z: -0.66, scale: 0.34, rx: 0.01, ry: 0.02, rz: 0, cameraX: 0, cameraY: 0.04, cameraZ: 8.52, assembled: 0, open: 0, operating: 0, structure: 0, prototype: 0, action: 0, blueprint: 1, resolution: 0, motion: 0.12 },
  { x: 1.35, y: 0.25, z: -0.22, scale: 0.66, rx: 0, ry: 0.03, rz: 0, cameraX: 0, cameraY: -0.04, cameraZ: 8.25, assembled: 0, open: 0, operating: 0, structure: 0, prototype: 0, action: 0, blueprint: 0, resolution: 1, motion: 0.08 },
];

const COMPACT_COMPOSITION = [
  { x: 0.72, y: 0.68, z: -0.48, scale: 0.78, rx: -0.06, ry: 0.1, rz: -0.06, cameraX: 0, cameraY: 0, cameraZ: 8.88 },
  { x: 0.72, y: 0.86, z: -0.52, scale: 0.74, rx: 0.04, ry: 0.16, rz: 0.02, cameraX: 0, cameraY: 0.02, cameraZ: 8.9 },
  { x: 0.7, y: 0.72, z: -0.5, scale: 0.76, rx: 0.02, ry: -0.05, rz: -0.02, cameraX: 0, cameraY: 0.02, cameraZ: 8.88 },
  { x: -0.65, y: 0.82, z: -0.68, scale: 0.7, cameraX: 0, cameraY: 0, cameraZ: 9 },
  { x: 0.72, y: 1.18, z: -0.64, scale: 0.48, cameraX: 0, cameraY: 0, cameraZ: 8.96 },
  { x: -1.02, y: 0.62, z: -0.76, scale: 0.6, cameraX: 0, cameraY: 0, cameraZ: 9.06 },
  { x: 0.78, y: -0.42, z: -0.9, scale: 0.42, cameraX: 0, cameraY: 0.04, cameraZ: 9.12 },
  { x: 0.55, y: 1.05, z: -0.62, scale: 0.62, cameraX: 0, cameraY: -0.02, cameraZ: 8.98 },
];

const MOBILE_COMPOSITION = [
  { x: 0, y: 1.32, z: -0.58, scale: 0.68, rx: -0.05, ry: 0.09, rz: -0.04, cameraX: 0, cameraY: 0, cameraZ: 8.98 },
  { x: -0.04, y: 1.85, z: -0.62, scale: 0.5, rx: 0.04, ry: 0.14, rz: 0.02, cameraX: 0, cameraY: 0.02, cameraZ: 9 },
  { x: 0, y: 1.5, z: -0.6, scale: 0.61, rx: 0.02, ry: -0.05, rz: -0.015, cameraX: 0, cameraY: 0.02, cameraZ: 8.98 },
  { x: 0, y: 1.6, z: -0.78, scale: 0.45, cameraX: 0, cameraY: 0, cameraZ: 9.04 },
  { x: 0, y: 1.95, z: -0.78, scale: 0.28, cameraX: 0, cameraY: 0, cameraZ: 9.02 },
  { x: 0.08, y: 1.82, z: -0.84, scale: 0.54, cameraX: 0, cameraY: 0, cameraZ: 9.08 },
  { x: 0.26, y: -1.62, z: -0.96, scale: 0.22, cameraX: 0, cameraY: 0.04, cameraZ: 9.14 },
  { x: 0.42, y: -2.2, z: -0.84, scale: 0.37, cameraX: 0, cameraY: -0.02, cameraZ: 9.04 },
];

const TABLET_COMPOSITION = MOBILE_COMPOSITION.map((state, index) => {
  if (index === 0) return { ...state, x: 0.02, y: 0.74, scale: 0.72, cameraZ: 8.96 };
  if (index === 4) return { ...state, x: 0.68, y: 1.55, z: -0.82, scale: 0.42, cameraZ: 9.02 };
  if (index === 5) return { ...state, x: 1.3, y: 0.2, z: -0.88, scale: 0.4, cameraZ: 9.16 };
  if (index === 6) return { ...state, x: 0.82, y: 0.42, z: -0.94, scale: 0.16, cameraZ: 9.14 };
  if (index === 7) return { ...state, x: 0.42, y: 1.82, scale: 0.52, cameraZ: 9 };
  return state;
});

const COMPACT_STATES = DESKTOP_STATES.map((state, index) => ({
  ...state,
  ...COMPACT_COMPOSITION[index],
}));

const MOBILE_STATES = DESKTOP_STATES.map((state, index) => ({
  ...state,
  ...MOBILE_COMPOSITION[index],
}));

const TABLET_STATES = DESKTOP_STATES.map((state, index) => ({
  ...state,
  ...TABLET_COMPOSITION[index],
}));

const NARROW_STATES = MOBILE_STATES.map((state, index) => {
  if (index === 4) return { ...state, x: 0, y: 2, scale: 0.27, cameraZ: 9.04 };
  if (index === 6) return { ...state, x: 0.2, y: -1.58, scale: 0.19, cameraZ: 9.16 };
  if (index === 7) return { ...state, x: 0.34, scale: 0.35 };
  return state;
});

function usesCompactComposition() {
  const aspectRatio = window.innerWidth / Math.max(1, window.innerHeight);
  return window.matchMedia('(max-width: 1100px)').matches || aspectRatio < 1.2;
}

function interpolateState(from, to, progress) {
  const output = {};
  Object.keys(from).forEach((key) => {
    output[key] = THREE.MathUtils.lerp(from[key], to[key], progress);
  });
  return output;
}

function smoothRange(value, start, end) {
  const progress = THREE.MathUtils.clamp((value - start) / Math.max(0.001, end - start), 0, 1);
  return progress * progress * (3 - 2 * progress);
}

function sceneTransitionProgress(progress) {
  // Hold the current composition while copy exits, then arrive before the next anchor.
  if (progress <= 0.45) return 0;
  if (progress >= 0.85) return 1;
  return smoothRange(progress, 0.45, 0.85);
}

function statesForViewport(compact) {
  if (!compact) return DESKTOP_STATES;
  if (window.matchMedia('(max-width: 360px) and (max-height: 780px)').matches) return NARROW_STATES;
  if (window.matchMedia('(max-width: 700px)').matches) return MOBILE_STATES;
  if (window.matchMedia('(max-width: 900px)').matches) return TABLET_STATES;
  return COMPACT_STATES;
}

function focusState(state) {
  return { ...state };
}

export class ScrollController {
  constructor({ reducedMotion = false, mobile = false, onUpdate = null } = {}) {
    this.sections = Array.from(document.querySelectorAll('[data-scene-index]'));
    this.mobile = mobile;
    this.states = statesForViewport(this.mobile);
    this.reducedMotion = reducedMotion;
    this.onUpdate = onUpdate;
    this.currentState = focusState({ ...this.states[0] });
    this.anchors = [];
    this.ticking = false;
    this.activeIndex = 0;
    this.onScroll = this.onScroll.bind(this);
    this.onResize = this.onResize.bind(this);

    this.sections[0]?.classList.add('is-active');

    this.measure();
    this.calculate();
    window.addEventListener('scroll', this.onScroll, { passive: true });
    window.addEventListener('resize', this.onResize, { passive: true });
  }

  measure() {
    const viewportHalf = window.innerHeight * 0.5;
    this.anchors = this.sections.map((section) => section.offsetTop + section.offsetHeight * 0.5 - viewportHalf);
  }

  onResize() {
    this.setMobile(usesCompactComposition());
    this.measure();
    this.onScroll();
  }

  setMobile(mobile) {
    this.mobile = mobile;
    this.states = statesForViewport(this.mobile);
  }

  onScroll() {
    if (this.ticking) return;
    this.ticking = true;
    window.requestAnimationFrame(() => {
      this.calculate();
      this.ticking = false;
    });
  }

  calculate() {
    const scrollY = window.scrollY;
    const viewportCenter = scrollY + window.innerHeight * 0.5;
    let textIndex = 0;
    this.sections.forEach((section, index) => {
      if (viewportCenter >= section.offsetTop) textIndex = index;
    });
    this.sections.forEach((section, index) => {
      section.classList.toggle('is-active', index === textIndex);
    });

    let fromIndex = 0;
    for (let index = 0; index < this.anchors.length - 1; index += 1) {
      if (scrollY >= this.anchors[index]) fromIndex = index;
    }
    fromIndex = Math.min(fromIndex, this.states.length - 1);
    const toIndex = Math.min(fromIndex + 1, this.states.length - 1);
    const start = this.anchors[fromIndex] ?? 0;
    const end = this.anchors[toIndex] ?? start + 1;
    const progress = fromIndex === toIndex ? 0 : THREE.MathUtils.clamp((scrollY - start) / Math.max(1, end - start), 0, 1);
    const transitionProgress = sceneTransitionProgress(progress);
    this.activeIndex = transitionProgress >= 0.5 ? toIndex : fromIndex;
    if (this.reducedMotion) {
      this.currentState = focusState({ ...this.states[this.activeIndex], visibility: 1 });
    } else {
      const interpolated = interpolateState(
        this.states[fromIndex], this.states[toIndex], transitionProgress,
      );
      interpolated.visibility = 1;
      this.currentState = focusState(interpolated);
    }
    this.onUpdate?.(this.currentState, this.activeIndex);
  }

  getState() {
    return this.currentState;
  }
}
