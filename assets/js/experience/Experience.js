import * as THREE from '../../vendor/three/three.module.min.js';
import { UMAPSeagullEngine } from './UMAPSeagullEngine.js?v=v6.2-motion-r7';
import { ScrollController } from './ScrollController.js?v=v6.2-motion-r7';

function usesCompactComposition() {
  const aspectRatio = window.innerWidth / Math.max(1, window.innerHeight);
  return window.matchMedia('(max-width: 1100px)').matches || aspectRatio < 1.2;
}

export class Experience {
  constructor(canvas, context, atlasData) {
    if (!canvas) throw new Error('Experience canvas is missing.');
    if (!context) throw new Error('WebGL2 is unavailable.');

    this.canvas = canvas;
    this.mobile = window.matchMedia('(max-width: 900px)').matches;
    this.compact = usesCompactComposition();
    this.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.pointer = new THREE.Vector2();
    this.pointerTarget = new THREE.Vector2();
    this.lastFrameTime = 0;
    this.elapsedTime = 0;
    this.frame = null;
    this.running = false;
    this.ready = false;

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      context,
      alpha: true,
      antialias: !this.mobile,
      preserveDrawingBuffer: this.reducedMotion,
      powerPreference: 'high-performance',
      failIfMajorPerformanceCaveat: true,
    });
    this.renderer.setClearColor(0x05070a, 0);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.02;
    this.renderer.shadowMap.enabled = false;

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x05070a, 0.04);
    this.camera = new THREE.PerspectiveCamera(36, 1, 0.1, 40);
    this.camera.position.set(0, 0, 8.5);

    this.sculpture = new UMAPSeagullEngine({ data: atlasData, mobile: this.mobile });
    this.scene.add(this.sculpture.root);
    this.scroll = new ScrollController({
      mobile: this.compact,
      reducedMotion: this.reducedMotion,
      onUpdate: () => {
        if (this.ready && this.reducedMotion) this.renderStaticFrame();
      },
    });

    const hemisphere = new THREE.HemisphereLight(0xe3eee9, 0x071019, 0.72);
    const key = new THREE.DirectionalLight(0xe7eee9, 1.2);
    key.position.set(4, 5.5, 7);
    const rim = new THREE.SpotLight(0x96cfc2, 4.5, 22, Math.PI * 0.22, 0.72, 1.25);
    rim.position.set(-5, 1.5, 5);
    rim.target = this.sculpture.root;
    const signal = new THREE.PointLight(0xd6aa67, 0, 10, 2);
    signal.position.set(2.5, -1.8, 3.5);
    this.rimLight = rim;
    this.signalLight = signal;
    this.scene.add(hemisphere, key, rim, signal);

    this.onPointerMove = this.onPointerMove.bind(this);
    this.onResize = this.onResize.bind(this);
    this.onVisibility = this.onVisibility.bind(this);
    this.tick = this.tick.bind(this);
    window.addEventListener('pointermove', this.onPointerMove, { passive: true });
    window.addEventListener('resize', this.onResize, { passive: true });
    document.addEventListener('visibilitychange', this.onVisibility);
    this.onResize();
  }

  onPointerMove(event) {
    if (this.reducedMotion) return;
    this.pointerTarget.set(
      (event.clientX / window.innerWidth) * 2 - 1,
      -((event.clientY / window.innerHeight) * 2 - 1),
    );
  }

  onResize() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    const mobile = window.matchMedia('(max-width: 900px)').matches;
    const compact = usesCompactComposition();
    if (compact !== this.compact) {
      this.compact = compact;
      this.scroll?.setMobile(compact);
    }
    if (mobile !== this.mobile) {
      this.mobile = mobile;
      this.sculpture?.setMobile(mobile);
    }
    this.camera.aspect = width / Math.max(1, height);
    this.camera.updateProjectionMatrix();
    const maxDpr = this.mobile ? 1.25 : 1.5;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, maxDpr));
    this.renderer.setSize(width, height, false);
    if (this.ready && this.reducedMotion) this.renderStaticFrame();
  }

  setProjectFocus(active) {
    this.sculpture.setProjectFocus(active && !this.reducedMotion);
  }

  onVisibility() {
    if (document.hidden) {
      this.stop();
    } else if (!this.reducedMotion) {
      this.startLoop();
    }
  }

  start(onFirstFrame) {
    this.renderStaticFrame();
    onFirstFrame();
    this.ready = true;

    if (this.reducedMotion) {
      window.setTimeout(() => {
        window.requestAnimationFrame(() => {
          this.scroll.measure();
          this.scroll.calculate();
          this.renderStaticFrame();
        });
      }, 160);
      return;
    }

    this.startLoop();
  }

  renderStaticFrame() {
    const state = this.scroll.getState();
    this.sculpture.update(0, this.pointer, state, this.reducedMotion);
    this.updateLights(state);
    this.camera.position.set(state.cameraX, state.cameraY, state.cameraZ);
    this.camera.lookAt(0, 0, 0);
    this.renderer.render(this.scene, this.camera);
  }

  updateLights(state) {
    const activeSignal = (state.operating ?? 0) * 0.18
      + (state.prototype ?? 0) * 0.28
      + (state.action ?? 0) * 0.32;
    const reveal = (state.structure ?? 0) + (state.resolution ?? 0) * 0.6;
    this.signalLight.intensity = 2.4 * activeSignal;
    this.rimLight.intensity = 3.8 + reveal * 1.4;
  }

  startLoop() {
    if (this.running) return;
    this.running = true;
    this.lastFrameTime = performance.now();
    this.frame = window.requestAnimationFrame(this.tick);
  }

  stop() {
    this.running = false;
    if (this.frame !== null) {
      window.cancelAnimationFrame(this.frame);
      this.frame = null;
    }
  }

  tick(timestamp) {
    if (!this.running) return;
    const delta = Math.min(Math.max(0, timestamp - this.lastFrameTime) / 1000, 0.05);
    this.lastFrameTime = timestamp;
    this.elapsedTime += delta;
    const time = this.elapsedTime;
    const pointerEase = 1 - Math.pow(0.002, delta);
    this.pointer.lerp(this.pointerTarget, pointerEase);

    const state = this.scroll.getState();
    this.sculpture.update(time, this.pointer, state, false);
    this.updateLights(state);
    this.camera.position.x = THREE.MathUtils.lerp(this.camera.position.x, state.cameraX + this.pointer.x * 0.012, 0.04);
    this.camera.position.y = THREE.MathUtils.lerp(this.camera.position.y, state.cameraY + this.pointer.y * 0.01, 0.04);
    this.camera.position.z = THREE.MathUtils.lerp(this.camera.position.z, state.cameraZ, 0.04);
    this.camera.lookAt(0, 0, 0);
    this.renderer.render(this.scene, this.camera);
    this.frame = window.requestAnimationFrame(this.tick);
  }
}
