import { Experience } from './Experience.js?v=v6.2-motion-r7';

const EXPECTED_POINT_COUNT = 6000;

function useStaticFallback(root, status) {
  window.clearTimeout(window.__experienceFallback);
  root.classList.remove('is-loading', 'is-webgl');
  root.classList.add('is-static', 'is-ready');
  if (status) status.hidden = true;
}

function validateAtlasData(data) {
  if (!data || typeof data !== 'object') throw new Error('UMAP atlas data is missing.');
  if (!Array.isArray(data.points) || data.points.length !== EXPECTED_POINT_COUNT) {
    throw new Error('UMAP atlas point count is invalid.');
  }
  if (!Array.isArray(data.populations) || data.populations.length < 1) {
    throw new Error('UMAP atlas populations are missing.');
  }
  if (!Array.isArray(data.centroids) || data.centroids.length !== data.populations.length) {
    throw new Error('UMAP atlas centroids are invalid.');
  }
  if (data.metadata?.pointCount !== EXPECTED_POINT_COUNT) {
    throw new Error('UMAP atlas metadata is invalid.');
  }
  for (let index = 0; index < data.points.length; index += 1) {
    const point = data.points[index];
    if (!Array.isArray(point) || point.length < 5
      || !Number.isInteger(point[0]) || !Number.isInteger(point[1])
      || point[1] < 0 || point[1] >= data.populations.length
      || !Number.isFinite(point[2]) || !Number.isFinite(point[3]) || !Number.isFinite(point[4])) {
      throw new Error(`UMAP atlas point ${index} is invalid.`);
    }
  }
  return data;
}

export async function boot() {
  const root = document.documentElement;
  const status = document.getElementById('load-status');
  const canvas = document.getElementById('experience-canvas');
  const mobile = window.matchMedia('(max-width: 900px)').matches;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const contextAttributes = {
    alpha: true,
    antialias: !mobile,
    preserveDrawingBuffer: reducedMotion,
    powerPreference: 'high-performance',
    failIfMajorPerformanceCaveat: true,
  };
  let context = null;

  try {
    context = canvas?.getContext('webgl2', contextAttributes) || null;
  } catch (_error) {
    context = null;
  }

  if (!context) {
    useStaticFallback(root, status);
    return;
  }

  let atlasData;
  try {
    const source = canvas?.dataset.umapSource || '/assets/data/seagull-umap.json?v=v6.2-motion-r7';
    const response = await fetch(source, { cache: 'force-cache', credentials: 'same-origin' });
    if (!response.ok) throw new Error(`UMAP atlas request failed with ${response.status}.`);
    atlasData = validateAtlasData(await response.json());
  } catch (error) {
    console.warn('V6 UMAP sculpture unavailable; using the static fallback.', error);
    useStaticFallback(root, status);
    return;
  }

  let experience;
  try {
    experience = new Experience(canvas, context, atlasData);
  } catch (error) {
    console.warn('V6 UMAP sculpture could not initialize; using the static fallback.', error);
    useStaticFallback(root, status);
    return;
  }

  document.querySelectorAll('.project-reactive').forEach((link) => {
    link.addEventListener('pointerenter', () => experience.setProjectFocus(true));
    link.addEventListener('pointerleave', () => experience.setProjectFocus(false));
    link.addEventListener('focus', () => experience.setProjectFocus(true));
    link.addEventListener('blur', () => experience.setProjectFocus(false));
  });

  root.classList.add('is-webgl');
  experience.start(() => {
    window.clearTimeout(window.__experienceFallback);
    root.classList.remove('is-loading', 'is-static');
    root.classList.add('is-ready');
    if (status) status.hidden = true;
  });
}
