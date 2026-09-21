import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { UMAP } from 'umap-js';

const GENERATOR_VERSION = 'v6.0.0';
const LIBRARY_VERSION = 'umap-js@1.4.0';
const FEATURE_SEED = 609162026;
const UMAP_SEED = 609162027;
const MOBILE_POINT_COUNT = 2100;

const UMAP_PARAMETERS = Object.freeze({
  nComponents: 3,
  nEpochs: 280,
  nNeighbors: 18,
  minDist: 0.12,
  spread: 1.25,
});

// The population graph is a synthetic, deterministic semantic manifold. Nearby
// latent centers share features and bridge samples, so UMAP can recover branches,
// lobes, and detached islands rather than being handed a finished visual layout.
const POPULATIONS = Object.freeze([
  { label: 'Behavioral signals', color: '#82b7a5', count: 820, parent: null, latent: [0, 0, 0, 0], spread: [0.36, 0.28, 0.3, 0.22] },
  { label: 'Human context', color: '#d3a55b', count: 690, parent: 0, latent: [-0.9, 0.18, 0.08, 0.1], spread: [0.3, 0.24, 0.24, 0.2] },
  { label: 'Causal evidence', color: '#9b86c8', count: 610, parent: 1, latent: [-1.8, 0.42, 0.18, 0.18], spread: [0.3, 0.23, 0.22, 0.18] },
  { label: 'Experiment cohorts', color: '#6caec4', count: 540, parent: 2, latent: [-2.75, 0.75, 0.34, 0.1], spread: [0.25, 0.21, 0.2, 0.16] },
  { label: 'Model features', color: '#d8838e', count: 500, parent: 0, latent: [0.92, 0.12, 0.36, -0.16], spread: [0.31, 0.25, 0.26, 0.18] },
  { label: 'Embedding space', color: '#78c690', count: 470, parent: 4, latent: [1.82, 0.38, 0.72, -0.34], spread: [0.28, 0.23, 0.22, 0.17] },
  { label: 'Decision paths', color: '#c4a0c8', count: 430, parent: 5, latent: [2.75, 0.22, 1.02, -0.52], spread: [0.25, 0.2, 0.18, 0.15] },
  { label: 'Product systems', color: '#db765d', count: 400, parent: 0, latent: [0.28, 1.12, -0.48, 0.38], spread: [0.3, 0.24, 0.22, 0.18] },
  { label: 'Adoption loops', color: '#a0d2c2', count: 360, parent: 7, latent: [0.58, 2.02, -0.78, 0.68], spread: [0.27, 0.21, 0.2, 0.16] },
  { label: 'Learning memory', color: '#e2c978', count: 340, parent: 8, latent: [0.84, 2.9, -1.05, 0.92], spread: [0.22, 0.18, 0.18, 0.14] },
  { label: 'Measurement', color: '#6e95bd', count: 300, parent: 0, latent: [-0.18, -1.12, 0.4, 0.54], spread: [0.29, 0.22, 0.22, 0.18] },
  { label: 'Risk bounds', color: '#b8a58e', count: 250, parent: 10, latent: [-0.68, -2.02, 0.66, 0.86], spread: [0.24, 0.19, 0.19, 0.15] },
  { label: 'Prototype space', color: '#93c0a4', count: 170, parent: 4, latent: [1.18, -1.42, 1.18, -0.76], spread: [0.23, 0.19, 0.18, 0.14] },
  { label: 'Operating context', color: '#b17592', count: 120, parent: 12, latent: [2.36, -2.34, 1.5, -1.08], spread: [0.2, 0.17, 0.16, 0.13] },
]);

const TOTAL_POINT_COUNT = POPULATIONS.reduce((sum, population) => sum + population.count, 0);
if (TOTAL_POINT_COUNT !== 6000) throw new Error(`Expected 6000 points, received ${TOTAL_POINT_COUNT}.`);

function seededRandom(seed) {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let mixed = value;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

function normal(random) {
  const u = Math.max(1e-8, random());
  const v = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(Math.PI * 2 * v);
}

function featureVector(latent, populationIndex, localIndex, random) {
  const [x, y, z, w] = latent;
  const style = (populationIndex - (POPULATIONS.length - 1) * 0.5) / POPULATIONS.length;
  const cadence = localIndex / Math.max(1, POPULATIONS[populationIndex].count - 1);
  const noise = () => normal(random) * 0.035;
  return [
    x + noise(),
    y + noise(),
    z + noise(),
    w + noise(),
    Math.sin(x * 0.84) + z * 0.31 + noise(),
    Math.cos(y * 0.72) + w * 0.28 + noise(),
    x * y * 0.19 + style * 0.38 + noise(),
    z * w * 0.24 + Math.sin(y * 0.4) + noise(),
    Math.sin((x + z) * 0.48) + cadence * 0.08 + noise(),
    Math.cos((y - w) * 0.52) - cadence * 0.06 + noise(),
    (x - y + z) * 0.2 + style * style + noise(),
    (x + y - w) * 0.18 + Math.sin(style * Math.PI) * 0.24 + noise(),
  ];
}

function buildDataset() {
  const random = seededRandom(FEATURE_SEED);
  const rows = [];
  const groups = [];
  const ids = [];
  let id = 0;

  POPULATIONS.forEach((population, populationIndex) => {
    for (let localIndex = 0; localIndex < population.count; localIndex += 1) {
      const bridge = population.parent !== null && localIndex % 9 === 0;
      const parent = bridge ? POPULATIONS[population.parent] : null;
      const bridgeMix = bridge ? 0.28 + random() * 0.38 : 0;
      const latent = population.latent.map((center, dimension) => {
        const target = parent ? parent.latent[dimension] : center;
        const bridged = center * (1 - bridgeMix) + target * bridgeMix;
        return bridged + normal(random) * population.spread[dimension];
      });
      rows.push(featureVector(latent, populationIndex, localIndex, random));
      groups.push(populationIndex);
      ids.push(id);
      id += 1;
    }
  });
  return { rows, groups, ids };
}

function quantile(values, amount) {
  const sorted = [...values].sort((a, b) => a - b);
  const position = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * amount)));
  return sorted[position];
}

function normalizeEmbedding(embedding) {
  const means = [0, 1, 2].map((dimension) => (
    embedding.reduce((sum, point) => sum + point[dimension], 0) / embedding.length
  ));
  const centered = embedding.map((point) => point.map((value, dimension) => value - means[dimension]));
  const scales = [0, 1, 2].map((dimension) => Math.max(
    1e-6,
    quantile(centered.map((point) => Math.abs(point[dimension])), 0.975),
  ));
  const visualScale = [2.65, 2.0, 1.35];
  return centered.map((point) => point.map((value, dimension) => (
    Math.max(-1.18, Math.min(1.18, value / scales[dimension])) * visualScale[dimension]
  )));
}

function interleavedMobileSubset(groups) {
  const byGroup = POPULATIONS.map(() => []);
  groups.forEach((group, index) => byGroup[group].push(index));
  const quotas = POPULATIONS.map((population) => Math.floor(population.count * MOBILE_POINT_COUNT / TOTAL_POINT_COUNT));
  let assigned = quotas.reduce((sum, value) => sum + value, 0);
  for (let index = 0; assigned < MOBILE_POINT_COUNT; index = (index + 1) % quotas.length) {
    if (quotas[index] < byGroup[index].length) {
      quotas[index] += 1;
      assigned += 1;
    }
  }

  const selectedByGroup = byGroup.map((indices, group) => {
    const quota = quotas[group];
    return Array.from({ length: quota }, (_, index) => indices[Math.floor(index * indices.length / quota)]);
  });
  const selected = [];
  let cursor = 0;
  while (selected.length < MOBILE_POINT_COUNT) {
    selectedByGroup.forEach((indices) => {
      if (cursor < indices.length) selected.push(indices[cursor]);
    });
    cursor += 1;
  }
  return selected;
}

function centroidForGroup(points, groupIndex) {
  const members = points.filter((point) => point[1] === groupIndex);
  const sums = members.reduce((accumulator, point) => {
    accumulator[0] += point[2];
    accumulator[1] += point[3];
    accumulator[2] += point[4];
    return accumulator;
  }, [0, 0, 0]);
  return sums.map((value) => Number((value / members.length).toFixed(5)));
}

async function main() {
  const outputPath = resolve(process.argv[2] || fileURLToPath(new URL('../../assets/data/seagull-umap.json', import.meta.url)));
  const { rows, groups, ids } = buildDataset();
  const umap = new UMAP({
    ...UMAP_PARAMETERS,
    random: seededRandom(UMAP_SEED),
  });
  const embedding = normalizeEmbedding(umap.fit(rows));
  const basePoints = embedding.map((point, index) => [
    ids[index],
    groups[index],
    ...point.map((value) => Number(value.toFixed(5))),
  ]);

  const mobileSourceIndices = interleavedMobileSubset(groups);
  const mobileSet = new Set(mobileSourceIndices);
  const remainingSourceIndices = ids.filter((id) => !mobileSet.has(id));
  const renderOrder = [...mobileSourceIndices, ...remainingSourceIndices];
  const points = renderOrder.map((sourceIndex) => basePoints[sourceIndex]);
  const centroids = POPULATIONS.map((_, groupIndex) => centroidForGroup(points, groupIndex));
  const checksum = createHash('sha256').update(JSON.stringify(points)).digest('hex');
  const payload = {
    metadata: {
      title: 'V6 semantic population atlas',
      generatorVersion: GENERATOR_VERSION,
      library: LIBRARY_VERSION,
      featureSeed: FEATURE_SEED,
      umapSeed: UMAP_SEED,
      parameters: UMAP_PARAMETERS,
      sourceDimensions: rows[0].length,
      pointCount: TOTAL_POINT_COUNT,
      mobilePointCount: MOBILE_POINT_COUNT,
      pointFormat: ['stableId', 'populationIndex', 'x', 'y', 'z'],
      mobileSubset: 'The first 2,100 render-ordered points are a deterministic stratified subset of the full embedding.',
      checksumSha256: checksum,
    },
    populations: POPULATIONS.map(({ label, color, count, parent }) => ({ label, color, count, parent })),
    centroids,
    points,
  };

  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(payload)}\n`, 'utf8');
  process.stdout.write(`Generated ${TOTAL_POINT_COUNT} UMAP points at ${outputPath}\n`);
  process.stdout.write(`sha256(points): ${checksum}\n`);
}

await main();
