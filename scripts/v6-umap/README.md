# V6 UMAP atlas generator

This generator creates the deterministic 6,000-point, three-dimensional UMAP
embedding used by the V6 seagull sculpture. The browser loads the precomputed
result and never performs dimensionality reduction at runtime.

```sh
npm ci
npm run generate
```

The output records the library version, parameters, seeds, stable point IDs,
population labels, mobile subset rule, and checksum needed to reproduce and
verify the atlas.
