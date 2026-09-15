import { mkdir, copyFile } from 'node:fs/promises';
import { build } from 'esbuild';
await mkdir('assets/vendor/three', { recursive: true });
await build({ entryPoints: ['node_modules/three/build/three.module.js'], outfile: 'assets/vendor/three/three.module.min.js', bundle: true, minify: true, format: 'esm', target: 'es2020', legalComments: 'inline' });
await copyFile('node_modules/three/LICENSE', 'assets/vendor/three/LICENSE.txt');
console.log('Vendored Three.js 0.186.0. Commit these files; Pages needs no Node build.');
