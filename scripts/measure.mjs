import {chromium} from '@playwright/test';
import {createSeagull} from '../assets/js/portfolio/SeagullAssembly.js';
import {readFile,writeFile} from 'node:fs/promises';
import {gzipSync} from 'node:zlib';
const browser=await chromium.launch({channel:process.env.PW_CHANNEL||undefined});
const measurements=[];
for(const [width,height] of [[1440,900],[390,844]]){
  const page=await browser.newPage({viewport:{width,height}});await page.goto('http://127.0.0.1:4174/');
  await page.waitForFunction(()=>document.querySelector('[data-seagull-stage]')?.dataset.state==='ready');
  const frameTimes=await page.evaluate(()=>new Promise(resolve=>{
    const samples=[];let last=0,index=0;
    const max=document.documentElement.scrollHeight-innerHeight;
    function frame(now){if(last)samples.push(now-last);last=now;scrollTo(0,max*(index/180));if(index++<180)requestAnimationFrame(frame);else resolve(samples);}
    requestAnimationFrame(frame);
  }));
  const sorted=[...frameTimes].sort((a,b)=>a-b),avg=frameTimes.reduce((a,b)=>a+b,0)/frameTimes.length;
  const bird=createSeagull({mobile:width<760});let triangles=0,meshes=0;
  bird.root.traverse(o=>{if(o.isMesh){meshes++;triangles+=(o.geometry.index?.count||o.geometry.attributes.position.count)/3;}});
  measurements.push({viewport:{width,height},samples:frameTimes.length,averageFrameMs:+avg.toFixed(2),p95FrameMs:+sorted[Math.floor(sorted.length*.95)].toFixed(2),meanFrameRate:+(1000/avg).toFixed(1),proceduralMeshes:meshes,trianglesExcludingCables:triangles});
  await page.close();
}
await browser.close();
const vendor=await readFile('assets/vendor/three/three.module.min.js');
const report={environment:'Headless Chrome on the development Mac; mobile is viewport emulation, not a physical device.',threeBytes:vendor.length,threeGzipBytes:gzipSync(vendor).length,posterBytes:(await readFile('assets/img/seagull-assembled.webp')).length,measurements};
console.log(JSON.stringify(report,null,2));
if(process.env.METRICS_PATH)await writeFile(process.env.METRICS_PATH,JSON.stringify(report,null,2)+'\n');
