import {test,expect} from '@playwright/test';
import fs from 'node:fs/promises';
const shots=process.env.SCREENSHOT_DIR||'test-results/screenshots';
const stage=page=>page.locator('[data-seagull-stage]');
async function ready(page){await page.goto('/');await expect(stage(page)).toHaveAttribute('data-state','ready');}
async function progress(page,p){
  await page.evaluate(p=>{
    const phases=[0,.15,.3,.48,.66,.8,.94,1];
    const max=document.documentElement.scrollHeight-innerHeight;
    const points=[...document.querySelectorAll('.chapter')].map(e=>Math.max(0,Math.min(max,e.offsetTop-innerHeight*.14)));points.push(max);
    let i=0;while(i<phases.length-2&&p>phases[i+1])i++;
    scrollTo(0,points[i]+(points[i+1]-points[i])*(p-phases[i])/(phases[i+1]-phases[i]));
  },p);
  await expect.poll(async()=>Number(await stage(page).getAttribute('data-progress'))).toBeCloseTo(p,2);
  await page.waitForTimeout(650); // Allow the camera damping and poster fade to settle before capture.
}
test('desktop scenes, reverse and fast scroll, anchors, pause and keyboard interaction',async({page})=>{
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  const failed=[];page.on('response',r=>{if(r.status()>=400)failed.push(r.url());});
  await ready(page);await fs.mkdir(shots,{recursive:true});
  for(const [p,name] of [[0,'01-introduction'],[.23,'02-perception'],[.40,'03-structure'],[.60,'04-wing-mechanisms'],[.76,'05-exploded'],[.96,'06-reassembled-lz']]){
    await progress(page,p);await page.screenshot({path:`${shots}/${name}.png`});
  }
  await progress(page,.25);await progress(page,.77);await progress(page,0);
  await page.getByRole('button',{name:'Pause motion'}).click();
  const frozen=await stage(page).getAttribute('data-progress');
  await page.evaluate(()=>scrollTo(0,document.body.scrollHeight));
  await page.waitForTimeout(150);await expect(stage(page)).toHaveAttribute('data-progress',frozen);
  await page.getByRole('button',{name:'Resume motion'}).click();await expect.poll(async()=>Number(await stage(page).getAttribute('data-progress'))).toBeGreaterThan(.94);
  await page.getByRole('link',{name:'First article',exact:true}).click();
  const article=page.locator('#writing-1 h2 a');await article.focus();await expect(article).toBeFocused();await article.hover();
  await article.press('Enter');await expect(page).toHaveURL(/reducing-hallucinations/);
  await page.goBack();await expect(page).toHaveURL(/#writing-1$/);await expect(stage(page)).toHaveAttribute('data-state','ready');
  await page.goForward();await expect(page).toHaveURL(/reducing-hallucinations/);
  expect(errors).toEqual([]);expect(failed).toEqual([]);
});
test('desktop and landscape framing stay within a dedicated text-free stage',async({page})=>{
  for(const [width,height,name] of [[1920,1080,'07-desktop-1920'],[1024,600,'08-short-landscape']]){
    await page.setViewportSize({width,height});await ready(page);await progress(page,.78);
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
    const image=await stage(page).boundingBox(),text=await page.locator('#writing-3 .chapter-copy').boundingBox();
    expect(text.x+text.width).toBeLessThanOrEqual(image.x+1);
    expect(text.y).toBeGreaterThan(80);expect(text.y+text.height).toBeLessThan(height-60);
    await page.screenshot({path:`${shots}/${name}.png`});
  }
});
test('mobile introduction and writing remain readable without horizontal overflow',async({page})=>{
  await page.setViewportSize({width:390,height:844});await ready(page);
  await progress(page,0);await page.screenshot({path:`${shots}/09-mobile-introduction.png`});
  const intro=await page.locator('#intro .chapter-copy').boundingBox(),image=await stage(page).boundingBox();expect(intro.y).toBeGreaterThan(image.y+image.height);
  await page.getByRole('link',{name:'Second article',exact:true}).click();await page.waitForTimeout(650);await page.screenshot({path:`${shots}/10-mobile-writing.png`});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await expect(page.getByRole('button',{name:'Pause motion'})).toBeVisible();
});
test('reduced motion serves the static scene without downloading Three.js',async({page})=>{
  await page.emulateMedia({reducedMotion:'reduce'});const loaded=[];page.on('request',r=>loaded.push(r.url()));
  await page.goto('/');await expect(page.locator('h1')).toHaveText('Leon Zhao');
  await expect(page.locator('.seagull-poster')).toBeVisible();await expect(page.locator('[data-motion]')).toBeHidden();
  expect(loaded.some(u=>u.includes('three.module')||u.includes('Experience.js'))).toBe(false);
  await page.screenshot({path:`${shots}/11-reduced-motion.png`});
});
test('unavailable WebGL, lost context and module failure leave usable content',async({page})=>{
  await page.addInitScript(()=>{const original=HTMLCanvasElement.prototype.getContext;HTMLCanvasElement.prototype.getContext=function(type,...rest){return type.includes('webgl')?null:original.call(this,type,...rest);};});
  await page.goto('/');await expect(stage(page)).toHaveAttribute('data-state','unavailable');
  await expect(page.locator('#writing-1 h2 a')).toHaveAttribute('href',/reducing-hallucinations/);
  await expect(page.locator('.seagull-poster')).toBeVisible();
});
test('a lost WebGL context returns to the poster',async({page})=>{
  await ready(page);
  await page.evaluate(()=>document.querySelector('canvas').getContext('webgl2').getExtension('WEBGL_lose_context').loseContext());
  await expect(stage(page)).toHaveAttribute('data-state','context-lost');await expect(page.locator('[data-motion]')).toBeHidden();await expect(page.locator('.seagull-poster')).toBeVisible();
});
test('a failed model module and disabled JavaScript preserve all articles and navigation',async({page,browser})=>{
  await page.route('**/SeagullAssembly.js',route=>route.abort());await page.goto('/');
  await expect(stage(page)).toHaveAttribute('data-state','load-failed');await expect(page.locator('.seagull-poster')).toBeVisible();
  const context=await browser.newContext({javaScriptEnabled:false});const staticPage=await context.newPage();await staticPage.goto('http://127.0.0.1:4174/');
  await expect(staticPage.locator('.chapter-writing')).toHaveCount(3);await expect(staticPage.locator('.original-bio')).toContainText('Leon is a builder.');
  await staticPage.getByRole('link',{name:'Writing',exact:true}).click();await expect(staticPage).toHaveURL(/writings/);await context.close();
});
test('existing content routes and feed are served with their original canonical URLs',async({request})=>{
  for(const path of ['/me','/writings/','/2023-05-01-sentiment_classification/','/2025-01-15-measuring-ai-impact-without-ab-tests/','/2025-03-01-reducing-hallucinations-enterprise-llms/','/feed.xml']){
    const response=await request.get(path);expect(response.ok(),path).toBe(true);
    if(path!='/feed.xml')expect(await response.text()).toContain('https://leonz.site');
  }
});
