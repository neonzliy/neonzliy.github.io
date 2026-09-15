import {chromium} from '@playwright/test';
import {mkdir} from 'node:fs/promises';
const destination=process.env.SCREENSHOT_DIR||'test-results/model';await mkdir(destination,{recursive:true});
const browser=await chromium.launch({channel:process.env.PW_CHANNEL||undefined});
const page=await browser.newPage({viewport:{width:1440,height:900}});
for(const view of ['front','side','three']){
  await page.goto(`http://127.0.0.1:4173/scripts/model-review.html?view=${view}`);
  await page.waitForFunction(()=>window.review);await page.screenshot({path:`${destination}/model-${view}.png`});
}
await page.goto('http://127.0.0.1:4173/scripts/model-review.html?view=three&poster');await page.waitForFunction(()=>window.review);
await page.screenshot({path:`${destination}/poster.png`});
await browser.close();
