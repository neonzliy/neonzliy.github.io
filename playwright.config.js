import {defineConfig} from '@playwright/test';
export default defineConfig({
  testDir:'tests',testMatch:'**/*.spec.js',fullyParallel:false,workers:1,
  timeout:30000,reporter:[['list'],['json',{outputFile:'test-results/results.json'}]],
  use:{baseURL:'http://127.0.0.1:4174',headless:true,channel:process.env.PW_CHANNEL||undefined,viewport:{width:1440,height:900},screenshot:'only-on-failure'},
  webServer:{command:'PORT=4174 npm run preview',url:'http://127.0.0.1:4174',reuseExistingServer:!process.env.CI}
});
