const stage=document.querySelector('[data-seagull-stage]');
if(stage&&!matchMedia('(prefers-reduced-motion: reduce)').matches){
  // HTML and the static poster paint before the heavy rendering module is fetched.
  const load=()=>import('./Experience.js').then(m=>m.startExperience(stage)).catch(()=>{
    stage.dataset.state='load-failed';document.documentElement.classList.remove('has-webgl');
    document.querySelector('[data-motion]').hidden=true;
  });
  requestAnimationFrame(()=>requestAnimationFrame(()=>{
    if('requestIdleCallback' in window)requestIdleCallback(load,{timeout:1200});else setTimeout(load,50);
  }));
}
