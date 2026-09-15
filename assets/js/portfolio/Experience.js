import * as T from '../../vendor/three/three.module.min.js';
import {createSeagull} from './SeagullAssembly.js';
import {createLighting} from './Lighting.js';
import {AssemblyTimeline,range,stages} from './AssemblyTimeline.js';
import {CameraRig} from './CameraRig.js';
import {CableSystem} from './CableSystem.js';
import {ScrollController} from './ScrollController.js';
import {getQuality} from './QualityController.js';
import {showFallback} from './StaticFallback.js';

export function startExperience(stage){
  let quality=getQuality();
  if(quality.reduced){showFallback(stage,'reduced-motion');return;}
  const abort=new AbortController(),options={signal:abort.signal};
  const canvas=stage.querySelector('canvas');
  let renderer;
  try {renderer=new T.WebGLRenderer({canvas,antialias:!quality.mobile,alpha:true,powerPreference:'low-power'});}
  catch {showFallback(stage,'unavailable');return;}
  renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=.88;
  const scene=new T.Scene(),camera=new T.PerspectiveCamera(33,1,.1,100);
  const disposeLight=createLighting(scene,renderer);
  const bird=createSeagull(quality);scene.add(bird.root);
  const timeline=new AssemblyTimeline(bird.parts),rig=new CameraRig(camera);
  const cables=new CableSystem(bird.root,bird.cableAnchors,quality.mobile);
  let progress=0,paused=false,disposed=false,raf=0,last=0,elapsed=0,hover=0,targetHover=0;
  const pointer=new T.Vector2(),targetPointer=new T.Vector2();
  const button=document.querySelector('[data-motion]');
  const label=document.querySelector('[data-phase]'),count=document.querySelector('[data-count]');
  const nav=[...document.querySelectorAll('[data-stage-link]')];
  const chapters=stages.map(s=>document.getElementById(s.id));
  function resize(){
    quality=getQuality(); const rect=stage.getBoundingClientRect();
    renderer.setPixelRatio(quality.dpr);renderer.setSize(rect.width,rect.height,false);
    camera.aspect=rect.width/Math.max(1,rect.height);camera.updateProjectionMatrix();rig.initial=true;
    if(paused)renderer.render(scene,camera);
  }
  const observer=new ResizeObserver(resize);observer.observe(stage);resize();
  const scroll=new ScrollController(p=>{
    progress=p;
    const index=stages.reduce((n,s,i)=>p>=s.at-.005?i:n,0);
    label.textContent=stages[index].label;count.textContent=String(index+1).padStart(2,'0');
    nav.forEach((a,i)=>i===index?a.setAttribute('aria-current','step'):a.removeAttribute('aria-current'));
    chapters.forEach((chapter,i)=>chapter?.classList.toggle('is-active',i===index));
    document.documentElement.style.setProperty('--journey-progress',p);
  });
  function frame(time){
    raf=0;if(disposed||document.hidden||paused)return;
    const dt=Math.min(.05,(time-(last||time))/1000);last=time;elapsed+=dt;
    const idle=1-range(progress,.8,.94);
    pointer.lerp(targetPointer,1-Math.exp(-3.5*dt));hover+=(targetHover-hover)*(1-Math.exp(-5*dt));
    timeline.apply(progress,quality.mobile);rig.update(progress,dt,quality.mobile);
    bird.root.rotation.set(.02+pointer.y*.035*idle,Math.sin(elapsed*.16)*.018*idle+pointer.x*.055*idle,-.075+Math.sin(elapsed*.23)*.012*idle);
    bird.root.position.y=Math.sin(elapsed*.62)*.025*idle;
    bird.glow.emissiveIntensity=1.8+hover*.9;bird.coreLight.intensity=.5+hover*.4;
    cables.update();
    stage.style.setProperty('--mark-opacity',range(progress,.80,.89));
    renderer.render(scene,camera);stage.dataset.state='ready';
    stage.dataset.progress=progress.toFixed(4);
    raf=requestAnimationFrame(frame);
  }
  function resume(){if(!raf&&!paused&&!disposed&&!document.hidden){last=0;raf=requestAnimationFrame(frame);}}
  button.hidden=false;
  button.addEventListener('click',()=>{
    paused=!paused;button.setAttribute('aria-pressed',String(paused));button.textContent=paused?'Resume motion':'Pause motion';
    stage.dataset.motion=paused?'paused':'playing';
    if(paused){cancelAnimationFrame(raf);raf=0;}else resume();
  },options);
  addEventListener('pointermove',e=>{if(!quality.mobile)targetPointer.set(e.clientX/innerWidth-.5,e.clientY/innerHeight-.5);},{...options,passive:true});
  document.addEventListener('pointerleave',()=>targetPointer.set(0,0),options);
  for(const link of document.querySelectorAll('[data-project-link]')){
    link.addEventListener('pointerenter',()=>targetHover=1,options);link.addEventListener('pointerleave',()=>targetHover=0,options);
    link.addEventListener('focus',()=>targetHover=1,options);link.addEventListener('blur',()=>targetHover=0,options);
  }
  for(const chapter of chapters)chapter?.addEventListener('focusin',()=>{
    // Keyboard traversal follows the HTML document order, including inactive scenes.
    if(!chapter.classList.contains('is-active'))chapter.scrollIntoView({block:'start',behavior:'instant'});
  },options);
  document.addEventListener('visibilitychange',()=>{if(document.hidden){cancelAnimationFrame(raf);raf=0;}else resume();},options);
  function dispose(){
    if(disposed)return;disposed=true;cancelAnimationFrame(raf);observer.disconnect();scroll.dispose();abort.abort();
    cables.dispose();const geometries=new Set();bird.root.traverse(o=>{if(o.geometry)geometries.add(o.geometry);});
    geometries.forEach(g=>g.dispose());bird.materials.forEach(m=>m.dispose());disposeLight();renderer.dispose();
  }
  canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();dispose();showFallback(stage,'context-lost');},options);
  const media=matchMedia('(prefers-reduced-motion: reduce)');
  media.addEventListener('change',e=>{if(e.matches){dispose();showFallback(stage,'reduced-motion');}},{signal:abort.signal});
  addEventListener('pagehide',e=>{if(!e.persisted)dispose();},options);
  document.documentElement.classList.add('has-webgl');
  resume();return {dispose};
}
