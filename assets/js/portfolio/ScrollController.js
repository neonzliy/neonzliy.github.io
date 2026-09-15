import {clamp,stages} from './AssemblyTimeline.js';
export class ScrollController {
  constructor(onChange){
    this.onChange=onChange;this.progress=0;this.points=[];this.abort=new AbortController();
    this.measure=this.measure.bind(this);this.update=this.update.bind(this);
    const options={passive:true,signal:this.abort.signal};
    addEventListener('scroll',this.update,options);addEventListener('resize',this.measure,options);
    addEventListener('pageshow',this.measure,options);addEventListener('hashchange',this.update,options);
    this.observer=new ResizeObserver(this.measure);this.observer.observe(document.querySelector('main'));
    this.measure();
  }
  measure(){
    const max=Math.max(1,document.documentElement.scrollHeight-innerHeight);
    this.points=stages.map(s=>({p:s.at,y:clamp((document.getElementById(s.id)?.offsetTop||0)-innerHeight*.14,0,max)}));
    this.points.push({p:1,y:max});this.update();
  }
  update(){
    const y=scrollY;let a=this.points[0],b=a;
    for(let i=1;i<this.points.length;i++){b=this.points[i];if(y<=b.y)break;a=b;}
    this.progress=a===b?a.p:a.p+(b.p-a.p)*clamp((y-a.y)/Math.max(1,b.y-a.y));
    this.onChange(this.progress);
  }
  dispose(){this.abort.abort();this.observer.disconnect();}
}
