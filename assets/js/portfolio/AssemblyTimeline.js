import * as T from '../../vendor/three/three.module.min.js';
export const clamp = (x, a=0, b=1) => Math.min(b,Math.max(a,x));
export const smooth = x => { x=clamp(x); return x*x*(3-2*x); };
export const range = (p,a,b) => smooth((p-a)/(b-a));
export const stages = [
  {id:'intro',at:0,label:'Complete system'},
  {id:'about',at:.15,label:'Perception'},
  {id:'writing-1',at:.30,label:'Structure'},
  {id:'writing-2',at:.48,label:'Movement'},
  {id:'writing-3',at:.66,label:'Connection'},
  {id:'reassembly',at:.80,label:'Reassembly'},
  {id:'contact',at:.94,label:'At rest'}
];
const phases={perception:[.15,.27,.866,.923],structure:[.30,.44,.8,.856],movement:[.48,.64,.833,.90],connection:[.66,.76,.886,.928]};
export function partAmount(p,part) {
  const [start,end,backStart,backEnd]=phases[part.phase];
  const delay=part.order*.006;
  return range(p,start+delay,end+delay)*(1-range(p,backStart+delay,backEnd+delay));
}
export class AssemblyTimeline {
  constructor(parts) { this.parts=parts; this.delta=new T.Vector3(); this.rotation=new T.Quaternion(); }
  apply(progress,mobile=false) {
    const p=clamp(progress), spread=mobile?.6:1;
    for(const part of this.parts.items) {
      const t=partAmount(p,part), {node,assembled:a,exploded:b}=part;
      this.delta.copy(b.position).sub(a.position).multiplyScalar(t*spread);
      node.position.copy(a.position).add(this.delta);
      node.quaternion.slerpQuaternions(a.quaternion,b.quaternion,t*spread);
      node.scale.lerpVectors(a.scale,b.scale,t);
    }
  }
}
