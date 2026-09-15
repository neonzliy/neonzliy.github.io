import * as T from '../../vendor/three/three.module.min.js';
import { range } from './AssemblyTimeline.js';
// Authored compositions stay separate from geometry and lifecycle code.
const shots=[
  {p:0,position:[4.2,3.1,6.3],target:[0,1.65,0]},
  {p:.22,position:[4.6,3.25,6.8],target:[0,1.85,.1]},
  {p:.38,position:[3.5,3.6,7.5],target:[0,1.7,0]},
  {p:.58,position:[4.7,3.5,8.0],target:[0,1.7,-.1]},
  {p:.76,position:[3.5,3.7,8.5],target:[0,1.75,-.1]},
  {p:.94,position:[4.2,3.1,6.8],target:[.35,1.65,0]}
];
export class CameraRig {
  constructor(camera){this.camera=camera;this.target=new T.Vector3();this.desired=new T.Vector3();this.look=new T.Vector3();this.initial=true;}
  update(p,dt,mobile){
    let a=shots[0],b=a;
    for(let i=1;i<shots.length;i++){b=shots[i];if(p<=b.p)break;a=b;}
    const t=a===b?0:range(p,a.p,b.p);
    this.desired.fromArray(a.position).lerp(new T.Vector3(...b.position),t);
    this.look.fromArray(a.target).lerp(new T.Vector3(...b.target),t);
    if(mobile){this.desired.set(3.4,2.8,6.8);this.look.set(0,1.65,0);}
    // Keep the standing sculpture and separated armor inside narrow viewports.
    const factor=Math.max(1,.88/this.camera.aspect);
    this.desired.sub(this.look).multiplyScalar(factor).add(this.look);
    const alpha=this.initial?1:1-Math.exp(-5*dt);
    this.camera.position.lerp(this.desired,alpha);this.target.lerp(this.look,alpha);
    this.camera.lookAt(this.target);this.initial=false;
  }
}
