import * as T from '../../vendor/three/three.module.min.js';
import { range } from './AssemblyTimeline.js';
// Authored compositions stay separate from geometry and lifecycle code.
const shots=[
  {p:0,position:[4.7,5.1,8.6],target:[0,0,-.25]},
  {p:.22,position:[6.8,3.5,8.4],target:[0,.1,.1]},
  {p:.38,position:[4.7,4.5,10],target:[0,.15,-.2]},
  {p:.58,position:[3.6,6.6,11.4],target:[0,.25,-.4]},
  {p:.76,position:[2.5,6.2,12],target:[0,.3,-.55]},
  {p:.94,position:[.8,4.4,10],target:[0,-.25,-.3]}
];
export class CameraRig {
  constructor(camera){this.camera=camera;this.target=new T.Vector3();this.desired=new T.Vector3();this.look=new T.Vector3();this.initial=true;}
  update(p,dt,mobile){
    let a=shots[0],b=a;
    for(let i=1;i<shots.length;i++){b=shots[i];if(p<=b.p)break;a=b;}
    const t=a===b?0:range(p,a.p,b.p);
    this.desired.fromArray(a.position).lerp(new T.Vector3(...b.position),t);
    this.look.fromArray(a.target).lerp(new T.Vector3(...b.target),t);
    if(mobile){this.desired.set(2.4,6.8,11.8);this.look.set(0,.05,-.35);}
    // Width-dependent framing prevents clipped wing tips in narrow desktop columns.
    const factor=Math.max(1,1.25/this.camera.aspect);
    this.desired.sub(this.look).multiplyScalar(factor).add(this.look);
    const alpha=this.initial?1:1-Math.exp(-5*dt);
    this.camera.position.lerp(this.desired,alpha);this.target.lerp(this.look,alpha);
    this.camera.lookAt(this.target);this.initial=false;
  }
}
