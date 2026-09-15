import * as T from '../../vendor/three/three.module.min.js';
export class CableSystem {
  constructor(root,anchors,mobile){
    this.root=root; this.segments=mobile?12:24;
    this.material=new T.MeshStandardMaterial({color:0x777a76,metalness:.65,roughness:.48});
    this.cables=anchors.map(([a,b])=>{
      const mesh=new T.Mesh(new T.BufferGeometry(),this.material);root.add(mesh);
      return {a,b,mesh,lastA:new T.Vector3(Infinity,0,0),lastB:new T.Vector3(Infinity,0,0)};
    });
    this.a=new T.Vector3();this.b=new T.Vector3();
  }
  update(){
    this.root.updateWorldMatrix(true,true);
    for(const c of this.cables){
      c.a.getWorldPosition(this.a);c.b.getWorldPosition(this.b);
      this.root.worldToLocal(this.a);this.root.worldToLocal(this.b);
      if(this.a.distanceToSquared(c.lastA)+this.b.distanceToSquared(c.lastB)<.00004)continue;
      c.lastA.copy(this.a);c.lastB.copy(this.b);
      const mid=this.a.clone().lerp(this.b,.5);mid.y-=.16+this.a.distanceTo(this.b)*.12;
      const curve=new T.CatmullRomCurve3([this.a.clone(),mid,this.b.clone()]);
      c.mesh.geometry.dispose();c.mesh.geometry=new T.TubeGeometry(curve,this.segments,.013,5,false);
    }
  }
  dispose(){for(const c of this.cables){c.mesh.geometry.dispose();this.root.remove(c.mesh);}this.material.dispose();}
}
