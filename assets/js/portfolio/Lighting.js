import * as T from '../../vendor/three/three.module.min.js';
export function createLighting(scene, renderer) {
  scene.add(new T.HemisphereLight(0xe8edf0,0x35302a,.8));
  const key = new T.DirectionalLight(0xfff0e1,2.3); key.position.set(2,6,5); scene.add(key);
  const rim = new T.DirectionalLight(0xc4d6ef,3.1); rim.position.set(-5,3,-4); scene.add(rim);
  const fill = new T.DirectionalLight(0xf6e3c7,.5); fill.position.set(3,-2,2); scene.add(fill);
  // A tiny procedural studio environment gives metal a broad, controlled reflection.
  const studio=new T.Scene(); studio.background=new T.Color(0x32383d);
  for(const [pos,scale,color,intensity] of [
    [[0,5,0],[8,1,6],0xffffff,3], [[-5,1,0],[1,5,8],0xb2c2d6,2], [[3,0,4],[3,5,1],0xf4e2c4,1.8]
  ]) {
    const card=new T.Mesh(new T.BoxGeometry(...scale),new T.MeshBasicMaterial({color, toneMapped:false}));
    card.material.color.multiplyScalar(intensity); card.position.set(...pos); studio.add(card);
  }
  const pmrem=new T.PMREMGenerator(renderer); const env=pmrem.fromScene(studio,.04); scene.environment=env.texture;
  studio.traverse(o=>{o.geometry?.dispose();o.material?.dispose();}); pmrem.dispose();
  // A low machined plinth anchors the planted feet without realtime shadows.
  const plinth=new T.Group();scene.add(plinth);
  const baseMaterial=new T.MeshStandardMaterial({color:0x171e23,metalness:.82,roughness:.36});
  const rimMaterial=new T.MeshStandardMaterial({color:0x666c6d,metalness:.9,roughness:.3});
  const base=new T.Mesh(new T.CylinderGeometry(.71,.75,.035,80),baseMaterial);base.position.set(0,.003,.04);plinth.add(base);
  for(const radius of [.69,.73]){
    const line=new T.Mesh(new T.TorusGeometry(radius,.003,6,96),rimMaterial);line.rotation.x=Math.PI/2;line.position.set(0,.022,.04);plinth.add(line);
  }
  return ()=>{env.dispose();scene.environment=null;plinth.traverse(o=>o.geometry?.dispose());baseMaterial.dispose();rimMaterial.dispose();scene.remove(plinth);};
}
