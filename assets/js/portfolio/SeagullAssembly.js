import * as T from '../../vendor/three/three.module.min.js';
import { loft, feather as makeFeather, rod, curveTube } from './Geometry.js';
import { AssemblyParts } from './AssemblyParts.js';

export function createSeagull({ mobile = false } = {}) {
  const root = new T.Group(); root.name = 'seagull';
  const parts = new AssemblyParts(root);
  const featherCache=new Map();
  const feather=(...args)=>{const key=args.join(',');if(!featherCache.has(key))featherCache.set(key,makeFeather(...args));return featherCache.get(key);};
  const white = new T.MeshStandardMaterial({ color: 0xd4d2cb, roughness: .34, metalness: .33, side: T.DoubleSide });
  const secondary = new T.MeshStandardMaterial({ color: 0xa6a9a5, roughness: .38, metalness: .6, side: T.DoubleSide });
  const titanium = new T.MeshStandardMaterial({ color: 0x929797, roughness: .26, metalness: .88 });
  const graphite = new T.MeshStandardMaterial({ color: 0x23292c, roughness: .46, metalness: .7, side: T.DoubleSide });
  const visorMaterial = new T.MeshPhysicalMaterial({ color: 0x111b20, metalness: .5, roughness: .18, clearcoat: 1, side: T.DoubleSide });
  const gold = new T.MeshStandardMaterial({ color: 0xb6a079, metalness: .75, roughness: .33, side: T.DoubleSide });
  const glow = new T.MeshStandardMaterial({ color: 0xf0d5a5, emissive: 0xd5b57f, emissiveIntensity: 2.1, roughness: .3 });
  const mesh = (geo, mat, parent) => { const m = new T.Mesh(geo, mat); parent.add(m); return m; };
  const frame = parts.add('torso-frame', root, [0,0,0], [0,0,0], [0,0,0], 'structure');
  const profile = [[-1.35,.03,.04,-.01],[-1.05,.25,.25,0],[-.5,.48,.42,0],[.05,.5,.47,0],[.6,.34,.4,.08],[1,.16,.22,.24],[1.13,.08,.1,.35]];
  // A visible rib cage, two structural rails and a narrow curved belly keel.
  for (const z of [-.9,-.5,-.1,.3,.65]) {
    const radius = .37 * (1 - Math.abs(z + .1) * .38);
    const r = mesh(new T.TorusGeometry(radius,.026,8,32), titanium, frame);
    r.position.z = z; r.scale.y = 1.12;
  }
  frame.add(curveTube([[0,-.16,-1.35],[0,-.4,-.6],[0,-.4,.3],[0,.1,1.1]],.035,graphite));
  for (const s of [-1,1]) frame.add(curveTube([[s*.12,.1,-1.2],[s*.3,.2,-.5],[s*.32,.25,.4],[s*.1,.4,1.12]],.045,titanium));
  const core = parts.add('power-core', frame, [0,-.035,.12], [0,-.04,.5], [0,0,0], 'structure');
  mesh(new T.TorusGeometry(.25,.047,12,48), graphite, core);
  mesh(new T.TorusGeometry(.194,.019,8,48), glow, core).position.z = .036;
  mesh(new T.CylinderGeometry(.19,.19,.1,40), titanium, core).rotation.x = Math.PI / 2;
  for (let i=0;i<8;i++) {
    const a=i/8*Math.PI*2;
    core.add(rod([Math.cos(a)*.19,Math.sin(a)*.19,.04],[Math.cos(a)*.09,Math.sin(a)*.09,.065],.009,gold));
  }
  const coreLight = new T.PointLight(0xffdda8,.65,2.6); core.add(coreLight);
  for (const s of [-1,1]) {
    const armor = parts.add(`${s<0?'left':'right'}-armor`, root, [0,0,0], [s*.95,.28,-.08], [0,0,-s*.23], 'structure', 2);
    const range = s > 0 ? [-Math.PI/2+.065,Math.PI/2-.065] : [Math.PI/2+.065,Math.PI*1.5-.065];
    // Longitudinal seams expose the cage; each shell is a continuous sculpted surface.
    mesh(loft(profile,{start:range[0],end:range[1],rings:40,sides:24}),white,armor);
    // Fine transverse seams follow the shell's exact elliptical cross section.
    for (const section of [profile[2],profile[3],profile[4]]) {
      const [z,rx,ry,cy]=section;
      const line=[];
      for(let j=0;j<=24;j++) {
        const a=range[0]+(range[1]-range[0])*j/24;
        line.push([(rx+.003)*Math.cos(a),(cy||0)+(ry+.003)*Math.sin(a),z]);
      }
      armor.add(curveTube(line,.005,graphite,24));
      for(const a of [range[0]+.25,range[1]-.25]) {
        const bolt=mesh(new T.CylinderGeometry(.014,.014,.008,8),titanium,armor);
        bolt.position.set((rx+.008)*Math.cos(a),(cy||0)+(ry+.008)*Math.sin(a),z+.055);
        bolt.quaternion.setFromUnitVectors(new T.Vector3(0,1,0),new T.Vector3(Math.cos(a),Math.sin(a),0));
      }
    }
    for (const z of [-.75,-.25,.3]) {
      const plate = mesh(feather(.62,.145,.015),secondary,armor);
      plate.position.set(s*.35,.29,z); plate.rotation.z=s*.45;
    }
  }
  const head = parts.add('head', root, [0,.39,1.05], [0,.52,.43], [-.08,0,0], 'perception', 1);
  mesh(loft([[-.25,.095,.11,0],[-.12,.21,.21,.1],[.12,.265,.245,.17],[.39,.2,.17,.12],[.54,.12,.075,.04]],{rings:32}),white,head);
  const neck = mesh(loft([[-.3,.105,.14,-.12],[-.1,.16,.19,0],[.1,.16,.16,.03]],{rings:16}),graphite,head);
  neck.position.z = -.15;
  const visor = parts.add('visor',head,[0,0,0],[0,.37,.05],[-.3,0,0],'perception',0);
  mesh(loft([[.16,.267,.204,.155],[.28,.241,.171,.155],[.4,.19,.115,.12],[.47,.147,.065,.11]],{start:0,end:Math.PI,rings:16,sides:24}),visorMaterial,visor);
  for (const s of [-1,1]) {
    const eye = mesh(new T.CylinderGeometry(.056,.063,.02,24),graphite,head);
    eye.rotation.z = Math.PI/2; eye.position.set(s*.237,.235,.24);
    const iris = mesh(new T.TorusGeometry(.026,.006,6,24),gold,head);
    iris.rotation.y=Math.PI/2; iris.position.set(s*.251,.235,.24);
  }
  const beak = parts.add('beak',head,[0,.045,.48],[0,-.06,.24],[.03,0,0],'perception',2);
  mesh(loft([[0,.116,.066,0],[.16,.10,.064,-.008],[.38,.066,.049,-.017],[.57,.021,.038,-.04],[.61,.003,.008,-.063]],{rings:24,sides:24}),gold,beak);
  beak.add(curveTube([[-.111,-.011,.01],[-.083,-.019,.2],[-.048,-.035,.43],[0,-.065,.6]],.006,graphite));
  const wingRootPoints = [[0,0,0],[.65,.19,-.1],[1.44,.39,-.3],[2.12,.43,-.71],[2.98,.3,-1.56]];
  for (const s of [-1,1]) {
    const side=s<0?'left':'right';
    const wing = parts.add(`${side}-wing-root`,root,[s*.36,.13,.02],[s*.65,.4,-.16],[0,-s*.05,s*.12],'movement',1);
    const wingPoints = wingRootPoints.map(p=>[p[0]*s,p[1],p[2]]);
    wing.add(curveTube(wingPoints,.065,graphite,30));
    const actuator = parts.add(`${side}-actuators`,wing,[0,0,0],[0,-.27,.28],[0,0,0],'movement',2);
    for (let i=0;i<3;i++) {
      const a=wingPoints[i], b=wingPoints[i+1];
      actuator.add(rod([a[0],a[1]-.06,a[2]],[b[0],b[1]-.06,b[2]],.038,titanium));
      const midway=a.map((v,j)=>v+(b[j]-v)*.62);
      actuator.add(rod([a[0],a[1]-.1,a[2]-.12],[midway[0],midway[1]-.1,midway[2]-.12],.068,secondary));
      actuator.add(rod([midway[0],midway[1]-.1,midway[2]-.12],[b[0],b[1]-.1,b[2]-.12],.027,titanium));
      const joint=mesh(new T.CylinderGeometry(.115,.115,.1,24),graphite,wing); joint.position.set(...b);
      const cap=mesh(new T.CylinderGeometry(.066,.066,.11,20),titanium,wing); cap.position.copy(joint.position);
    }
    const feathers=parts.add(`${side}-feather-plates`,wing,[0,0,0],[s*.18,.28,-.36],[0,s*.04,0],'movement',3);
    const count=mobile?16:23;
    for(let i=0;i<count;i++) {
      const t=i/(count-1), x=.1+t*2.93;
      const z=-.035-.12*x-.13*x*x;
      const y=.08+.28*Math.sin(t*Math.PI*.85);
      const length=1.03-.1*t + .15*Math.sin(t*Math.PI);
      const individual=parts.add(`${side}-feather-${i}`,feathers,[s*x,y,z],[s*t*.15,.06+(i%3)*.035,-t*.15],[0,s*t*.12,0],'movement',3+t);
      const blade=mesh(feather(length,.145,-s*.08,mobile?12:18),t>.82?graphite:white,individual);
      blade.rotation.y=-s*(.12+t*.53); blade.rotation.z=s*.06;
      if(!mobile) {
        const ridge=mesh(feather(length*.75,.018,-s*.04,12),secondary,individual);
        ridge.rotation.copy(blade.rotation); ridge.position.y=.025;
      }
    }
    for(let i=0;i<12;i++) {
      const t=i/11,x=.05+t*2.86,z=-.04-.12*x-.13*x*x;
      const cover=mesh(feather(.45,.19,-s*.035),white,wing);
      cover.position.set(s*x,.19+.28*Math.sin(t*Math.PI*.85),z+.12);
      cover.rotation.y=-s*(.12+t*.48);
    }
  }
  const tail=parts.add('tail',root,[0,.035,-1.07],[0,.05,-.95],[.05,0,0],'connection',1);
  for(let i=0;i<7;i++) {
    const f=mesh(feather(.98,.125,0),i===0||i===6?secondary:white,tail);
    f.position.set((i-3)*.1,Math.abs(i-3)*.012,0); f.rotation.y=(i-3)*.06; f.rotation.x=-.035;
  }
  const cableAnchors=[
    ['torso-frame',[0,.2,.55],'head',[0,0,-.22]],
    ['torso-frame',[-.28,.13,0],'left-wing-root',[-.5,.13,-.08]],
    ['torso-frame',[.28,.13,0],'right-wing-root',[.5,.13,-.08]],
    ['torso-frame',[0,0,-.8],'tail',[0,0,0]]
  ].map(([a,pa,b,pb])=>[parts.attachment(a,pa),parts.attachment(b,pb)]);
  return {root,parts,cableAnchors,coreLight,glow,materials:[white,secondary,titanium,graphite,visorMaterial,gold,glow]};
}
