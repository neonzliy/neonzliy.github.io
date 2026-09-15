import * as T from '../../vendor/three/three.module.min.js';
import {loft,verticalLoft,thicken,feather as makeFeather,rod,curveTube} from './Geometry.js';
import {AssemblyParts} from './AssemblyParts.js';

export function createSeagull({mobile=false}={}) {
  const root=new T.Group();root.name='seagull';
  const parts=new AssemblyParts(root),cache=new Map();
  const feather=(...args)=>{const key=args.join(',');if(!cache.has(key))cache.set(key,makeFeather(...args));return cache.get(key);};
  const pearl=new T.MeshPhysicalMaterial({color:0xd8d8d0,metalness:.36,roughness:.3,clearcoat:.5,clearcoatRoughness:.25,side:T.DoubleSide});
  const pale=new T.MeshStandardMaterial({color:0xa7aaa6,metalness:.72,roughness:.31,side:T.DoubleSide});
  const titanium=new T.MeshStandardMaterial({color:0x8c9293,metalness:.92,roughness:.27});
  const graphite=new T.MeshStandardMaterial({color:0x262c2e,metalness:.75,roughness:.37,side:T.DoubleSide});
  const dark=new T.MeshPhysicalMaterial({color:0x101a20,metalness:.72,roughness:.18,clearcoat:.85,side:T.DoubleSide});
  const brass=new T.MeshStandardMaterial({color:0x9e8a63,metalness:.85,roughness:.29,side:T.DoubleSide});
  const glow=new T.MeshStandardMaterial({color:0xf4dfb5,emissive:0xe2ba73,emissiveIntensity:1.8,roughness:.3});
  const materials=[pearl,pale,titanium,graphite,dark,brass,glow];
  const mesh=(g,m,parent,position=[0,0,0])=>{const node=new T.Mesh(g,m);node.position.set(...position);parent.add(node);return node;};
  const panel=(profile,angles,parent,material=pearl)=>mesh(thicken(verticalLoft(profile,{start:angles[0],end:angles[1],rings:mobile?18:28,sides:mobile?16:24}),.018),material,parent);
  const ring=(r,t,parent,position,material=titanium)=>mesh(new T.TorusGeometry(r,t,8,mobile?28:44),material,parent,position);
  const blade=(parent,a,b,width,material=pearl,normal=[1,0,0],bend=.035)=>{
    const from=new T.Vector3(...a),to=new T.Vector3(...b),z=from.clone().sub(to).normalize();
    const x=new T.Vector3(...normal).cross(z).normalize(),y=z.clone().cross(x).normalize();
    const node=mesh(feather(from.distanceTo(to),width,bend,mobile?14:22),material,parent,a);
    node.quaternion.setFromRotationMatrix(new T.Matrix4().makeBasis(x,y,z));return node;
  };
  const bolt=(parent,p,r=.021,normal=[0,0,1])=>{
    const n=mesh(new T.CylinderGeometry(r,r,.012,12),titanium,parent,p);
    n.quaternion.setFromUnitVectors(new T.Vector3(0,1,0),new T.Vector3(...normal).normalize());return n;
  };

  // The body is carried vertically over two articulated feet.
  const frame=parts.add('torso-frame',root,[0,1.7,0],[0,0,0],[0,0,0],'structure');
  mesh(verticalLoft([[-.56,.13,.15,0],[-.24,.2,.23,0],[.2,.21,.22,0],[.55,.12,.13,.12]]),graphite,frame);
  for(const s of [-1,1]){
    frame.add(curveTube([[s*.13,-.55,.04],[s*.32,-.2,.08],[s*.32,.25,.08],[s*.14,.58,.16]],.035,titanium));
    for(let i=0;i<4;i++){
      const y=-.38+i*.24;
      frame.add(rod([s*.12,y,-.16],[s*.35,y+.12,.15],.025,pale));bolt(frame,[s*.32,y+.09,.15],.035);
    }
    mesh(new T.CylinderGeometry(.063,.063,.55,16),titanium,frame,[s*.17,0,-.08]);
  }
  const core=parts.add('power-core',frame,[0,.08,.23],[0,.02,.34],[0,0,0],'structure',0);
  ring(.22,.037,core,[0,0,0],graphite);ring(.184,.014,core,[0,0,.036],brass);ring(.156,.014,core,[0,0,.055],glow);
  const disk=mesh(new T.CylinderGeometry(.147,.147,.048,36),dark,core,[0,0,.015]);disk.rotation.x=Math.PI/2;
  for(let i=0;i<12;i++){
    const a=i/12*Math.PI*2;
    core.add(rod([Math.cos(a)*.22,Math.sin(a)*.22,.005],[Math.cos(a)*.172,Math.sin(a)*.172,.043],.009,titanium));
    bolt(core,[Math.cos(a)*.22,Math.sin(a)*.22,.042],.011);
  }
  ring(.05,.01,core,[0,0,.06],brass);const coreLight=new T.PointLight(0xffd29a,.5,1.7);core.add(coreLight);
  const bodyProfile=[[-.62,.10,.15,.08],[-.44,.3,.28,.015],[-.05,.49,.43,0],[.35,.43,.37,.01],[.66,.32,.28,.09],[.83,.20,.20,.21]];
  for(const s of [-1,1]){
    const side=s<0?'left':'right';
    const armor=parts.add(`${side}-armor`,root,[0,1.7,0],[s*.64,.11,.12],[0,0,-s*.12],'structure',2);
    const angles=s>0?[-Math.PI/2+.022,Math.PI/2-.055]:[Math.PI/2+.055,Math.PI*1.5-.022];
    panel(bodyProfile,angles,armor);
    for(let i=0;i<4;i++){
      const x=s*(.11+i*.057);
      blade(armor,[x,.7-i*.085,.25-i*.01],[s*(.22+i*.059),.32-i*.065,.31-i*.014],.105,pearl,[s*.4,.1,1],s*.016);
    }
    for(const [y,z] of [[-.2,.422],[.16,.419],[.45,.332]])bolt(armor,[s*.07,y,z],.014);
  }

  // The curved ceramic neck reveals vertebral joints along its back.
  const neckProfile=[[2.25,.20,.20,.13],[2.4,.21,.23,.20],[2.55,.18,.21,.29],[2.7,.17,.19,.36],[2.8,.17,.17,.42]];
  panel(neckProfile,[-Math.PI+.08,-.08],root);
  for(let i=0;i<6;i++){
    const y=2.34+i*.073,z=.17+i*.048;
    const collar=ring(.109+(i<2?.022:0),.017,root,[0,y,z],titanium);collar.rotation.x=Math.PI/2;
    if(i<5)root.add(rod([0,y,z-.1],[0,y+.12,z-.06],.049,graphite));
  }
  for(const s of [-1,1])root.add(curveTube([[s*.13,2.3,.12],[s*.17,2.48,.17],[s*.145,2.66,.29],[s*.145,2.8,.39]],.012,brass));
  const head=parts.add('head',root,[0,2.82,.43],[0,.5,.17],[-.045,0,0],'perception',1);
  const headProfile=[[-.27,.04,.07,.02],[-.18,.16,.18,.04],[.02,.223,.238,.075],[.24,.207,.195,.045],[.43,.133,.097,-.015],[.49,.10,.065,-.025]];
  mesh(thicken(loft(headProfile,{rings:36,sides:36}),.012),pearl,head);
  const visor=parts.add('visor',head,[0,0,0],[0,.24,-.02],[-.2,0,0],'perception',0);
  const canopy=mesh(thicken(loft(headProfile,{start:.12,end:Math.PI-.12,tStart:.13,tEnd:.62,rings:24,sides:32}),.009),dark,visor);
  canopy.scale.set(1.026,1.026,1.026);
  for(const s of [-1,1]){
    const eye=mesh(new T.CylinderGeometry(.052,.06,.028,28),graphite,head,[s*.218,.097,.17]);eye.rotation.z=Math.PI/2;
    const rim=ring(.035,.006,head,[s*.237,.097,.17],brass);rim.rotation.y=Math.PI/2;
    const lens=mesh(new T.CylinderGeometry(.029,.029,.009,24),dark,head,[s*.242,.097,.17]);lens.rotation.z=Math.PI/2;
    bolt(head,[s*.196,-.01,.23],.012,[s,0,0]);
    head.add(curveTube([[s*.19,-.02,.01],[s*.199,-.04,.18],[s*.13,-.043,.42]],.005,graphite));
  }
  const beakAssembly=parts.add('beak',head,[0,-.023,.45],[0,-.03,.15],[0,0,0],'perception',2);
  const beak=new T.Group();beakAssembly.add(beak);
  mesh(thicken(loft([[0,.101,.055,0],[.17,.086,.051,-.004],[.38,.057,.043,-.008],[.56,.024,.04,-.02],[.63,.002,.007,-.052]],{rings:28,sides:24}),.008),brass,beak);
  beak.scale.z=.8;
  for(const s of [-1,1]){
    beak.add(curveTube([[s*.1,-.015,0],[s*.082,-.02,.18],[s*.045,-.03,.43],[0,-.052,.62]],.0035,graphite));
    const nostril=mesh(new T.CylinderGeometry(.012,.012,.012,14),dark,beak,[s*.078,.012,.23]);nostril.rotation.z=Math.PI/2;nostril.scale.z=2.4;
  }

  // Folded wings: long flight feathers under three nested rows of coverts.
  for(const s of [-1,1]){
    const side=s<0?'left':'right',normal=[s,.05,.08];
    const wing=parts.add(`${side}-wing-root`,root,[s*.37,2.22,-.075],[s*1.15,.15,-.06],[0,-s*.65,-s*.23],'movement',1);
    const wingProfile=[[-1.05,.015,.035,-.48],[-.75,.11,.18,-.33],[-.3,.155,.26,-.14],[.05,.12,.21,-.065],[.16,.025,.05,-.06]];
    panel(wingProfile,s>0?[-Math.PI/2,Math.PI/2]:[Math.PI/2,Math.PI*1.5],wing,graphite);
    const actuator=parts.add(`${side}-actuators`,wing,[0,0,0],[s*.18,0,.26],[0,0,0],'movement',2);
    actuator.add(rod([s*.07,.02,0],[s*.14,-.57,-.23],.047,titanium));
    actuator.add(rod([s*.08,-.25,-.04],[s*.16,-.73,-.34],.026,brass));
    actuator.add(rod([s*.14,-.57,-.23],[s*.08,-.98,-.47],.028,titanium));
    for(const p of [[s*.07,.02,0],[s*.14,-.57,-.23]]){
      const joint=mesh(new T.CylinderGeometry(.074,.074,.10,24),graphite,actuator,p);joint.rotation.z=Math.PI/2;
      bolt(actuator,[p[0]+s*.059,p[1],p[2]],.043,[s,0,0]);
    }
    const feathers=parts.add(`${side}-feather-plates`,wing,[0,0,0],[s*.24,-.02,-.3],[0,s*.12,0],'movement',3);
    const count=mobile?9:13;
    for(let i=0;i<count;i++){
      const t=i/(count-1),a=[s*(.115+.027*Math.sin(t*Math.PI)),-.1-t*.35,.075-t*.27],b=[s*(.11-.1*t),-.96-t*.16,-.34-t*.34];
      const part=parts.add(`${side}-feather-${i}`,feathers,a,[s*.04*t,-.04*t,-.05*t],[0,s*t*.08,0],'movement',3+t);
      const end=b.map((v,j)=>v-a[j]);blade(part,[0,0,0],end,.091,t>.53?graphite:pale,normal,s*.025);
      if(!mobile)blade(part,[s*.012,-.08,-.025],end.map((v,j)=>j===1?v*.91:v),.014,t>.53?titanium:pearl,normal,s*.01);
    }
    for(let row=0;row<3;row++)for(let i=0;i<(mobile?5:7);i++){
      const t=i/(mobile?4:6),x=s*(.13+.035*Math.sin(t*Math.PI));
      blade(wing,[x,.12-row*.185-t*.14,.08-t*.26-row*.065],[x+s*.018,-.2-row*.20-t*.15,-.05-t*.29-row*.09],.079-row*.008,row===2?pale:pearl,normal,s*.025);
    }
    blade(wing,[s*.09,.17,.07],[s*.173,-.22,.01],.15,pearl,normal,s*.02);bolt(wing,[s*.187,.015,.04],.02,[s,0,0]);
  }

  const tail=parts.add('tail',root,[0,1.22,-.38],[0,.02,-.62],[.05,0,0],'connection',1);
  for(let i=0;i<7;i++){
    const x=(i-3)*.052;blade(tail,[x,0,0],[x*1.7,-.39,-.67-Math.abs(i-3)*.018],.09,i===0||i===6?pale:pearl,[0,1,.3],x*.1);
  }
  for(const s of [-1,1]){
    const side=s<0?'left':'right';
    const leg=parts.add(`${side}-leg`,root,[s*.205,0,.06],[s*.17,0,.14],[0,0,0],'connection',2);
    leg.add(rod([0,.12,.035],[0,.75,-.018],.027,titanium));leg.add(rod([0,.75,-.018],[s*.055,1.13,.025],.042,graphite));
    leg.add(rod([s*.034,.2,.05],[s*.034,.63,.008],.013,brass));
    for(const [y,z,r] of [[.17,.034,.052],[.74,-.016,.062],[1.04,.013,.064]]){
      const joint=mesh(new T.CylinderGeometry(r,r,.085,24),graphite,leg,[0,y,z]);joint.rotation.z=Math.PI/2;
      bolt(leg,[s*.052,y,z],r*.65,[s,0,0]);
    }
    for(let i=0;i<3;i++){
      const spread=(i-1)*.145,z=i===1?.36:.27;
      leg.add(curveTube([[0,.12,.045],[spread*.55,.062,.17],[spread,.035,z]],.02,titanium,12));
      blade(leg,[spread*.18,.065,.095],[spread*.92,.031,z-.013],.08,pale,[0,1,.2],spread*.05);
    }
    leg.add(curveTube([[0,.12,.035],[0,.055,-.095],[s*.045,.035,-.16]],.018,graphite,10));
  }
  const cableAnchors=[
    ['torso-frame',[0,.61,.19],'head',[0,-.1,-.1]],
    ['torso-frame',[-.28,.3,-.04],'left-wing-root',[-.02,-.14,-.08]],
    ['torso-frame',[.28,.3,-.04],'right-wing-root',[.02,-.14,-.08]],
    ['torso-frame',[0,-.4,-.2],'tail',[0,0,0]]
  ].map(([a,pa,b,pb])=>[parts.attachment(a,pa),parts.attachment(b,pb)]);
  return {root,parts,cableAnchors,coreLight,glow,materials};
}
