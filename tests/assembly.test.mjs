import test from 'node:test';
import assert from 'node:assert/strict';
import {createSeagull} from '../assets/js/portfolio/SeagullAssembly.js';
import {AssemblyTimeline,partAmount} from '../assets/js/portfolio/AssemblyTimeline.js';
import {CableSystem} from '../assets/js/portfolio/CableSystem.js';

function pose(bird){return bird.parts.items.map(p=>[...p.node.position,...p.node.quaternion,...p.node.scale]);}
function release(bird){const geometries=new Set();bird.root.traverse(o=>o.geometry&&geometries.add(o.geometry));geometries.forEach(g=>g.dispose());bird.materials.forEach(m=>m.dispose());}
test('arbitrary fast scrolling returns every component to its assembled transform without drift',()=>{
  const bird=createSeagull(),timeline=new AssemblyTimeline(bird.parts);const initial=pose(bird);
  for(let i=0;i<500;i++)timeline.apply(((i*137)%503)/503);
  timeline.apply(0);assert.deepEqual(pose(bird),initial);
  timeline.apply(1);assert.deepEqual(pose(bird),initial);
  for(const p of bird.parts.items)assert.equal(partAmount(.94,p),0,`${p.name} must settle before contact`);
  release(bird);
});
test('exploded state is reversible, stays finite and keeps a connected hierarchy',()=>{
  const bird=createSeagull(),timeline=new AssemblyTimeline(bird.parts);
  for(const name of ['head','visor','beak','left-armor','right-armor','torso-frame','power-core','left-wing-root','right-wing-root','left-actuators','right-actuators','left-feather-plates','right-feather-plates','tail'])assert.ok(bird.parts.get(name),name);
  assert.equal(bird.parts.get('visor').parent,bird.parts.get('head'));
  timeline.apply(.78);const exploded=pose(bird);timeline.apply(.1);timeline.apply(.99);timeline.apply(.78);assert.deepEqual(pose(bird),exploded);
  for(const row of exploded)assert.ok(row.every(Number.isFinite));
  for(const p of bird.parts.items)assert.ok(p.node.scale.x>0);
  release(bird);
});
test('mobile reduces mesh density and separation; cables track transformed attachment points',()=>{
  const desktop=createSeagull(),mobile=createSeagull({mobile:true});
  assert.ok(mobile.parts.items.length<desktop.parts.items.length);
  new AssemblyTimeline(desktop.parts).apply(.78);new AssemblyTimeline(mobile.parts).apply(.78,true);
  assert.ok(mobile.parts.get('left-armor').position.length()<desktop.parts.get('left-armor').position.length());
  const cable=new CableSystem(desktop.root,desktop.cableAnchors,false);cable.update();
  const id=cable.cables[0].mesh.geometry.id; cable.update();assert.equal(cable.cables[0].mesh.geometry.id,id,'unchanged endpoints reuse geometry');
  new AssemblyTimeline(desktop.parts).apply(0);cable.update();assert.notEqual(cable.cables[0].mesh.geometry.id,id);
  for(const c of cable.cables)assert.ok([...c.mesh.geometry.attributes.position.array].every(Number.isFinite));
  cable.dispose();release(desktop);release(mobile);
});
