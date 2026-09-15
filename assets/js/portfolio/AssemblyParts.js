import * as T from '../../vendor/three/three.module.min.js';

export class AssemblyParts {
  constructor(root) { this.root = root; this.items = []; this.named = new Map(); }
  add(name, parent = this.root, position = [0,0,0], explosion = [0,0,0], rotation = [0,0,0], phase = 'structure', order = 0) {
    const node = new T.Group(); node.name = name; node.position.set(...position); parent.add(node);
    const part = {
      name, node, parent: parent.name || 'assembly', phase, order,
      assembled: { position: node.position.clone(), quaternion: node.quaternion.clone(), scale: node.scale.clone() },
      exploded: { position: node.position.clone().add(new T.Vector3(...explosion)), quaternion: new T.Quaternion().setFromEuler(new T.Euler(...rotation)), scale: node.scale.clone() }
    };
    this.items.push(part); this.named.set(name, part); return node;
  }
  get(name) { return this.named.get(name)?.node; }
  attachment(name, position) {
    const anchor = new T.Object3D(); anchor.position.set(...position); this.get(name).add(anchor); return anchor;
  }
}
