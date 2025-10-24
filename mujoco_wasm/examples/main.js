
import * as THREE           from 'three';
import * as ort             from 'onnxruntime-web';
import { GUI              } from '../node_modules/three/examples/jsm/libs/lil-gui.module.min.js';
import { OrbitControls    } from '../node_modules/three/examples/jsm/controls/OrbitControls.js';
import { DragStateManager } from './utils/DragStateManager.js';
import { setupGUI, downloadExampleScenesFolder, loadSceneFromURL, getPosition, getQuaternion, toMujocoPos, standardNormal } from './mujocoUtils.js';
import { ONNXModule } from './onnxHelper.js';
import { Observations } from './observationHelpers.js';
import   load_mujoco        from '../dist/mujoco_wasm.js';

// Simple loading overlay (like facet)
const loadingOverlay = document.createElement('div');
loadingOverlay.style.position = 'fixed';
loadingOverlay.style.inset = '0';
loadingOverlay.style.background = 'rgba(0,0,0,0.6)';
loadingOverlay.style.display = 'flex';
loadingOverlay.style.alignItems = 'center';
loadingOverlay.style.justifyContent = 'center';
loadingOverlay.style.zIndex = '9999';
const loadingBox = document.createElement('div');
loadingBox.style.padding = '16px 24px';
loadingBox.style.borderRadius = '8px';
loadingBox.style.background = '#1f2937';
loadingBox.style.color = 'white';
loadingBox.style.font = '500 16px system-ui, sans-serif';
const loadingText = document.createElement('div');
loadingText.textContent = 'Loading MuJoCo (WASM)...';
const loadingBarWrap = document.createElement('div');
loadingBarWrap.style.marginTop = '10px';
loadingBarWrap.style.width = '260px';
loadingBarWrap.style.height = '6px';
loadingBarWrap.style.borderRadius = '999px';
loadingBarWrap.style.background = 'rgba(255,255,255,0.2)';
const loadingBar = document.createElement('div');
loadingBar.style.width = '40%';
loadingBar.style.height = '100%';
loadingBar.style.borderRadius = '999px';
loadingBar.style.background = '#60a5fa';
loadingBar.style.transition = 'width 300ms ease';
loadingBarWrap.appendChild(loadingBar);
loadingBox.appendChild(loadingText);
loadingBox.appendChild(loadingBarWrap);
loadingOverlay.appendChild(loadingBox);
document.body.appendChild(loadingOverlay);

// Load the MuJoCo Module
loadingText.textContent = 'Loading MuJoCo (WASM)...';
const mujoco = await load_mujoco();
loadingBar.style.width = '55%';

// Set up Emscripten's Virtual File System
var initialScene = "unitree_go2/scene.xml";
mujoco.FS.mkdir('/working');
mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');

// Download all example scenes and dependencies first
loadingText.textContent = 'Downloading assets...';
await downloadExampleScenesFolder(mujoco);
loadingBar.style.width = '70%';

export class MuJoCoDemo {
  constructor() {
    this.mujoco = mujoco;

    // Load in the state from XML
    this.model      = new mujoco.Model("/working/" + initialScene);
    this.state      = new mujoco.State(this.model);
    this.simulation = new mujoco.Simulation(this.model, this.state);

    // Define Random State Variables
    this.params = { 
      scene: initialScene, 
      paused: false, 
      help: false, 
      ctrlnoiserate: 0.0, 
      ctrlnoisestd: 0.0, 
      keyframeNumber: 0,
      policy: "./examples/checkpoints/robust.json",
      command_vel_x: 0.0,
      command_vel_y: 0.0,
      command_vel_z: 0.0,
      command_vel_yaw: 0.0,
      impedance_kp: 25.0,
      use_setpoint: false,
      compliant_mode: false,
      impulse_remain_time: 0.0
    };
    this.mujoco_time = 0.0;
    this.bodies  = {}, this.lights = {};
    this.tmpVec  = new THREE.Vector3();
    this.tmpQuat = new THREE.Quaternion();
    this.quat    = new THREE.Quaternion();
    this.updateGUICallbacks = [];
    
    // Policy-related properties
    this.policy = null;
    this.observations = {};
    this.inputDict = {};
    this.lastActions = null;
    this.isInferencing = false;
    this.inferenceGen = 0;
    this.adapt_hx = new Float32Array(128).fill(0);
    this.rpy = new THREE.Euler(0, 0, 0);

    this.container = document.createElement( 'div' );
    document.body.appendChild( this.container );

    this.scene = new THREE.Scene();
    this.scene.name = 'scene';

    this.camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 0.001, 100 );
    this.camera.name = 'PerspectiveCamera';
    this.camera.position.set(2.0, 1.7, 1.7);
    this.scene.add(this.camera);

    this.scene.background = new THREE.Color(0.15, 0.25, 0.35);
    this.scene.fog = new THREE.Fog(this.scene.background, 15, 25.5 );

    this.ambientLight = new THREE.AmbientLight( 0xffffff, 0.1 );
    this.ambientLight.name = 'AmbientLight';
    this.scene.add( this.ambientLight );

    this.renderer = new THREE.WebGLRenderer( { antialias: true } );
    this.renderer.setPixelRatio(1.0);////window.devicePixelRatio );
    this.renderer.setSize( window.innerWidth, window.innerHeight );
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap; // default THREE.PCFShadowMap
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    //this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.setAnimationLoop( this.render.bind(this) );

    this.container.appendChild( this.renderer.domElement );

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.7, 0);
    this.controls.panSpeed = 2;
    this.controls.zoomSpeed = 1;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.10;
    this.controls.screenSpacePanning = true;
    this.controls.update();

    window.addEventListener('resize', this.onWindowResize.bind(this));

    // Initialize the Drag State Manager.
    this.dragStateManager = new DragStateManager(this.scene, this.renderer, this.camera, this.container.parentElement, this.controls);
  }

  async init() {
    // Download the the examples to MuJoCo's virtual file system
    // await downloadExampleScenesFolder(mujoco);

    // Initialize the three.js Scene using the .xml Model in initialScene
    loadingText.textContent = 'Loading scene XML...';
    [this.model, this.state, this.simulation, this.bodies, this.lights] =  
      await loadSceneFromURL(mujoco, initialScene, this);
    loadingBar.style.width = '80%';

    this.gui = new GUI();
    setupGUI(this);
    // Auto-load robust policy on start
    loadingText.textContent = 'Loading ONNX policy...';
    await this.loadPolicy(this.params.policy);
    loadingBar.style.width = '100%';
    setTimeout(() => { loadingOverlay.remove(); }, 300);
  }

  async loadPolicy(policyPath) {
    console.log("Loading policy:", policyPath);
    
    // Wait until inference is not running
    while (this.isInferencing) {
      await new Promise(resolve => setTimeout(resolve, 10));
    }

    // Load policy config from JSON
    const response = await fetch(policyPath);
    const config = await response.json();

    // Initialize ONNX model (defer assigning to this.policy until session is ready)
    const policy = new ONNXModule(config.onnx);
    await policy.init();
    this.adapt_hx.fill(0);
    this.rpy.set(0, 0, 0);

    this.simulation.resetData();
    this.simulation.forward();

    // Initialize action buffers before constructing observations so PrevActions has correct dims
    this.numActions = this.model.nu;
    this.actionBuffer = new Array(4).fill().map(() => new Float32Array(this.numActions));
    this.lastActions = new Float32Array(this.numActions);

    // Helper function to create observation instance
    const createObservation = (obsConfig) => {
      const ObsClass = Observations[obsConfig.name];
      if (!ObsClass) {
        throw new Error(`Unknown observation type: ${obsConfig.name}`);
      }

      // Handle special case for joint names
      const kwargs = {...obsConfig};
      delete kwargs.name;

      if (kwargs.joint_names === "isaac") {
        kwargs.joint_names = this.jointNamesIsaac;
      }

      return new ObsClass(this.model, this.simulation, this, kwargs);
    };

    // Set up observations based on config
    this.observations = {};
    for (const [key, obsList] of Object.entries(config.obs_config)) {
      this.observations[key] = obsList.map(obsConfig => createObservation(obsConfig));
    }

    this.action_scale = new Float32Array(this.model.nu).fill(config.action_scale);
    this.jntKp = new Float32Array(this.model.nu).fill(config.stiffness);
    this.jntKd = new Float32Array(this.model.nu).fill(config.damping);
    this.control_type = config.control_type ?? "joint_position";
    // Assign policy only after it has an initialized session
    this.policy = policy;
    // Initialize recurrent inputs (is_init, adapt_hx)
    this.inputDict = this.policy.initInput();
    
    console.log("Policy loaded successfully");
    // Reset recurrent inputs and invalidate any in-flight inference
    this.inputDict = this.policy.initInput();
    this.inferenceGen++;
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize( window.innerWidth, window.innerHeight );
  }

  render(timeMS) {
    this.controls.update();

    if (!this.params["paused"]) {
      let timestep = this.model.getOptions().timestep;
      if (timeMS - this.mujoco_time > 35.0) { this.mujoco_time = timeMS; }
      while (this.mujoco_time < timeMS) {

        // Jitter the control state with gaussian random noise
        if (this.params["ctrlnoisestd"] > 0.0) {
          let rate  = Math.exp(-timestep / Math.max(1e-10, this.params["ctrlnoiserate"]));
          let scale = this.params["ctrlnoisestd"] * Math.sqrt(1 - rate * rate);
          let currentCtrl = this.simulation.ctrl;
          for (let i = 0; i < currentCtrl.length; i++) {
            currentCtrl[i] = rate * currentCtrl[i] + scale * standardNormal();
            this.params["Actuator " + i] = currentCtrl[i];
          }
        }

        // Clear old perturbations, apply new ones.
        for (let i = 0; i < this.simulation.qfrc_applied.length; i++) { this.simulation.qfrc_applied[i] = 0.0; }
        let dragged = this.dragStateManager.physicsObject;
        if (dragged && dragged.bodyID) {
          for (let b = 0; b < this.model.nbody; b++) {
            if (this.bodies[b]) {
              getPosition  (this.simulation.xpos , b, this.bodies[b].position);
              getQuaternion(this.simulation.xquat, b, this.bodies[b].quaternion);
              this.bodies[b].updateWorldMatrix();
            }
          }
          let bodyID = dragged.bodyID;
          this.dragStateManager.update(); // Update the world-space force origin
          const dragOffset = this.dragStateManager.currentWorld.clone().sub(this.dragStateManager.worldHit);
          let force = toMujocoPos(dragOffset.multiplyScalar(25));
          let point = toMujocoPos(this.dragStateManager.worldHit.clone());
          this.simulation.applyForce(force.x, force.y, force.z, 0, 0, 0, point.x, point.y, point.z, bodyID);

          // TODO: Apply pose perturbations (mocap bodies only).
        }

        // Run policy and apply control at control rate (decimated by dt ~ 20ms)
        if (this.policy && this.policy.session && !this.isInferencing) {
          // Update base quat/euler
          const q = this.simulation.qpos.subarray(3, 7);
          this.tmpQuat.set(q[1], q[2], q[3], q[0]);
          this.quat.copy(this.tmpQuat);
          this.rpy.setFromQuaternion(this.tmpQuat);
          // Build observations and run inference
          for (const [obs_key, obs_funcs] of Object.entries(this.observations)) {
            let flat = [];
            let debugParts = [];
            for (const fn of obs_funcs) {
              const arr = fn.compute();
              flat.push(...arr);
              if (obs_key === 'policy') debugParts.push({ name: fn.constructor.name, len: arr.length });
            }
            if (obs_key === 'policy' && (this.simStepCount % 50) === 0) {
              console.log('policy obs classes', debugParts.map(p => p.name));
              console.log('policy parts', debugParts, 'total', flat.length, 'expected', 117);
            }
            this.inputDict[obs_key] = new ort.Tensor('float32', flat, [1, flat.length]);
          }
          if (this.policy && this.policy.session && (this.simStepCount % 50) === 0) {
            console.log('feed keys', Object.keys(this.inputDict), 'session inputs', this.policy.session.inputNames);
          }
          // Run inference asynchronously to avoid await inside render loop
          const runGen = this.inferenceGen;
          this.isInferencing = true;
          const inputCopy = { ...this.inputDict };
          this.policy.runInference(inputCopy).then(([result, carry]) => {
            if (runGen !== this.inferenceGen) { this.isInferencing = false; return; }
            const action = result['action'].data;
            for (let i = 0; i < this.lastActions.length; i++) {
              this.lastActions[i] = this.lastActions[i] * 0.2 + action[i] * 0.8;
            }
            for (let i = this.actionBuffer.length - 1; i > 0; i--) this.actionBuffer[i] = this.actionBuffer[i - 1];
            this.actionBuffer[0] = this.lastActions;
            this.inputDict = carry;
            this.isInferencing = false;
          }).catch((e) => { console.error('Inference error', e); this.isInferencing = false; });
          // Apply PD control
          for (let i = 0; i < this.numActions; i++) {
            const qpos_adr = this.qpos_adr_isaac[i];
            const qvel_adr = this.qvel_adr_isaac[i];
            const ctrl_adr = this.ctrl_adr_isaac[i];
            const target = this.action_scale[i] * this.lastActions[i] + this.defaultJpos[i];
            const torque = this.jntKp[i] * (target - this.simulation.qpos[qpos_adr]) + this.jntKd[i] * (0 - this.simulation.qvel[qvel_adr]);
            this.simulation.ctrl[ctrl_adr] = torque;
          }
          if (this.params["impulse_remain_time"] > 0) {
            const force = new THREE.Vector3(0, 50, 0);
            const point = new THREE.Vector3(0, 0, 0);
            getPosition(this.simulation.xpos, this.pelvis_body_id, point, false);
            this.simulation.applyForce(force.x, force.y, force.z, 0, 0, 0, point.x, point.y, point.z, this.pelvis_body_id);
            this.params["impulse_remain_time"] -= timestep;
          }

        }
        this.simulation.step();

        this.mujoco_time += timestep * 1000.0;
      }

    } else if (this.params["paused"]) {
      this.dragStateManager.update(); // Update the world-space force origin
      let dragged = this.dragStateManager.physicsObject;
      if (dragged && dragged.bodyID) {
        let b = dragged.bodyID;
        getPosition  (this.simulation.xpos , b, this.tmpVec , false); // Get raw coordinate from MuJoCo
        getQuaternion(this.simulation.xquat, b, this.tmpQuat, false); // Get raw coordinate from MuJoCo

        let offset = toMujocoPos(this.dragStateManager.currentWorld.clone()
          .sub(this.dragStateManager.worldHit).multiplyScalar(0.3));
        if (this.model.body_mocapid[b] >= 0) {
          // Set the root body's mocap position...
          console.log("Trying to move mocap body", b);
          let addr = this.model.body_mocapid[b] * 3;
          let pos  = this.simulation.mocap_pos;
          pos[addr+0] += offset.x;
          pos[addr+1] += offset.y;
          pos[addr+2] += offset.z;
        } else {
          // Set the root body's position directly...
          let root = this.model.body_rootid[b];
          let addr = this.model.jnt_qposadr[this.model.body_jntadr[root]];
          let pos  = this.simulation.qpos;
          pos[addr+0] += offset.x;
          pos[addr+1] += offset.y;
          pos[addr+2] += offset.z;

          //// Save the original root body position
          //let x  = pos[addr + 0], y  = pos[addr + 1], z  = pos[addr + 2];
          //let xq = pos[addr + 3], yq = pos[addr + 4], zq = pos[addr + 5], wq = pos[addr + 6];

          //// Clear old perturbations, apply new ones.
          //for (let i = 0; i < this.simulation.qfrc_applied().length; i++) { this.simulation.qfrc_applied()[i] = 0.0; }
          //for (let bi = 0; bi < this.model.nbody(); bi++) {
          //  if (this.bodies[b]) {
          //    getPosition  (this.simulation.xpos (), bi, this.bodies[bi].position);
          //    getQuaternion(this.simulation.xquat(), bi, this.bodies[bi].quaternion);
          //    this.bodies[bi].updateWorldMatrix();
          //  }
          //}
          ////dragStateManager.update(); // Update the world-space force origin
          //let force = toMujocoPos(this.dragStateManager.currentWorld.clone()
          //  .sub(this.dragStateManager.worldHit).multiplyScalar(this.model.body_mass()[b] * 0.01));
          //let point = toMujocoPos(this.dragStateManager.worldHit.clone());
          //// This force is dumped into xrfc_applied
          //this.simulation.applyForce(force.x, force.y, force.z, 0, 0, 0, point.x, point.y, point.z, b);
          //this.simulation.integratePos(this.simulation.qpos(), this.simulation.qfrc_applied(), 1);

          //// Add extra drag to the root body
          //pos[addr + 0] = x  + (pos[addr + 0] - x ) * 0.1;
          //pos[addr + 1] = y  + (pos[addr + 1] - y ) * 0.1;
          //pos[addr + 2] = z  + (pos[addr + 2] - z ) * 0.1;
          //pos[addr + 3] = xq + (pos[addr + 3] - xq) * 0.1;
          //pos[addr + 4] = yq + (pos[addr + 4] - yq) * 0.1;
          //pos[addr + 5] = zq + (pos[addr + 5] - zq) * 0.1;
          //pos[addr + 6] = wq + (pos[addr + 6] - wq) * 0.1;


        }
      }

      this.simulation.forward();
    }

    // Update body transforms.
    for (let b = 0; b < this.model.nbody; b++) {
      if (this.bodies[b]) {
        getPosition  (this.simulation.xpos , b, this.bodies[b].position);
        getQuaternion(this.simulation.xquat, b, this.bodies[b].quaternion);
        this.bodies[b].updateWorldMatrix();
      }
    }

    // Update light transforms.
    for (let l = 0; l < this.model.nlight; l++) {
      if (this.lights[l]) {
        getPosition(this.simulation.light_xpos, l, this.lights[l].position);
        getPosition(this.simulation.light_xdir, l, this.tmpVec);
        this.lights[l].lookAt(this.tmpVec.add(this.lights[l].position));
      }
    }

    // Update tendon transforms.
    let numWraps = 0;
    if (this.mujocoRoot && this.mujocoRoot.cylinders) {
      let mat = new THREE.Matrix4();
      for (let t = 0; t < this.model.ntendon; t++) {
        let startW = this.simulation.ten_wrapadr[t];
        let r = this.model.tendon_width[t];
        for (let w = startW; w < startW + this.simulation.ten_wrapnum[t] -1 ; w++) {
          let tendonStart = getPosition(this.simulation.wrap_xpos, w    , new THREE.Vector3());
          let tendonEnd   = getPosition(this.simulation.wrap_xpos, w + 1, new THREE.Vector3());
          let tendonAvg   = new THREE.Vector3().addVectors(tendonStart, tendonEnd).multiplyScalar(0.5);

          let validStart = tendonStart.length() > 0.01;
          let validEnd   = tendonEnd  .length() > 0.01;

          if (validStart) { this.mujocoRoot.spheres.setMatrixAt(numWraps    , mat.compose(tendonStart, new THREE.Quaternion(), new THREE.Vector3(r, r, r))); }
          if (validEnd  ) { this.mujocoRoot.spheres.setMatrixAt(numWraps + 1, mat.compose(tendonEnd  , new THREE.Quaternion(), new THREE.Vector3(r, r, r))); }
          if (validStart && validEnd) {
            mat.compose(tendonAvg, new THREE.Quaternion().setFromUnitVectors(
              new THREE.Vector3(0, 1, 0), tendonEnd.clone().sub(tendonStart).normalize()),
              new THREE.Vector3(r, tendonStart.distanceTo(tendonEnd), r));
            this.mujocoRoot.cylinders.setMatrixAt(numWraps, mat);
            numWraps++;
          }
        }
      }
      this.mujocoRoot.cylinders.count = numWraps;
      this.mujocoRoot.spheres  .count = numWraps > 0 ? numWraps + 1: 0;
      this.mujocoRoot.cylinders.instanceMatrix.needsUpdate = true;
      this.mujocoRoot.spheres  .instanceMatrix.needsUpdate = true;
    }

    // Render!
    this.renderer.render( this.scene, this.camera );
  }
}

let demo = new MuJoCoDemo();
await demo.init();
