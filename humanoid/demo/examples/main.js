
import * as THREE           from 'three';
import * as ort             from 'onnxruntime-web';
import { GUI              } from '../node_modules/three/examples/jsm/libs/lil-gui.module.min.js';
import { OrbitControls    } from '../node_modules/three/examples/jsm/controls/OrbitControls.js';
import { DragStateManager } from './utils/DragStateManager.js';
import { setupGUI, downloadExampleScenesFolder, loadSceneFromURL, getPosition, getQuaternion, toMujocoPos, standardNormal } from './mujocoUtils.js';
import { ONNXModule } from './onnxHelper.js';
import { Observations } from './observationHelpers.js';
import { parseYAMLConfig, yamlConfigToPolicyConfig } from './yamlParser.js';
import   load_mujoco        from 'mujoco-js/dist/mujoco_wasm.js';

// Simple loading overlay
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
var initialScene = "unitree_g1/scene_23dof.xml";
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
    this.model = mujoco.MjModel.loadFromXML("/working/" + initialScene);
    this.data  = new mujoco.MjData(this.model);

    // Define Random State Variables
    this.params = { 
      scene: initialScene, 
      paused: false, 
      help: false, 
      ctrlnoiserate: 0.0, 
      ctrlnoisestd: 0.0, 
      keyframeNumber: 0,
      policy: "./examples/checkpoints/g1/balance/balance_deploy_state_projection.yaml",
      policyOnnx: "./examples/checkpoints/g1/balance/balance_policy_state_projection.onnx",
      policyLabel: 'Ours',
      motionDataset: null,
      command_vel_x: 0.0,
      command_vel_z: 0.0,
      command_vel_yaw: 0.0,
      use_setpoint: false,
      impulse_remain_time: 0.0
    };
    this.mujoco_time = 0.0;
    this.simStepCount = 0;
    this.control_decimation = 1; // Default to no decimation, will be set when policy loads
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
    this.motionObservationsPresent = false;
    this.observationsReady = false;
    this.policyLoading = false;
    this.policyLoadedListeners = [];
    this.motionDatasetOverride = null;
    this.motionProgress = 0;
    this.motionDuration = 0;
    this.motionUISeconds = 0;
    this.motionStartTimeMS = null;
    this.replayProgressController = null;
    this.replayMotionController = null;
    this.replayControls = null;
    this._updatingReplaySlider = false;

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

    this.ambientLight = new THREE.AmbientLight( 0xffffff, 0.25 * Math.PI );
    this.ambientLight.name = 'AmbientLight';
    this.scene.add( this.ambientLight );

    this.fillLight = new THREE.HemisphereLight(0xb0c4ff, 0x3a2a1a, 0.6);
    this.scene.add(this.fillLight);

    this.spotlight = new THREE.SpotLight();
    this.spotlight.angle = 1.15;
    this.spotlight.distance = 10000;
    this.spotlight.penumbra = 0.5;
    this.spotlight.castShadow = true;
    this.spotlight.shadow.bias = -5e-4;
    this.spotlight.shadow.normalBias = 5e-4;
    this.spotlight.shadow.radius = 2.0;
    this.spotlight.shadow.mapSize.width = 1024;
    this.spotlight.shadow.mapSize.height = 1024;
    this.spotlight.shadow.camera.near = 0.1;
    this.spotlight.shadow.camera.far = 100;
    this.spotlight.position.set(0, 3, 3);
    this.spotlight.intensity = this.spotlight.intensity * Math.PI * 14.0;
    this.scene.add(this.spotlight);
    this.spotlightTarget = new THREE.Object3D();
    this.spotlightTarget.position.set(0, 1, 0);
    this.scene.add(this.spotlightTarget);
    this.spotlight.target = this.spotlightTarget;
    this.spotlightOffset = new THREE.Vector3(0, 3, 3);
    this.spotlightTargetOffset = new THREE.Vector3(0, 1, 0);

    this.renderer = new THREE.WebGLRenderer( { antialias: true } );
    this.renderer.setPixelRatio(1.0);////window.devicePixelRatio );
    this.renderer.setSize( window.innerWidth, window.innerHeight );
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap; // default THREE.PCFShadowMap
    THREE.ColorManagement.enabled = false;
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.renderer.useLegacyLights = true;
    //this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    // Don't start animation loop yet - wait for policy to load in init()
    // this.renderer.setAnimationLoop( this.render.bind(this) );

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

  getMotionObservation() {
    if (!this.observations || !Array.isArray(this.observations.obs)) {
      return null;
    }
    return this.observations.obs.find(
      (obsInstance) => obsInstance && obsInstance.motionLoader
    ) || null;
  }

  updateMotionSliderRange(duration) {
    if (!this.replayProgressController || typeof this.replayProgressController.min !== 'function') {
      return;
    }
    const roundedMax = Math.max(0.1, Math.round(Math.max(0, duration) * 10) / 10);
    this.replayProgressController.min(0);
    this.replayProgressController.max(roundedMax);
    this.replayProgressController.step(0.1);
  }

  async setMotionProgress(normalized, options = {}) {
    const { syncState = true } = options;

    const clamped = Math.min(Math.max(normalized, 0), 1);
    if (!this.motionObservationsPresent) {
      this.motionProgress = clamped;
      this.motionDuration = 0;
      this.motionUISeconds = 0;
      this.updateMotionSliderRange(1);
      return;
    }
    if (!this.observations || !Array.isArray(this.observations.obs)) return;

    for (const obs of this.observations.obs) {
      if (typeof obs.setMotionProgress === 'function') {
        await obs.setMotionProgress(clamped);
      }
    }

    if (syncState) {
      mujoco.mj_resetData(this.model, this.data);
      mujoco.mj_forward(this.model, this.data);
    }

    const motionObs = this.getMotionObservation();
    let displayTime = 0;
    if (motionObs && motionObs.motionLoader) {
      const duration = motionObs.motionLoader.duration || 0;
      this.motionDuration = duration;
      const currentTime = Math.max(0, Math.min(duration, motionObs.motionTime));
      this.motionProgress = duration > 0 ? currentTime / duration : 0;
      displayTime = Math.round(currentTime * 10) / 10;
      this.motionUISeconds = displayTime;
      this.updateMotionSliderRange(duration);
    } else {
      this.motionProgress = clamped;
      this.motionDuration = 0;
      this.motionUISeconds = 0;
      this.updateMotionSliderRange(1);
    }

    this.motionStartTimeMS = typeof performance !== 'undefined' ? performance.now() : null;

    if (this.replayControls && this.replayProgressController) {
      this.replayControls.replayTime = this.motionUISeconds;
      this._updatingReplaySlider = true;
      this.replayProgressController.updateDisplay();
      this._updatingReplaySlider = false;
    }
  }

  async init() {
    // Download the the examples to MuJoCo's virtual file system
    // await downloadExampleScenesFolder(mujoco);

    // Initialize the three.js Scene using the .xml Model in initialScene
    loadingText.textContent = 'Loading scene XML...';
    [this.model, this.data, this.bodies, this.lights] =  
      await loadSceneFromURL(mujoco, initialScene, this);
    loadingBar.style.width = '80%';

    this.gui = new GUI();
    setupGUI(this);
    // Auto-load robust policy on start
    loadingText.textContent = 'Loading ONNX policy...';
    await this.loadPolicy(this.params.policy, this.params.policyOnnx);
    loadingBar.style.width = '100%';
    setTimeout(() => { loadingOverlay.remove(); }, 300);
    
    // Start the animation loop AFTER policy is loaded
    console.log("Starting render loop...");
    this.renderer.setAnimationLoop( this.render.bind(this) );
  }

  resetSimulation() {
    console.log("[Reset] begin");
    
    // Stop any in-flight inference
    this.inferenceGen++;
    this.isInferencing = false;
    
    // Reset MuJoCo simulation
    mujoco.mj_resetData(this.model, this.data);
    
    // Reset robot to default pose
    if (this.defaultJpos && this.joint_ids_map) {
      for (let i = 0; i < this.numActions; i++) {
        const joint_idx = this.joint_ids_map[i];
        const joint_name = this.jointNamesIsaac[joint_idx];
        const mjc_idx = this.jointNamesMJC.indexOf(joint_name);
        if (mjc_idx >= 0) {
          const qpos_adr = this.model.jnt_qposadr[mjc_idx];
          this.data.qpos[qpos_adr] = this.defaultJpos[i];
        }
      }
    }
    
    // Reset action buffers
    if (this.lastActions) {
      this.lastActions.fill(0);
    }
    if (this.actionBuffer) {
      this.actionBuffer.forEach(buf => buf.fill(0));
    }
    
    // Reset observation history by reinitializing observations
    if (this.observations && this.observations.obs) {
      for (const obs of this.observations.obs) {
        if (typeof obs.reset === 'function') {
          try {
            obs.reset();
          } catch (err) {
            console.warn('[Reset] observation reset threw error', err);
          }
        }
        if (obs.history) {
          // Re-initialize history with current values (all zeros since we reset)
          obs.history_initialized = false;
        }
      }
    }
    
    // Reset recurrent state
    if (this.policy) {
      this.inputDict = this.policy.initInput();
    }
    this.adapt_hx.fill(0);
    this.rpy.set(0, 0, 0);
    
    // Reset simulation counters
    this.simStepCount = 0;
    
    // Forward kinematics to update visualization
    mujoco.mj_forward(this.model, this.data);
    
    console.log("[Reset] complete");
  }

  addPolicyLoadedListener(listener) {
    if (typeof listener !== 'function') {
      return;
    }
    if (!this.policyLoadedListeners.includes(listener)) {
      this.policyLoadedListeners.push(listener);
    }
  }

  notifyPolicyLoaded() {
    for (const listener of this.policyLoadedListeners) {
      try {
        listener(this);
      } catch (err) {
        console.warn('[PolicyLoadedListener] listener threw error', err);
      }
    }
  }

  async replayMotion() {
    if (!this.observations) {
      return false;
    }
    let replayed = false;
    for (const obsList of Object.values(this.observations)) {
      for (const obsInstance of obsList) {
        if (typeof obsInstance?.replayMotion === 'function') {
          try {
            const result = await obsInstance.replayMotion();
            replayed = replayed || Boolean(result);
          } catch (err) {
            console.warn('[ReplayMotion] observation replay failed', err);
          }
        }
      }
    }
    if (!replayed) {
      console.warn('[ReplayMotion] no motion-enabled observations available to replay');
    }
    return replayed;
  }

  async loadPolicy(policyPath, onnxOverride = null) {
    console.log("[LoadPolicy] begin", policyPath, onnxOverride ? `(onnx override: ${onnxOverride})` : '');
    
    try {
      // Wait until inference is not running
      while (this.isInferencing) {
        await new Promise(resolve => setTimeout(resolve, 10));
      }

      let config;
      this.policyLoading = true;
      this.observationsReady = false;
      this.observations = {};
      this.motionObservationsPresent = false;
      this.policy = null;
      this.inputDict = {};
      const motionOverride = this.params.motionDataset ?? this.motionDatasetOverride ?? null;
    
    // Check if it's a YAML or JSON file
    if (policyPath.endsWith('.yaml') || policyPath.endsWith('.yml')) {
      // Parse YAML and convert to policy config
      console.log("[LoadPolicy] parsing YAML", policyPath);
      const yamlConfig = await parseYAMLConfig(policyPath);
      if (motionOverride) {
        yamlConfig.__motionDataset = motionOverride;
      }
      
      // Determine ONNX path based on YAML filename
      const onnxPath = onnxOverride;
      if (!onnxPath) {
        throw new Error(`ONNX path not provided for policy: ${policyPath}`);
      }
      
      config = yamlConfigToPolicyConfig(yamlConfig, onnxPath);
      this.params.policy = policyPath;
      this.params.policyOnnx = onnxPath;
      console.log("[LoadPolicy] YAML parsed; onnx:", onnxPath);
    } else {
      // Load policy config from JSON (legacy)
      const response = await fetch(policyPath);
      config = await response.json();
    }
    this.motionDatasetOverride = motionOverride;

    // Initialize ONNX model (defer assigning to this.policy until session is ready)
    const policy = new ONNXModule(config.onnx);
    console.log("[LoadPolicy] initializing ONNX session");
    await policy.init();
    console.log("[LoadPolicy] ONNX session ready");
    this.adapt_hx.fill(0);
    this.rpy.set(0, 0, 0);

    console.log("[LoadPolicy] resetting simulation data");
    mujoco.mj_resetData(this.model, this.data);
    
    // Initialize action buffers before constructing observations so PrevActions has correct dims
    this.numActions = config.num_joints || this.model.nu;
    
    // Initialize lastActions to zeros (normalized action space)
    // The network outputs actions in normalized space, which then get scaled:
    // target_position = action_scale * action + action_offset
    // So action=0 means target=action_offset (the default pose)
    this.lastActions = new Float32Array(this.numActions).fill(0);
    
    // Initialize action buffer with zeros
    this.actionBuffer = new Array(4).fill().map(() => new Float32Array(this.numActions).fill(0));

    // Helper function to create observation instance
    const createObservation = async (obsConfig) => {
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

      const instance = new ObsClass(this.model, this.data, this, kwargs);
      if (instance && instance.ready && typeof instance.ready.then === 'function') {
        try {
          await instance.ready;
        } catch (readyError) {
          console.warn('[LoadPolicy] observation readiness promise rejected', readyError);
        }
      }
      return instance;
    };

    // Set up observations based on config
    this.observations = {};
    console.log("[LoadPolicy] building observations");
    for (const [key, obsList] of Object.entries(config.obs_config)) {
      const instances = [];
      for (const obsConfig of obsList) {
        instances.push(await createObservation(obsConfig));
      }
      this.observations[key] = instances;
    }
    this.motionObservationsPresent = Object.values(this.observations).some(obsList =>
      obsList.some(obsInstance => obsInstance && obsInstance.motionLoader)
    );
    this.observationsReady = true;
    this.motionProgress = 0;
    const initialMotionObs = this.getMotionObservation();
    if (initialMotionObs && initialMotionObs.motionLoader) {
      this.motionDuration = initialMotionObs.motionLoader.duration || 0;
      const startTime = Math.max(0, Math.min(this.motionDuration, initialMotionObs.motionTime || 0));
      this.motionUISeconds = Math.round(startTime * 10) / 10;
      this.updateMotionSliderRange(this.motionDuration);
    } else {
      this.motionDuration = 0;
      this.motionUISeconds = 0;
      this.updateMotionSliderRange(1);
    }
    this.motionStartTimeMS = typeof performance !== 'undefined' ? performance.now() : null;

    // Handle per-joint stiffness/damping arrays or single values
    if (config.stiffness_array && config.stiffness_array.length > 0) {
      this.jntKp = new Float32Array(config.stiffness_array);
    } else {
      this.jntKp = new Float32Array(this.numActions).fill(config.stiffness);
    }
    
    if (config.damping_array && config.damping_array.length > 0) {
      this.jntKd = new Float32Array(config.damping_array);
    } else {
      this.jntKd = new Float32Array(this.numActions).fill(config.damping);
    }
    
    // Store action scale and offset
    if (Array.isArray(config.action_scale)) {
      this.action_scale = new Float32Array(config.action_scale);
    } else {
      this.action_scale = new Float32Array(this.numActions).fill(config.action_scale);
    }
    
    this.action_offset = config.action_offset ? new Float32Array(config.action_offset) : new Float32Array(this.numActions).fill(0);
    this.joint_ids_map = config.joint_ids_map || null;
    this.defaultJpos = config.default_joint_pos ? new Float32Array(config.default_joint_pos) : null;
    
    this.control_type = config.control_type ?? "joint_position";
    
    // Initialize robot to default pose (critical for policy to work correctly)
    if (this.defaultJpos && this.joint_ids_map) {
      console.log("Initializing robot to default pose...");
      for (let i = 0; i < this.numActions; i++) {
        const joint_idx = this.joint_ids_map[i];
        const joint_name = this.jointNamesIsaac[joint_idx];
        const mjc_idx = this.jointNamesMJC.indexOf(joint_name);
        if (mjc_idx >= 0) {
          const qpos_adr = this.model.jnt_qposadr[mjc_idx];
          this.data.qpos[qpos_adr] = this.defaultJpos[i];
        }
      }
      mujoco.mj_forward(this.model, this.data);
      console.log("Robot initialized to default pose");
    }
    
    // Assign policy only after it has an initialized session
    this.policy = policy;
    console.log("[LoadPolicy] policy assigned; computing decimation");
    // Initialize recurrent inputs (is_init, adapt_hx)
    this.inputDict = this.policy.initInput();
    
    // Calculate control decimation: how many physics steps per policy step
    // step_dt is the policy rate (e.g., 0.02s = 50Hz)
    // model.opt.timestep is the physics timestep (e.g., 0.002s = 500Hz or 0.005s = 200Hz)
    const physics_dt = this.model?.opt?.timestep || 0.005; // Default to 0.005s (200Hz) if not available
    this.control_decimation = Math.round(config.step_dt / physics_dt);
    console.log("[LoadPolicy] loaded: actions=" + this.numActions + ", control_rate=" + config.step_dt + "s, physics_dt=" + physics_dt + "s, decimation=" + this.control_decimation);
    
      // Reset recurrent inputs and invalidate any in-flight inference
      this.inputDict = this.policy.initInput();
      this.inferenceGen++;
    this.policyLoading = false;
    this.notifyPolicyLoaded();
      console.log("[LoadPolicy] complete");
    } catch (error) {
      console.error("[LoadPolicy] ERROR", error);
      console.error("Stack trace:", error.stack);
      this.policyLoading = false;
      throw error; // Re-throw to see it in console
    }
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize( window.innerWidth, window.innerHeight );
  }

  render(timeMS) {
    this.controls.update();

    if (!this.params["paused"]) {
      let timestep = this.model.opt?.timestep ?? 0.005;
      if (timeMS - this.mujoco_time > 35.0) { this.mujoco_time = timeMS; }
      while (this.mujoco_time < timeMS) {

        // Jitter the control state with gaussian random noise
        if (this.params["ctrlnoisestd"] > 0.0) {
          let rate  = Math.exp(-timestep / Math.max(1e-10, this.params["ctrlnoiserate"]));
          let scale = this.params["ctrlnoisestd"] * Math.sqrt(1 - rate * rate);
          let currentCtrl = this.data.ctrl;
          for (let i = 0; i < currentCtrl.length; i++) {
            currentCtrl[i] = rate * currentCtrl[i] + scale * standardNormal();
            this.params["Actuator " + i] = currentCtrl[i];
          }
        }

        // Clear old perturbations, apply new ones.
        for (let i = 0; i < this.data.qfrc_applied.length; i++) { this.data.qfrc_applied[i] = 0.0; }
        let dragged = this.dragStateManager.physicsObject;
        if (dragged && dragged.bodyID) {
          for (let b = 0; b < this.model.nbody; b++) {
            if (this.bodies[b]) {
              getPosition  (this.data.xpos , b, this.bodies[b].position);
              getQuaternion(this.data.xquat, b, this.bodies[b].quaternion);
              this.bodies[b].updateWorldMatrix();
            }
          }
          let bodyID = dragged.bodyID;
          this.dragStateManager.update(); // Update the world-space force origin
          const dragOffset = this.dragStateManager.currentWorld.clone().sub(this.dragStateManager.worldHit);
          const force = toMujocoPos(dragOffset.multiplyScalar(20));
          const point = toMujocoPos(this.dragStateManager.worldHit.clone());
          mujoco.mj_applyFT(
            this.model,
            this.data,
            [force.x, force.y, force.z],
            [0, 0, 0],
            [point.x, point.y, point.z],
            bodyID,
            this.data.qfrc_applied
          );

        }

        // Run policy and apply control at control rate (decimated)
        // Only run inference every control_decimation steps to match training (e.g., 50Hz policy with 200Hz physics)
        const shouldRunInference = this.policy && this.policy.session && !this.isInferencing && 
                                   (this.control_decimation > 0) && 
                                   (this.simStepCount % this.control_decimation === 0);
        
        if (shouldRunInference) {
          if (this.policyLoading || !this.observationsReady) {
            // Skip inference until policy + observations are ready
            if ((this.simStepCount % 100) === 0) {
              console.warn('[Inference] policy not ready; skipping step');
            }
            continue;
          }
          if (!this.observations || !Array.isArray(this.observations.obs) || !this.observations.obs.length) {
            if ((this.simStepCount % 200) === 0) {
              console.warn('[Inference] observations unavailable; skipping step');
            }
            continue;
          }
          // Update base quat/euler
          const q = this.data.qpos.subarray(3, 7);
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
              if (obs_key === 'obs') debugParts.push({ name: fn.constructor.name, len: arr.length });
            }
            // Check observations for NaN/Inf
            const obsHasNaN = flat.some(v => isNaN(v));
            const obsHasInf = flat.some(v => !isFinite(v));
            
            if (obsHasNaN || obsHasInf) {
              console.error('=== Observation Error (step', this.simStepCount, ') ===');
              console.error('Obs has NaN?', obsHasNaN, 'has Inf?', obsHasInf);
              console.error('First 10 obs values:', flat.slice(0, 10));
            }
            this.inputDict[obs_key] = new ort.Tensor('float32', flat, [1, flat.length]);
          }
          // Run inference asynchronously to avoid await inside render loop
          const runGen = this.inferenceGen;
          this.isInferencing = true;
          const inputCopy = { ...this.inputDict };
          this.policy.runInference(inputCopy).then(([result, carry]) => {
            if (runGen !== this.inferenceGen) { this.isInferencing = false; return; }
            const action_tensor = result['actions'];
            const action = action_tensor.data;
            
            // Validate action dimensions
            if (action.length !== this.numActions) {
              console.error(`Action dimension mismatch! Expected ${this.numActions}, got ${action.length}`);
              this.isInferencing = false;
              return;
            }
            
            // Check for NaN in actions
            const hasNaN = Array.from(action).some(v => isNaN(v));
            const hasInf = Array.from(action).some(v => !isFinite(v));
            
            if (hasNaN || hasInf) {
              console.error('ERROR: NaN or Inf in actions at step', this.simStepCount);
            }
            
            // Store raw actions for observation feedback
            for (let i = 0; i < this.lastActions.length; i++) {
              this.lastActions[i] = action[i];
            }
            
            this.inputDict = carry;
            this.isInferencing = false;
          }).catch((e) => { console.error('Inference error', e); this.isInferencing = false; });
        }
        
        // Apply PD control at EVERY physics step (not just when running inference)
        // The policy updates at 50Hz but PD control runs at 200Hz
        if (this.policy && this.lastActions) {
          for (let i = 0; i < this.numActions; i++) {
            // Use joint_ids_map if available (for G1), otherwise use direct mapping
            const joint_idx = this.joint_ids_map ? this.joint_ids_map[i] : i;
            const joint_name = this.jointNamesIsaac[joint_idx];
            const mjc_idx = this.jointNamesMJC.indexOf(joint_name);
            
            if (mjc_idx === -1) {
              if ((this.simStepCount % 50) === 0 && i < 3) {
                console.error(`Control: Cannot find joint "${joint_name}" (action ${i}, isaac idx ${joint_idx})`);
              }
              continue;
            }
            
            const qpos_adr = this.model.jnt_qposadr[mjc_idx];
            const qvel_adr = this.model.jnt_dofadr[mjc_idx];
            
            // Actuators are named without the "_joint" suffix
            const actuator_name = joint_name.replace('_joint', '');
            const ctrl_adr = this.actuatorNamesMJC.indexOf(actuator_name);
            
            if (ctrl_adr === -1) {
              if ((this.simStepCount % 50) === 0 && i < 3) {
                console.error(`Control: Cannot find actuator "${actuator_name}" for joint "${joint_name}"`);
                console.log(`Searching for: "${actuator_name}" (length ${actuator_name.length})`);
                console.log('Available actuators:', this.actuatorNamesMJC);
                // Try to find similar names
                const similar = this.actuatorNamesMJC.filter(name => name.includes(actuator_name.split('_')[0]));
                console.log('Similar actuators:', similar);
              }
              continue;
            }
            
            const action_val = this.lastActions[i];
            const target = this.action_scale[i] * action_val + this.action_offset[i];
            const qpos_val = this.data.qpos[qpos_adr];
            const qvel_val = this.data.qvel[qvel_adr];
            const torque = this.jntKp[i] * (target - qpos_val) + this.jntKd[i] * (0 - qvel_val);
            
            // Check for NaN/Inf in control calculation
            if (!isFinite(torque)) {
              console.error(`Control NaN/Inf at action ${i} (${joint_name}):`, {
                action_val,
                action_scale: this.action_scale[i],
                action_offset: this.action_offset[i],
                target,
                qpos_val,
                qvel_val,
                kp: this.jntKp[i],
                kd: this.jntKd[i],
                torque
              });
            }
            
            this.data.ctrl[ctrl_adr] = torque;
          }
        }
        
        // Apply external impulses if requested
        if (this.params["impulse_remain_time"] > 0 && this.pelvis_body_id !== undefined) {
          const force = new THREE.Vector3(0, 65, 0);
          const point = new THREE.Vector3(0, 0, 0);
          getPosition(this.data.xpos, this.pelvis_body_id, point, false);
          mujoco.mj_applyFT(
            this.model,
            this.data,
            [force.x, force.y, force.z],
            [0, 0, 0],
            [point.x, point.y, point.z],
            this.pelvis_body_id,
            this.data.qfrc_applied
          );
          this.params["impulse_remain_time"] -= timestep;
        }
        mujoco.mj_step(this.model, this.data);
        this.simStepCount++;
        this.mujoco_time += timestep * 1000.0;
      }

    } else if (this.params["paused"]) {
      this.dragStateManager.update(); // Update the world-space force origin
      let dragged = this.dragStateManager.physicsObject;
      if (dragged && dragged.bodyID) {
        let b = dragged.bodyID;
        getPosition  (this.data.xpos , b, this.tmpVec , false); // Get raw coordinate from MuJoCo
        getQuaternion(this.data.xquat, b, this.tmpQuat, false); // Get raw coordinate from MuJoCo

        let offset = toMujocoPos(this.dragStateManager.currentWorld.clone()
          .sub(this.dragStateManager.worldHit).multiplyScalar(0.3));
        if (this.model.body_mocapid[b] >= 0) {
          // Set the root body's mocap position...
          console.log("Trying to move mocap body", b);
          let addr = this.model.body_mocapid[b] * 3;
          let pos  = this.data.mocap_pos;
          pos[addr+0] += offset.x;
          pos[addr+1] += offset.y;
          pos[addr+2] += offset.z;
        } else {
          // Set the root body's position directly...
          let root = this.model.body_rootid[b];
          let addr = this.model.jnt_qposadr[this.model.body_jntadr[root]];
          let pos  = this.data.qpos;
          pos[addr+0] += offset.x;
          pos[addr+1] += offset.y;
          pos[addr+2] += offset.z;

        }
      }

      mujoco.mj_forward(this.model, this.data);
    }

    if (this.motionObservationsPresent && this.replayControls && this.replayProgressController) {
      const motionObs = this.getMotionObservation();
      if (motionObs && motionObs.motionLoader) {
        const duration = motionObs.motionLoader.duration || 0;
        const currentTime = Math.max(0, Math.min(duration, motionObs.motionTime));
        const normalized = duration > 0 ? currentTime / duration : 0;
        const displayTime = Math.round(currentTime * 10) / 10;
        this.motionDuration = duration;
        this.motionProgress = normalized;
        this.motionUISeconds = displayTime;
        this.replayControls.replayTime = displayTime;
        this.updateMotionSliderRange(duration);
        if (!this._updatingReplaySlider) {
          this._updatingReplaySlider = true;
          this.replayProgressController.updateDisplay();
          this._updatingReplaySlider = false;
        }
      }
    }

    // Update body transforms.
    for (let b = 0; b < this.model.nbody; b++) {
      if (this.bodies[b]) {
        getPosition  (this.data.xpos , b, this.bodies[b].position);
        getQuaternion(this.data.xquat, b, this.bodies[b].quaternion);
        this.bodies[b].updateWorldMatrix();
      }
    }
    if (this.spotlight) {
      const pelvis = this.pelvis_body_id ?? 0;
      const pelvisPos = new THREE.Vector3();
      getPosition(this.data.xpos, pelvis, pelvisPos);
      this.spotlight.position.copy(pelvisPos).add(this.spotlightOffset);
      this.spotlightTarget.position.copy(pelvisPos).add(this.spotlightTargetOffset);
    }

    // Update light transforms.
    for (let l = 0; l < this.model.nlight; l++) {
      if (this.lights[l]) {
        getPosition(this.data.light_xpos, l, this.lights[l].position);
        getPosition(this.data.light_xdir, l, this.tmpVec);
        this.lights[l].lookAt(this.tmpVec.add(this.lights[l].position));
      }
    }

    // Update tendon transforms.
    let numWraps = 0;
    if (this.mujocoRoot && this.mujocoRoot.cylinders) {
      let mat = new THREE.Matrix4();
      for (let t = 0; t < this.model.ntendon; t++) {
        let startW = this.data.ten_wrapadr[t];
        let r = this.model.tendon_width[t];
        for (let w = startW; w < startW + this.data.ten_wrapnum[t] -1 ; w++) {
          let tendonStart = getPosition(this.data.wrap_xpos, w    , new THREE.Vector3());
          let tendonEnd   = getPosition(this.data.wrap_xpos, w + 1, new THREE.Vector3());
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

    // Update joint position displays in GUI
    if (this.updateJointPositions) {
      this.updateJointPositions();
    }

    // Render!
    this.renderer.render( this.scene, this.camera );
  }
}

let demo = new MuJoCoDemo();
await demo.init();
