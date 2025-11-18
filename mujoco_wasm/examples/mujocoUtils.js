import * as THREE from 'three';
import { Reflector  } from './utils/Reflector.js';
import { MuJoCoDemo } from './main.js';

export async function reloadFunc() {
  // Delete the old scene and load the new scene
  this.scene.remove(this.scene.getObjectByName("MuJoCo Root"));
  [this.model, this.data, this.bodies, this.lights] =
    await loadSceneFromURL(this.mujoco, this.params.scene, this);
  this.mujoco.mj_forward(this.model, this.data);
  for (let i = 0; i < this.updateGUICallbacks.length; i++) {
    this.updateGUICallbacks[i](this.model, this.data, this.params);
  }
}

/** @param {MuJoCoDemo} parentContext*/
export function setupGUI(parentContext) {

  // Make sure we reset the camera when the scene is changed or reloaded.
  parentContext.updateGUICallbacks.length = 0;
  parentContext.updateGUICallbacks.push((model, data, params) => {
    // TODO: Use free camera parameters from MuJoCo
    parentContext.camera.position.set(2.0, 1.7, 1.7);
    parentContext.controls.target.set(0, 0.7, 0);
    parentContext.controls.update(); });

  // Add task/policy/velocity controls at root to avoid nested sections.
  // Switching tasks loads the corresponding checkpoints and toggles velocity controls.
  parentContext.params.task = parentContext.params.task || 'Balance';
  parentContext.params.policyLabel = parentContext.params.policyLabel || 'Ours';
  const policyMap = {
    Balance: {
      "Ours": {
        yaml: "./examples/checkpoints/g1/balance/balance_deploy_state_projection.yaml",
        onnx: "./examples/checkpoints/g1/balance/balance_policy_state_projection.onnx",
      },
      "Baseline": {
        yaml: "./examples/checkpoints/g1/balance/balance_deploy_baseline.yaml",
        onnx: "./examples/checkpoints/g1/balance/balance_policy_baseline.onnx",
      }
    },
    Velocity: {
      "Ours": {
        yaml: "./examples/checkpoints/g1/velocity/velocity_deploy_ours.yaml",
        onnx: "./examples/checkpoints/g1/velocity/velocity_policy_ours.onnx",
      },
      "Baseline": {
        yaml: "./examples/checkpoints/g1/velocity/velocity_deploy_baseline.yaml",
        onnx: "./examples/checkpoints/g1/velocity/velocity_policy_baseline.onnx",
      }
    },
    Squat: {
      "Ours": {
        yaml: "./examples/checkpoints/g1/squat/squat_deploy_state_projection.yaml",
        onnx: "./examples/checkpoints/g1/squat/squat_policy_state_projection.onnx",
        motion: "./examples/checkpoints/g1/squat/squat_29dof.csv",
      },
      "Baseline": {
        yaml: "./examples/checkpoints/g1/squat/squat_deploy_baseline.yaml",
        onnx: "./examples/checkpoints/g1/squat/squat_policy_baseline.onnx",
        motion: "./examples/checkpoints/g1/squat/squat_29dof.csv",
      }
    },
    Single_Leg: {
      "Ours": {
        yaml: "./examples/checkpoints/g1/single_leg_2/single_leg_2_deploy_state_projection.yaml",
        onnx: "./examples/checkpoints/g1/single_leg_2/single_leg_2_policy_state_projection.onnx",
        motion: "./examples/checkpoints/g1/single_leg_2/nezha_29dof.csv",
      },
      "Baseline": {
        yaml: "./examples/checkpoints/g1/single_leg_2/single_leg_2_deploy_baseline.yaml",
        onnx: "./examples/checkpoints/g1/single_leg_2/single_leg_2_policy_baseline.onnx",
        motion: "./examples/checkpoints/g1/single_leg_2/nezha_29dof.csv",
      }
    },
    Swallow_Balance: {
      "Ours": {
        yaml: "./examples/checkpoints/g1/swallow_balance/swallow_balance_deploy_state_projection.yaml",
        onnx: "./examples/checkpoints/g1/swallow_balance/swallow_balance_policy_state_projection.onnx",
        motion: "./examples/checkpoints/g1/swallow_balance/swallow_balance_29dof.csv",
      },
      "Baseline": {
        yaml: "./examples/checkpoints/g1/swallow_balance/swallow_balance_deploy_baseline.yaml",
        onnx: "./examples/checkpoints/g1/swallow_balance/swallow_balance_policy_baseline.onnx",
        motion: "./examples/checkpoints/g1/swallow_balance/swallow_balance_29dof.csv",
      }
    },
  };

  const getPolicyLabelsForTask = (task) => {
    const available = policyMap[task] || {};
    return Object.keys(available);
  };

  const ensurePolicyLabel = (task, label) => {
    const available = policyMap[task] || {};
    if (label && (label in available)) {
      return label;
    }
    if ('Ours' in available) {
      return 'Ours';
    }
    const labels = getPolicyLabelsForTask(task);
    return labels.length > 0 ? labels[0] : null;
  };

  let policyController;
  let replayMotionCtrl;

  function insertControllerAfter(controller, referenceController) {
    if (!controller || !referenceController) { return; }
    const ul = parentContext.gui && parentContext.gui.__ul;
    if (!ul) { return; }
    const controllerLi = controller.domElement?.parentElement;
    const referenceLi = referenceController.domElement?.parentElement;
    if (!controllerLi || !referenceLi) { return; }
    const currentNext = referenceLi.nextSibling;
    if (currentNext === controllerLi) { return; }
    ul.insertBefore(controllerLi, currentNext);
  }

  const updateReplayMotionVisibility = () => {
    if (!replayMotionCtrl || !replaySlider) { return; }
    ensureControlOrdering();
    const hasMotionReplay = !!parentContext.motionObservationsPresent;
    if (hasMotionReplay) {
      replayMotionCtrl.show();
      replaySlider.show();
      if (parentContext.replayControls && parentContext.replayProgressController) {
        parentContext.replayControls.replayTime = parentContext.motionUISeconds ?? 0;
        parentContext._updatingReplaySlider = true;
        parentContext.replayProgressController.updateDisplay();
        parentContext._updatingReplaySlider = false;
      }
    } else {
      replayMotionCtrl.hide();
      replaySlider.hide();
    }
  };

  async function applyPolicySelection(label, { fromController = false } = {}) {
    const task = parentContext.params.task;
    const available = policyMap[task];
    if (!available) {
      console.warn('[Policy Change] no policies found for task', task);
      return;
    }
    const normalizedLabel = ensurePolicyLabel(task, label);
    if (!normalizedLabel) {
      console.warn('[Policy Change] unable to determine policy label for task', task);
      return;
    }
    const entry = available[normalizedLabel];
    if (!entry) {
      console.warn('[Policy Change] missing entry for task', task, 'label', normalizedLabel);
      return;
    }
    const { yaml: yamlPath, onnx: onnxPath, motion: motionPath } = entry;
    if (!yamlPath || !onnxPath) {
      console.warn('[Policy Change] missing YAML/ONNX paths for task', task, 'label', normalizedLabel, entry);
      return;
    }

    parentContext.params.policyLabel = normalizedLabel;
    parentContext.params.policy = yamlPath;
    parentContext.params.policyOnnx = onnxPath;
    parentContext.params.motionDataset = motionPath ?? null;
    parentContext.motionDatasetOverride = motionPath ?? null;
    if (!fromController && policyController) {
      policyController.updateDisplay();
    }
    parentContext.motionObservationsPresent = false;
    parentContext.motionUISeconds = 0;
    updateReplayMotionVisibility();

    console.log('[Policy Change] BEGIN', { task, label: normalizedLabel, yaml: yamlPath, onnx: onnxPath });
    try {
      if (typeof parentContext.resetSimulation === 'function') {
        console.log('[Policy Change] calling resetSimulation');
        parentContext.resetSimulation();
        console.log('[Policy Change] resetSimulation completed');
      } else {
        console.log('[Policy Change] resetSimulation not available');
      }
      console.log('[Policy Change] calling loadPolicy', yamlPath);
      await parentContext.loadPolicy(yamlPath, onnxPath);
      console.log('[Policy Change] SUCCESS loaded policy', yamlPath, onnxPath);
      updateReplayMotionVisibility();
      if (parentContext.replayControls && parentContext.replayProgressController) {
        parentContext.replayControls.replayTime = parentContext.motionUISeconds ?? 0;
        parentContext._updatingReplaySlider = true;
        parentContext.replayProgressController.updateDisplay();
        parentContext._updatingReplaySlider = false;
      }
    } catch (e) {
      console.error('[Policy Change] ERROR while switching policy', e);
    }
  }

  const taskController = parentContext.gui.add(parentContext.params, 'task', {
    "Balance": "Balance",
    "Velocity": "Velocity",
    "Squat": "Squat",
    "Single Leg": "Single_Leg",
    "Swallow Balance": "Swallow_Balance"
  }).name('Task');

  const ensurePolicyControllerPosition = () => {
    if (!taskController || !policyController) { return; }
    const controllers = parentContext.gui && Array.isArray(parentContext.gui.__controllers)
      ? parentContext.gui.__controllers
      : null;
    if (!controllers) { return; }
    const taskIndex = controllers.indexOf(taskController);
    const policyIndex = controllers.indexOf(policyController);
    if (taskIndex === -1 || policyIndex === -1) { return; }
    if (policyIndex !== taskIndex + 1) {
      controllers.splice(policyIndex, 1);
      controllers.splice(taskIndex + 1, 0, policyController);
    }

    const ul = parentContext.gui && parentContext.gui.__ul ? parentContext.gui.__ul : null;
    const taskLi = taskController.domElement?.parentElement;
    const policyLi = policyController.domElement?.parentElement;
    if (!taskLi || !policyLi || !ul) { return; }

    ul.insertBefore(policyLi, taskLi.nextSibling);
  };

  const labelsToOptions = (labels) => labels.reduce((acc, label) => {
    acc[label] = label;
    return acc;
  }, {});

  const updatePolicyControllerOptions = (labels) => {
    if (!policyController) { return; }
    const optionsObj = labelsToOptions(labels);
    policyController.__options = optionsObj;
    const select = policyController.domElement?.querySelector('select');
    if (!select) { return; }
    select.innerHTML = '';
    for (const label of labels) {
      const option = document.createElement('option');
      option.value = label;
      option.textContent = label;
      select.appendChild(option);
    }
    policyController.__select = select;
    ensurePolicyControllerPosition();
  };

  // Initialize policy selection state before creating controller
  parentContext.params.policyLabel = ensurePolicyLabel(parentContext.params.task, parentContext.params.policyLabel);
  if (parentContext.params.policyLabel) {
    const initialYaml = policyMap[parentContext.params.task]?.[parentContext.params.policyLabel];
    if (initialYaml) {
      const initial = policyMap[parentContext.params.task]?.[parentContext.params.policyLabel];
      if (initial) {
        parentContext.params.policy = initial.yaml;
        parentContext.params.policyOnnx = initial.onnx;
        parentContext.params.motionDataset = initial.motion ?? null;
        parentContext.motionDatasetOverride = initial.motion ?? null;
      }
    }
  }

  const initialLabels = getPolicyLabelsForTask(parentContext.params.task);
  policyController = parentContext.gui.add(
    parentContext.params,
    'policyLabel',
    labelsToOptions(initialLabels)
  ).name('Policy');

  policyController.onChange(async (label) => {
    await applyPolicySelection(label, { fromController: true });
  });

  ensurePolicyControllerPosition();

  // Velocity controls (visible only in Velocity task)
  parentContext.params.command_vel_x = parentContext.params.command_vel_x ?? 0.0;
  const velocityXCtrl = parentContext.gui.add(parentContext.params, 'command_vel_x', -0.5, 1.0).name('velocity');

  const replayControls = {
    replayTime: 0
  };
  const replaySlider = parentContext.gui
    .add(replayControls, 'replayTime', 0, 1, 0.1)
    .name('Motion Time (s)')
    .listen();
  if (typeof replaySlider.disable === 'function') {
    replaySlider.disable();
  }
  const sliderRange = replaySlider.domElement?.querySelector('input[type="range"]');
  if (sliderRange) {
    sliderRange.setAttribute('disabled', '');
    sliderRange.style.pointerEvents = 'none';
    sliderRange.style.opacity = '0.7';
  }
  const sliderNumber = replaySlider.domElement?.querySelector('input[type="number"]');
  if (sliderNumber) {
    sliderNumber.setAttribute('disabled', '');
    sliderNumber.style.pointerEvents = 'none';
    sliderNumber.style.opacity = '0.7';
  }
  replaySlider.hide();

  parentContext.replayProgressController = replaySlider;
  parentContext.replayControls = replayControls;

  const replayMotionAction = {
    replayMotion: async () => {
      if (parentContext._updatingReplaySlider) { return; }
      parentContext._updatingReplaySlider = true;
      replayControls.replayTime = 0;
      replaySlider.updateDisplay();
      if (typeof parentContext.setMotionProgress === 'function') {
        await parentContext.setMotionProgress(0, { syncState: false });
      }
      parentContext._updatingReplaySlider = false;
    }
  };
  replayMotionCtrl = parentContext.gui.add(replayMotionAction, 'replayMotion').name('Replay Motion');
  replayMotionCtrl.hide();
  parentContext.replayMotionController = replayMotionCtrl;
  function ensureControlOrdering() {
    if (!policyController) { return; }
    if (velocityXCtrl) {
      insertControllerAfter(velocityXCtrl, policyController);
    }
    const replayParent = velocityXCtrl ?? policyController;
    if (replaySlider) {
      insertControllerAfter(replaySlider, replayParent);
    }
    const lastReference = replaySlider ?? replayParent;
    if (replayMotionCtrl) {
      insertControllerAfter(replayMotionCtrl, lastReference);
    }
  }
  ensureControlOrdering();

  function updateVelocityControlsVisibility() {
    const isVelocity = parentContext.params.task === 'Velocity';
    if (isVelocity) { velocityXCtrl.show(); }
    else { velocityXCtrl.hide(); }
    ensureControlOrdering();
  }

  function setPolicyOptionsForTask(task) {
    if (!policyController) { return null; }
    const labels = getPolicyLabelsForTask(task);
    if (!labels.length) {
      return null;
    }
    updatePolicyControllerOptions(labels);
    parentContext.params.policyLabel = ensurePolicyLabel(task, parentContext.params.policyLabel);
    policyController.updateDisplay();
    ensurePolicyControllerPosition();
    return parentContext.params.policyLabel;
  }

  taskController.onChange(async () => {
    const task = parentContext.params.task;
    const label = setPolicyOptionsForTask(task);
    console.log('[Task Change] applying policy for task', task, 'label', label);
    await applyPolicySelection(label, { fromController: false });
    console.log('[Task Change] policy loaded for task', task);
    updateVelocityControlsVisibility();
    updateReplayMotionVisibility();
  });

  // Initialize defaults according to current task
  const initialLabel = setPolicyOptionsForTask(parentContext.params.task);
  if (initialLabel) {
    const initialEntry = policyMap[parentContext.params.task][initialLabel];
    parentContext.params.policy = initialEntry?.yaml;
    parentContext.params.policyOnnx = initialEntry?.onnx;
    parentContext.params.motionDataset = initialEntry?.motion ?? null;
    parentContext.motionDatasetOverride = initialEntry?.motion ?? null;
    policyController.updateDisplay();
  }
  updateVelocityControlsVisibility();
  updateReplayMotionVisibility();
  if (parentContext.replayControls) {
    parentContext.replayControls.replayTime = 0;
    if (parentContext.replayProgressController) {
      parentContext._updatingReplaySlider = true;
      parentContext.replayProgressController.updateDisplay();
      parentContext._updatingReplaySlider = false;
    }
  }

  // Fix robust behavior: zero command velocity
  parentContext.params.command_vel_x = 0.0;
  parentContext.params.command_vel_z = 0.0;
  parentContext.params.command_vel_yaw = 0.0;
  parentContext.params.use_setpoint = false;

  // Add a help menu.
  // Parameters:
  //  Name: "Help".
  //  When pressed, a help menu is displayed in the top left corner. When pressed again
  //  the help menu is removed.
  //  Can also be triggered by pressing F1.
  // Has a dark transparent background.
  // Has two columns: one for putting the action description, and one for the action key trigger.keyframeNumber
  let keyInnerHTML = '';
  let actionInnerHTML = '';
  const displayHelpMenu = () => {
    if (parentContext.params.help) {
      const helpMenu = document.createElement('div');
      helpMenu.style.position = 'absolute';
      helpMenu.style.top = '10px';
      helpMenu.style.left = '10px';
      helpMenu.style.color = 'white';
      helpMenu.style.font = 'normal 18px Arial';
      helpMenu.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
      helpMenu.style.padding = '10px';
      helpMenu.style.borderRadius = '10px';
      helpMenu.style.display = 'flex';
      helpMenu.style.flexDirection = 'column';
      helpMenu.style.alignItems = 'center';
      helpMenu.style.justifyContent = 'center';
      helpMenu.style.width = '400px';
      helpMenu.style.height = '400px';
      helpMenu.style.overflow = 'auto';
      helpMenu.style.zIndex = '1000';

      const helpMenuTitle = document.createElement('div');
      helpMenuTitle.style.font = 'bold 24px Arial';
      helpMenuTitle.innerHTML = '';
      helpMenu.appendChild(helpMenuTitle);

      const helpMenuTable = document.createElement('table');
      helpMenuTable.style.width = '100%';
      helpMenuTable.style.marginTop = '10px';
      helpMenu.appendChild(helpMenuTable);

      const helpMenuTableBody = document.createElement('tbody');
      helpMenuTable.appendChild(helpMenuTableBody);

      const helpMenuRow = document.createElement('tr');
      helpMenuTableBody.appendChild(helpMenuRow);

      const helpMenuActionColumn = document.createElement('td');
      helpMenuActionColumn.style.width = '50%';
      helpMenuActionColumn.style.textAlign = 'right';
      helpMenuActionColumn.style.paddingRight = '10px';
      helpMenuRow.appendChild(helpMenuActionColumn);

      const helpMenuKeyColumn = document.createElement('td');
      helpMenuKeyColumn.style.width = '50%';
      helpMenuKeyColumn.style.textAlign = 'left';
      helpMenuKeyColumn.style.paddingLeft = '10px';
      helpMenuRow.appendChild(helpMenuKeyColumn);

      const helpMenuActionText = document.createElement('div');
      helpMenuActionText.innerHTML = actionInnerHTML;
      helpMenuActionColumn.appendChild(helpMenuActionText);

      const helpMenuKeyText = document.createElement('div');
      helpMenuKeyText.innerHTML = keyInnerHTML;
      helpMenuKeyColumn.appendChild(helpMenuKeyText);

      // Close buttom in the top.
      const helpMenuCloseButton = document.createElement('button');
      helpMenuCloseButton.innerHTML = 'Close';
      helpMenuCloseButton.style.position = 'absolute';
      helpMenuCloseButton.style.top = '10px';
      helpMenuCloseButton.style.right = '10px';
      helpMenuCloseButton.style.zIndex = '1001';
      helpMenuCloseButton.onclick = () => {
        helpMenu.remove();
      };
      helpMenu.appendChild(helpMenuCloseButton);

      document.body.appendChild(helpMenu);
    } else {
      document.body.removeChild(document.body.lastChild);
    }
  }
  document.addEventListener('keydown', (event) => {
    if (event.key === 'F1') {
      parentContext.params.help = !parentContext.params.help;
      displayHelpMenu();
      event.preventDefault();
    }
  });
  keyInnerHTML += 'F1<br>';
  actionInnerHTML += 'Help<br>';

  let simulationFolder = parentContext.gui.addFolder("Simulation");

  // Add pause simulation checkbox.
  // Parameters:
  //  Under "Simulation" folder.
  //  Name: "Pause Simulation".
  //  When paused, a "pause" text in white is displayed in the top left corner.
  //  Can also be triggered by pressing the spacebar.
  const pauseSimulation = simulationFolder.add(parentContext.params, 'paused').name('Pause Simulation');
  pauseSimulation.onChange((value) => {
    if (value) {
      const pausedText = document.createElement('div');
      pausedText.style.position = 'absolute';
      pausedText.style.top = '10px';
      pausedText.style.left = '10px';
      pausedText.style.color = 'white';
      pausedText.style.font = 'normal 18px Arial';
      pausedText.innerHTML = 'pause';
      parentContext.container.appendChild(pausedText);
    } else {
      parentContext.container.removeChild(parentContext.container.lastChild);
    }
  });
  document.addEventListener('keydown', (event) => {
    if (event.code === 'Space') {
      parentContext.params.paused = !parentContext.params.paused;
      pauseSimulation.setValue(parentContext.params.paused);
      event.preventDefault();
    }
  });
  actionInnerHTML += 'Play / Pause<br>';
  keyInnerHTML += 'Space<br>';

  const impulseSeconds = 0.45;
  simulationFolder.add({ impulse: () => { parentContext.params["impulse_remain_time"] = impulseSeconds; } }, 'impulse').name('Impulse');
  document.addEventListener('keydown', (event) => {
    if (event.code === 'KeyI') { parentContext.params["impulse_remain_time"] = impulseSeconds; event.preventDefault(); }
  });
  actionInnerHTML += 'Impulse<br>';
  keyInnerHTML += 'I<br>';


  // Add reset simulation button.
  // Parameters:
  //  Under "Simulation" folder.
  //  Name: "Reset".
  //  When pressed, resets the simulation to the initial state.
  //  Can also be triggered by pressing backspace.
  const resetSimulation = () => {
    // Use the demo's resetSimulation method if available (properly resets policy state)
    if (typeof parentContext.resetSimulation === 'function') {
      parentContext.resetSimulation();
    } else if (parentContext.mujoco && parentContext.model && parentContext.data) {
      parentContext.mujoco.mj_resetData(parentContext.model, parentContext.data);
      parentContext.mujoco.mj_forward(parentContext.model, parentContext.data);
    }
  };
  simulationFolder.add({reset: () => { resetSimulation(); }}, 'reset').name('Reset');
  document.addEventListener('keydown', (event) => {
    if (event.code === 'Backspace') { resetSimulation(); event.preventDefault(); }});
  actionInnerHTML += 'Reset simulation<br>';
  keyInnerHTML += 'Backspace<br>';

  let textDecoder = new TextDecoder("utf-8");
  let nullChar    = textDecoder.decode(new ArrayBuffer(1));

  // Add joint position displays (read-only, shows actual joint angles)
  // Only show the 23 actuated joints from the policy
  let jointFolder = simulationFolder.addFolder("Joint Positions");
  const addJointPositions = (model, data, params) => {
    let jointGUIs = [];
    
    // List of non-actuated joints to exclude (6 joints)
    const excludedJoints = [
      "left_wrist_pitch_joint",
      "left_wrist_yaw_joint", 
      "right_wrist_pitch_joint",
      "right_wrist_yaw_joint",
      "waist_roll_joint",
      "waist_pitch_joint"
    ];
    
    for (let i = 0; i < model.njnt; i++) {
      let name = textDecoder.decode(
        model.names.subarray(model.name_jntadr[i])).split(nullChar)[0];
      
      // Skip the floating base joint and non-actuated joints
      if (name === "floating_base_joint" || excludedJoints.includes(name)) {
        continue;
      }
      
      let qpos_adr = model.jnt_qposadr[i];
      let joint_range = model.jnt_range;
      let has_limits = model.jnt_limited[i];
      
      // Create a params entry for this joint position
      let paramName = name + "_pos";
      parentContext.params[paramName] = data.qpos[qpos_adr];
      
      // Add GUI controller - make it listen to updates
      let min_val = has_limits ? joint_range[2 * i] : -Math.PI;
      let max_val = has_limits ? joint_range[2 * i + 1] : Math.PI;
      let jointGUI = jointFolder.add(parentContext.params, paramName, min_val, max_val).name(name).listen().disable();
      jointGUIs.push({ gui: jointGUI, qpos_adr: qpos_adr, paramName: paramName });
    }
    return jointGUIs;
  };
  let jointGUIs = addJointPositions(parentContext.model, parentContext.data, parentContext.params);
  
  // Update joint positions in the render loop
  parentContext.updateJointPositions = () => {
    for (let jointInfo of jointGUIs) {
      parentContext.params[jointInfo.paramName] = parentContext.data.qpos[jointInfo.qpos_adr];
    }
  };
  
  parentContext.updateGUICallbacks.push((model, data, params) => {
    for (let jointInfo of jointGUIs) {
      jointInfo.gui.destroy();
    }
    jointGUIs = addJointPositions(model, data, parentContext.params);
  });
  jointFolder.close();

  // Add function that resets the camera to the default position.
  // Can be triggered by pressing ctrl + A.
  document.addEventListener('keydown', (event) => {
    if (event.ctrlKey && event.code === 'KeyA') {
      // TODO: Use free camera parameters from MuJoCo
      parentContext.camera.position.set(2.0, 1.7, 1.7);
      parentContext.controls.target.set(0, 0.7, 0);
      parentContext.controls.update(); 
      event.preventDefault();
    }
  });
  actionInnerHTML += 'Reset free camera<br>';
  keyInnerHTML += 'Ctrl A<br>';

  parentContext.gui.open();
}


/** Loads a scene for MuJoCo
 * @param {mujoco} mujoco This is a reference to the mujoco namespace object
 * @param {string} filename This is the name of the .xml file in the /working/ directory of the MuJoCo/Emscripten Virtual File System
 * @param {MuJoCoDemo} parent The three.js Scene Object to add the MuJoCo model elements to
 */
export async function loadSceneFromURL(mujoco, filename, parent) {
    // Free the old data.
    parent.model = null;
    parent.data = null;

    // Load in the state from XML.
    parent.model = mujoco.MjModel.loadFromXML("/working/" + filename);
    parent.data  = new mujoco.MjData(parent.model);

    let model = parent.model;
    let data = parent.data;
    const mj = parent.mujoco ?? mujoco;

    // Decode the null-terminated string names.
    let textDecoder = new TextDecoder("utf-8");
    let names_array = new Uint8Array(model.names);
    let fullString = textDecoder.decode(model.names);
    let names = fullString.split(textDecoder.decode(new ArrayBuffer(1)));

  // Parse joint names (needed by observations)
  parent.jointNamesMJC = [];
  for (let j = 0; j < model.njnt; j++) {
    let start_idx = model.name_jntadr[j];
    let end_idx = start_idx;
    while (end_idx < names_array.length && names_array[end_idx] !== 0) {
      end_idx++;
    }
    let name_buffer = names_array.subarray(start_idx, end_idx);
    parent.jointNamesMJC.push(textDecoder.decode(name_buffer));
  }

  // Parse actuator names (needed for control)
  parent.actuatorNamesMJC = [];
  for (let a = 0; a < model.nu; a++) {
    let start_idx = model.name_actuatoradr[a];
    let end_idx = start_idx;
    while (end_idx < names_array.length && names_array[end_idx] !== 0) {
      end_idx++;
    }
    let name_buffer = names_array.subarray(start_idx, end_idx);
    const actuator_name = textDecoder.decode(name_buffer);
    parent.actuatorNamesMJC.push(actuator_name);
  }

  let asset_meta = null;
  try {
    const metaPathGuess = parent.params && parent.params.scene && parent.params.scene.includes('unitree_g1')
      ? './examples/checkpoints/g1/asset_meta.json'
      : './examples/checkpoints/g1/asset_meta.json';
    asset_meta = await fetch(metaPathGuess).then(r => r.json());
  } catch (e) {
    console.warn('asset_meta.json not found; using actuator order', e);
  }

  let actuator2joint = [];
  for (let i = 0; i < model.nu; i++) {
    actuator2joint.push(model.actuator_trnid[2 * i]);
  }

  if (asset_meta && asset_meta["joint_names_isaac"]) {
    parent.jointNamesIsaac = asset_meta["joint_names_isaac"];
  } else {
    parent.jointNamesIsaac = actuator2joint.map(jid => parent.jointNamesMJC[jid]);
  }

  // Addresses in isaac joint order
  parent.ctrl_adr_isaac = new Array(model.nu);
  parent.qpos_adr_isaac = new Array(model.nu);
  parent.qvel_adr_isaac = new Array(model.nu);
  for (let i = 0; i < parent.jointNamesIsaac.length; i++) {
    const name = parent.jointNamesIsaac[i];
    const jid = parent.jointNamesMJC.indexOf(name);
    // map to actuator index
    let actIdx = -1;
    for (let a = 0; a < model.nu; a++) {
      if (model.actuator_trnid[2 * a] === jid) { actIdx = a; break; }
    }
    parent.ctrl_adr_isaac[i] = actIdx;
    parent.qpos_adr_isaac[i] = model.jnt_qposadr[jid];
    parent.qvel_adr_isaac[i] = model.jnt_dofadr[jid];
  }

  // Default joint positions from asset_meta
  if (asset_meta && asset_meta["default_joint_pos"]) {
    parent.defaultJpos = new Float32Array(asset_meta["default_joint_pos"]);
  } else {
    parent.defaultJpos = new Float32Array(parent.jointNamesIsaac.length);
    for (let i = 0; i < parent.jointNamesIsaac.length; i++) {
      parent.defaultJpos[i] = parent.data.qpos[parent.qpos_adr_isaac[i]];
    }
  }

    // Create the root object.
    let mujocoRoot = new THREE.Group();
    mujocoRoot.name = "MuJoCo Root"
    parent.scene.add(mujocoRoot);

    /** @type {Object.<number, THREE.Group>} */
    let bodies = {};
    /** @type {Object.<number, THREE.BufferGeometry>} */
    let meshes = {};
    /** @type {THREE.Light[]} */
    let lights = [];

    // Default material definition.
    let material = new THREE.MeshPhysicalMaterial();
    material.color = new THREE.Color(1, 1, 1);

    
    // Loop through the MuJoCo geoms and recreate them in three.js.
    for (let g = 0; g < model.ngeom; g++) {
      // Only visualize geom groups up to 2 (same default behavior as simulate).
      if (!(model.geom_group[g] < 3)) { continue; }

      // Get the body ID and type of the geom.
      let b    = model.geom_bodyid[g];
      let type = model.geom_type  [g];
      
      // Get geom name for debugging
      let geomName = 'unknown';
      if (model.name_geomadr && model.names) {
        let nameStart = model.name_geomadr[g];
        let nameEnd = nameStart;
        let names_array = new Uint8Array(model.names);
        while (nameEnd < names_array.length && names_array[nameEnd] !== 0) nameEnd++;
        geomName = new TextDecoder("utf-8").decode(names_array.subarray(nameStart, nameEnd));
      }
      let size = [
        model.geom_size[(g*3) + 0],
        model.geom_size[(g*3) + 1],
        model.geom_size[(g*3) + 2]
      ];

      // Create the body if it doesn't exist.
      if (!(b in bodies)) {
        bodies[b] = new THREE.Group();
        
        let start_idx = model.name_bodyadr[b];
        let end_idx = start_idx;
        while (end_idx < names_array.length && names_array[end_idx] !== 0) {
          end_idx++;
        }
        let name_buffer = names_array.subarray(start_idx, end_idx);
        bodies[b].name = textDecoder.decode(name_buffer);
        
        bodies[b].bodyID = b;

        // Mark pelvis/base id for impulse application
        if (bodies[b].name === 'base' || bodies[b].name === 'pelvis') {
          parent.pelvis_body_id = b;
        }
        bodies[b].has_custom_mesh = false;
      }

      // Set the default geometry. In MuJoCo, this is a sphere.
      let geometry = new THREE.SphereGeometry(size[0] * 0.5);
      if (type == mj.mjtGeom.mjGEOM_PLANE.value) {
        // Special handling for plane later.
      } else if (type == mj.mjtGeom.mjGEOM_HFIELD.value) {
        // TODO: Implement this.
      } else if (type == mj.mjtGeom.mjGEOM_SPHERE.value) {
        geometry = new THREE.SphereGeometry(size[0]);
      } else if (type == mj.mjtGeom.mjGEOM_CAPSULE.value) {
        geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2.0, 20, 20);
      } else if (type == mj.mjtGeom.mjGEOM_ELLIPSOID.value) {
        geometry = new THREE.SphereGeometry(1); // Stretch this below
      } else if (type == mj.mjtGeom.mjGEOM_CYLINDER.value) {
        geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2.0);
      } else if (type == mj.mjtGeom.mjGEOM_BOX.value) {
        geometry = new THREE.BoxGeometry(size[0] * 2.0, size[2] * 2.0, size[1] * 2.0);
      } else if (type == mj.mjtGeom.mjGEOM_MESH.value) {
        let meshID = model.geom_dataid[g];

        if (!(meshID in meshes)) {
          geometry = new THREE.BufferGeometry(); // TODO: Populate the Buffer Geometry with Generic Mesh Data

          let vertex_buffer = model.mesh_vert.subarray(
             model.mesh_vertadr[meshID] * 3,
            (model.mesh_vertadr[meshID]  + model.mesh_vertnum[meshID]) * 3);
          for (let v = 0; v < vertex_buffer.length; v+=3){
            //vertex_buffer[v + 0] =  vertex_buffer[v + 0];
            let temp             =  vertex_buffer[v + 1];
            vertex_buffer[v + 1] =  vertex_buffer[v + 2];
            vertex_buffer[v + 2] = -temp;
          }

          let normal_buffer = model.mesh_normal.subarray(
             model.mesh_vertadr[meshID] * 3,
            (model.mesh_vertadr[meshID]  + model.mesh_vertnum[meshID]) * 3);
          for (let v = 0; v < normal_buffer.length; v+=3){
            //normal_buffer[v + 0] =  normal_buffer[v + 0];
            let temp             =  normal_buffer[v + 1];
            normal_buffer[v + 1] =  normal_buffer[v + 2];
            normal_buffer[v + 2] = -temp;
          }

          let uv_buffer = model.mesh_texcoord.subarray(
             model.mesh_texcoordadr[meshID] * 2,
            (model.mesh_texcoordadr[meshID]  + model.mesh_vertnum[meshID]) * 2);
          let triangle_buffer = model.mesh_face.subarray(
             model.mesh_faceadr[meshID] * 3,
            (model.mesh_faceadr[meshID]  + model.mesh_facenum[meshID]) * 3);
          geometry.setAttribute("position", new THREE.BufferAttribute(vertex_buffer, 3));
          geometry.setAttribute("normal"  , new THREE.BufferAttribute(normal_buffer, 3));
          geometry.setAttribute("uv"      , new THREE.BufferAttribute(    uv_buffer, 2));
          geometry.setIndex    (Array.from(triangle_buffer));
          meshes[meshID] = geometry;
        } else {
          geometry = meshes[meshID];
        }

        bodies[b].has_custom_mesh = true;
      }
      // Done with geometry creation.

      // Set the Material Properties of incoming bodies
      let texture = undefined;
      let color = [
        model.geom_rgba[(g * 4) + 0],
        model.geom_rgba[(g * 4) + 1],
        model.geom_rgba[(g * 4) + 2],
        model.geom_rgba[(g * 4) + 3]];
      if (model.geom_matid[g] != -1) {
        let matId = model.geom_matid[g];
        color = [
          model.mat_rgba[(matId * 4) + 0],
          model.mat_rgba[(matId * 4) + 1],
          model.mat_rgba[(matId * 4) + 2],
          model.mat_rgba[(matId * 4) + 3]];

        // Construct texture from model.tex_data via texture roles (MuJoCo 3.3.8)
        texture = undefined;
        const mjNTEXROLE = 10;
        const mjTEXROLE_RGB = 1;
        const texId = model.mat_texid[(matId * mjNTEXROLE) + mjTEXROLE_RGB];

        if (texId !== -1) {
          const width    = model.tex_width [texId];
          const height   = model.tex_height[texId];
          const offset   = model.tex_adr   [texId];
          const channels = model.tex_nchannel[texId];
          const texData  = model.tex_data;
          const rgbaArray = new Uint8Array(width * height * 4);
          for (let p = 0; p < width * height; p++) {
            rgbaArray[(p * 4) + 0] = texData[offset + ((p * channels) + 0)];
            rgbaArray[(p * 4) + 1] = channels > 1 ? texData[offset + ((p * channels) + 1)] : rgbaArray[(p * 4) + 0];
            rgbaArray[(p * 4) + 2] = channels > 2 ? texData[offset + ((p * channels) + 2)] : rgbaArray[(p * 4) + 0];
            rgbaArray[(p * 4) + 3] = channels > 3 ? texData[offset + ((p * channels) + 3)] : 255;
          }
          texture = new THREE.DataTexture(rgbaArray, width, height, THREE.RGBAFormat, THREE.UnsignedByteType);
          if (texId === 2) {
            texture.repeat = new THREE.Vector2(50, 50);
          } else {
            const repeatIndex = model.geom_matid[g] * 2;
            texture.repeat = new THREE.Vector2(
              model.mat_texrepeat[repeatIndex + 0] ?? 1,
              model.mat_texrepeat[repeatIndex + 1] ?? 1
            );
          }
          texture.wrapS = THREE.RepeatWrapping;
          texture.wrapT = THREE.RepeatWrapping;
          texture.needsUpdate = true;
        }
      }

      const materialParams = {
        color: new THREE.Color(color[0], color[1], color[2]),
        transparent: color[3] < 1.0,
        opacity: color[3] / 255.0
      };

      if (texture) {
        materialParams.map = texture;
      }

      const materialId = model.geom_matid[g];
      if (materialId !== -1) {
        const specular = model.mat_specular?.[materialId];
        const reflectance = model.mat_reflectance?.[materialId];
        const shininess = model.mat_shininess?.[materialId];
        const metallic = model.mat_metallic?.[materialId];

        if (specular !== undefined) {
          materialParams.specularIntensity = specular;
        }
        if (reflectance !== undefined) {
          materialParams.reflectivity = reflectance;
        }
        if (shininess !== undefined) {
          materialParams.roughness = 1.0 - shininess;
        }
        if (metallic !== undefined) {
          materialParams.metalness = metallic;
        }
      }

      for (const key of Object.keys(materialParams)) {
        if (materialParams[key] === undefined) {
          delete materialParams[key];
        }
      }

      // Create a new material for each geom to avoid cross-contamination
      const currentMaterial = new THREE.MeshPhysicalMaterial(materialParams);

      let mesh = new THREE.Mesh();
      if (type == 0) {
        mesh = new Reflector( new THREE.PlaneGeometry( 100, 100 ), { clipBias: 0.003, texture: texture } );
        mesh.rotateX( - Math.PI / 2 );
      } else {
        mesh = new THREE.Mesh(geometry, currentMaterial);
      }

      mesh.castShadow = g == 0 ? false : true;
      mesh.receiveShadow = type != 7;
      mesh.bodyID = b;
      bodies[b].add(mesh);
      getPosition  (model.geom_pos, g, mesh.position  );
      if (type != 0) { getQuaternion(model.geom_quat, g, mesh.quaternion); }
      if (type == 4) { mesh.scale.set(size[0], size[2], size[1]) } // Stretch the Ellipsoid
    }

    // Parse tendons.
    let tendonMat = new THREE.MeshPhongMaterial();
    tendonMat.color = new THREE.Color(0.8, 0.3, 0.3);
    mujocoRoot.cylinders = new THREE.InstancedMesh(
        new THREE.CylinderGeometry(1, 1, 1),
        tendonMat, 1023);
    mujocoRoot.cylinders.receiveShadow = true;
    mujocoRoot.cylinders.castShadow    = true;
    mujocoRoot.add(mujocoRoot.cylinders);
    mujocoRoot.spheres = new THREE.InstancedMesh(
        new THREE.SphereGeometry(1, 10, 10),
        tendonMat, 1023);
    mujocoRoot.spheres.receiveShadow = true;
    mujocoRoot.spheres.castShadow    = true;
    mujocoRoot.add(mujocoRoot.spheres);

    // Parse lights.
    for (let l = 0; l < model.nlight; l++) {
      let light = new THREE.SpotLight();
      if (model.light_type[l] == 0) {
        light = new THREE.SpotLight();
      } else if (model.light_type[l] == 1) {
        light = new THREE.DirectionalLight();
      } else if (model.light_type[l] == 2) {
        light = new THREE.PointLight();
      }else if (model.light_type[l] == 3) {
        light = new THREE.HemisphereLight();
      }
      light.decay = model.light_attenuation[l] * 100;
      light.penumbra = 0.5;
      light.castShadow = true; // default false

      light.shadow.mapSize.width = 1024; // default
      light.shadow.mapSize.height = 1024; // default
      light.shadow.camera.near = 0.1; // default
      light.shadow.camera.far = 10; // default
      //bodies[model.light_bodyid()].add(light);
      if (bodies[0]) {
        bodies[0].add(light);
      } else {
        mujocoRoot.add(light);
      }
      lights.push(light);
    }
    if (model.nlight == 0) {
      let light = new THREE.DirectionalLight();
      mujocoRoot.add(light);
    }

    for (let b = 0; b < model.nbody; b++) {
      //let parent_body = model.body_parentid()[b];
      if (b == 0 || !bodies[0]) {
        mujocoRoot.add(bodies[b]);
      } else if(bodies[b]){
        bodies[0].add(bodies[b]);
      } else {
        bodies[b] = new THREE.Group(); bodies[b].name = names[b + 1]; bodies[b].bodyID = b; bodies[b].has_custom_mesh = false;
        bodies[0].add(bodies[b]);
      }
    }
  
    parent.mujocoRoot = mujocoRoot;

    return [model, data, bodies, lights]
}

/** Downloads the scenes/examples folder to MuJoCo's virtual filesystem
 * @param {mujoco} mujoco */
export async function downloadExampleScenesFolder(mujoco) {
  let allFiles = [
    // G1 dependencies
    "unitree_g1/meshes/head_link.STL",
    "unitree_g1/meshes/left_ankle_pitch_link.STL",
    "unitree_g1/meshes/left_ankle_roll_link.STL",
    "unitree_g1/meshes/left_elbow_link.STL",
    "unitree_g1/meshes/left_hand_index_0_link.STL",
    "unitree_g1/meshes/left_hand_index_1_link.STL",
    "unitree_g1/meshes/left_hand_middle_0_link.STL",
    "unitree_g1/meshes/left_hand_middle_1_link.STL",
    "unitree_g1/meshes/left_hand_palm_link.STL",
    "unitree_g1/meshes/left_hand_thumb_0_link.STL",
    "unitree_g1/meshes/left_hand_thumb_1_link.STL",
    "unitree_g1/meshes/left_hand_thumb_2_link.STL",
    "unitree_g1/meshes/left_hip_pitch_link.STL",
    "unitree_g1/meshes/left_hip_roll_link.STL",
    "unitree_g1/meshes/left_hip_yaw_link.STL",
    "unitree_g1/meshes/left_knee_link.STL",
    "unitree_g1/meshes/left_rubber_hand.STL",
    "unitree_g1/meshes/left_shoulder_pitch_link.STL",
    "unitree_g1/meshes/left_shoulder_roll_link.STL",
    "unitree_g1/meshes/left_shoulder_yaw_link.STL",
    "unitree_g1/meshes/left_wrist_pitch_link.STL",
    "unitree_g1/meshes/left_wrist_roll_link.STL",
    "unitree_g1/meshes/left_wrist_roll_rubber_hand.STL",
    "unitree_g1/meshes/left_wrist_yaw_link.STL",
    "unitree_g1/meshes/logo_link.STL",
    "unitree_g1/meshes/pelvis_contour_link.STL",
    "unitree_g1/meshes/pelvis.STL",
    "unitree_g1/meshes/right_ankle_pitch_link.STL",
    "unitree_g1/meshes/right_ankle_roll_link.STL",
    "unitree_g1/meshes/right_elbow_link.STL",
    "unitree_g1/meshes/right_hand_index_0_link.STL",
    "unitree_g1/meshes/right_hand_index_1_link.STL",
    "unitree_g1/meshes/right_hand_middle_0_link.STL",
    "unitree_g1/meshes/right_hand_middle_1_link.STL",
    "unitree_g1/meshes/right_hand_palm_link.STL",
    "unitree_g1/meshes/right_hand_thumb_0_link.STL",
    "unitree_g1/meshes/right_hand_thumb_1_link.STL",
    "unitree_g1/meshes/right_hand_thumb_2_link.STL",
    "unitree_g1/meshes/right_hip_pitch_link.STL",
    "unitree_g1/meshes/right_hip_roll_link.STL",
    "unitree_g1/meshes/right_hip_yaw_link.STL",
    "unitree_g1/meshes/right_knee_link.STL",
    "unitree_g1/meshes/right_rubber_hand.STL",
    "unitree_g1/meshes/right_shoulder_pitch_link.STL",
    "unitree_g1/meshes/right_shoulder_roll_link.STL",
    "unitree_g1/meshes/right_shoulder_yaw_link.STL",
    "unitree_g1/meshes/right_wrist_pitch_link.STL",
    "unitree_g1/meshes/right_wrist_roll_link.STL",
    "unitree_g1/meshes/right_wrist_roll_rubber_hand.STL",
    "unitree_g1/meshes/right_wrist_yaw_link.STL",
    "unitree_g1/meshes/torso_constraint_L_link.STL",
    "unitree_g1/meshes/torso_constraint_L_rod_link.STL",
    "unitree_g1/meshes/torso_constraint_R_link.STL",
    "unitree_g1/meshes/torso_constraint_R_rod_link.STL",
    "unitree_g1/meshes/torso_link.STL",
    "unitree_g1/meshes/waist_constraint_L.STL",
    "unitree_g1/meshes/waist_constraint_R.STL",
    "unitree_g1/meshes/waist_roll_link.STL",
    "unitree_g1/meshes/waist_support_link.STL",
    "unitree_g1/meshes/waist_yaw_link.STL",
    "unitree_g1/g1_23dof.xml",
    "unitree_g1/scene_23dof.xml",

  ];

  let requests = allFiles.map((url) => fetch("./examples/scenes/" + url));
  let responses = await Promise.all(requests);
  for (let i = 0; i < responses.length; i++) {
      let split = allFiles[i].split("/");
      let working = '/working/';
      for (let f = 0; f < split.length - 1; f++) {
          working += split[f];
          if (!mujoco.FS.analyzePath(working).exists) { mujoco.FS.mkdir(working); }
          working += "/";
      }

      if (allFiles[i].endsWith(".png") || allFiles[i].endsWith(".stl") || allFiles[i].endsWith(".STL") || allFiles[i].endsWith(".skn")) {
          mujoco.FS.writeFile("/working/" + allFiles[i], new Uint8Array(await responses[i].arrayBuffer()));
      } else {
          mujoco.FS.writeFile("/working/" + allFiles[i], await responses[i].text());
      }
  }
}

/** Access the vector at index, swizzle for three.js, and apply to the target THREE.Vector3
 * @param {Float32Array|Float64Array} buffer
 * @param {number} index
 * @param {THREE.Vector3} target */
export function getPosition(buffer, index, target, swizzle = true) {
  if (swizzle) {
    return target.set(
       buffer[(index * 3) + 0],
       buffer[(index * 3) + 2],
      -buffer[(index * 3) + 1]);
  } else {
    return target.set(
       buffer[(index * 3) + 0],
       buffer[(index * 3) + 1],
       buffer[(index * 3) + 2]);
  }
}

/** Access the quaternion at index, swizzle for three.js, and apply to the target THREE.Quaternion
 * @param {Float32Array|Float64Array} buffer
 * @param {number} index
 * @param {THREE.Quaternion} target */
export function getQuaternion(buffer, index, target, swizzle = true) {
  if (swizzle) {
    return target.set(
      -buffer[(index * 4) + 1],
      -buffer[(index * 4) + 3],
       buffer[(index * 4) + 2],
      -buffer[(index * 4) + 0]);
  } else {
    return target.set(
       buffer[(index * 4) + 0],
       buffer[(index * 4) + 1],
       buffer[(index * 4) + 2],
       buffer[(index * 4) + 3]);
  }
}

/** Converts this Vector3's Handedness to MuJoCo's Coordinate Handedness
 * @param {THREE.Vector3} target */
export function toMujocoPos(target) { return target.set(target.x, -target.z, target.y); }

/** Standard normal random number generator using Box-Muller transform */
export function standardNormal() {
  return Math.sqrt(-2.0 * Math.log( Math.random())) *
         Math.cos ( 2.0 * Math.PI * Math.random()); }

