import * as THREE from 'three';

function getOscillator(time) {
  const omega = 4.0 * Math.PI
  const phase = [
    omega * time + Math.PI,
    omega * time,
    omega * time,
    omega * time + Math.PI,
  ];
  return [...phase.map(Math.sin), ...phase.map(Math.cos), omega, omega, omega, omega];
}

class VelocityCommand {
  constructor(model, simulation, demo, kwargs = {}) {
    this.model = model;
    this.simulation = simulation;
    this.demo = demo;
    // Unpack kwargs with defaults
    const { angvel_kp = 1.0 } = kwargs;
    this.angvel_kp = angvel_kp;
  }

  compute(extra_info) {
    const osc = getOscillator(this.demo.mujoco_time / 1000.);
    const command_vel_x = new THREE.Vector3(this.demo.params["command_vel_x"], 0, 0);
    const setvel_b = command_vel_x.clone().applyQuaternion(this.demo.quat.clone().invert());
    return [setvel_b.x, setvel_b.y, this.angvel_kp * (0 - this.demo.rpy.z), 0, ...osc];
  }

}


class ImpedanceCommand {
  constructor(model, simulation, demo, kwargs = {}) {
    this.model = model;
    this.simulation = simulation;
    this.demo = demo;
    const { mass = 1.0 } = kwargs;
    this.mass = mass;
  }

  compute(extra_info) {
    const kp = this.demo.params["impedance_kp"];
    const kd = 1.8 * Math.sqrt(kp);
    const osc = getOscillator(this.demo.mujoco_time / 1000.);

    // Get base position in world frame
    const base_pos_w = new THREE.Vector3(...this.simulation.qpos.subarray(0, 3));
    const command_vel_x = new THREE.Vector3(this.demo.params["command_vel_x"], 0, 0);
    const setpoint = command_vel_x.clone().multiplyScalar(kd / kp).add(base_pos_w);
    if (this.demo.params["compliant_mode"]) {
      setpoint.copy(base_pos_w);
    }

    if (this.demo.params["use_setpoint"]) {
      setpoint.x = this.demo.ball.position.x;
      setpoint.y = -this.demo.ball.position.z;
      setpoint.z = 0.0;
    } else {
      this.demo.ball.position.set(setpoint.x, 0.5, -setpoint.y);
    }

    // Transform setpoint to body frame
    let setpoint_b = setpoint.sub(base_pos_w).applyQuaternion(this.demo.quat.clone().invert());

    let setpoint_b_norm = setpoint_b.length();
    setpoint_b.normalize().multiplyScalar(Math.min(setpoint_b_norm, 2.0));

    const command = [
      setpoint_b.x, setpoint_b.y,
      0 - this.demo.rpy.z,
      kp * setpoint_b.x, kp * setpoint_b.y,
      kd, kd, kd,
      kp * (0 - this.demo.rpy.z),
      this.mass,
      kp * setpoint_b.x / this.mass, kp * setpoint_b.y / this.mass,
      kd / this.mass, kd / this.mass, kd / this.mass
    ];

    return [...command, ...osc];
  }
}

class BaseAngVelMultistep {
  /**
   * 
   * @param {mujoco.Model} model 
   * @param {mujoco.Simulation} simulation 
   * @param {MuJoCoDemo} demo
   * @param {string} base_joint_name
   * @param {number} steps 
   */
  constructor(model, simulation, demo, kwargs = {}) {
    this.model = model;
    this.simulation = simulation;
    // Unpack kwargs with defaults
    const { 
      base_joint_name = "floating_base_joint",
      history_steps = 4 
    } = kwargs;
    
    this.steps = history_steps;
    this.angvel_multistep = new Array(this.steps).fill().map(() => new Float32Array(3));

    const joint_idx = demo.jointNamesMJC.indexOf(base_joint_name);
    this.joint_qvel_adr = model.jnt_dofadr[joint_idx];
    console.log("ang vel dof adr", this.joint_qvel_adr);
  }

  /**
   * 
   * @param {dict} extra_info
   * @returns {Float32Array}
   */
  compute(extra_info) {
    // Update history - shift all elements forward
    for (let i = this.angvel_multistep.length - 1; i > 0; i--) {
      this.angvel_multistep[i] = this.angvel_multistep[i - 1];
    }
    const qv = this.simulation.qvel;
    const angvel0 = new Float32Array([
      qv[this.joint_qvel_adr + 0],
      qv[this.joint_qvel_adr + 1],
      qv[this.joint_qvel_adr + 2]
    ]);
    this.angvel_multistep[0] = angvel0;
    // Flatten all steps into single array
    const flattened = new Float32Array(this.steps * 3);
    for (let i = 0; i < this.steps; i++) {
      flattened.set(this.angvel_multistep[i], i * 3);
    }
    return flattened;
  }
}

class GravityMultistep {
  /**
   * 
   * @param {mujoco.Model} model 
   * @param {mujoco.Simulation} simulation 
   * @param {MuJoCoDemo} demo
   * @param {string} base_joint_name
   * @param {number} steps 
   */
  constructor(model, simulation, demo, kwargs = {}) {
    this.model = model;
    this.simulation = simulation;
    // Unpack kwargs with defaults
    const { 
      joint_name = "floating_base_joint",
      history_steps = 4,
      gravity = [0, 0, -1.0]
    } = kwargs;
    
    this.steps = history_steps;
    this.gravity_multistep = new Array(this.steps).fill().map(() => new Float32Array(3));

    const joint_idx = demo.jointNamesMJC.indexOf(joint_name);
    this.joint_qpos_adr = model.jnt_qposadr[joint_idx];
    this.gravity = new THREE.Vector3(...gravity);
  }

  /**
   * 
   * @param {dict} extra_info
   * @returns {Float32Array}
   */
  compute(extra_info) {
    // Get projected gravity and normalize it
    const qpos = this.simulation.qpos;
    const qw = qpos[this.joint_qpos_adr + 3];
    const qx = qpos[this.joint_qpos_adr + 4];
    const qy = qpos[this.joint_qpos_adr + 5];
    const qz = qpos[this.joint_qpos_adr + 6];
    // Create quaternion directly from the array values
    const quat_inv = new THREE.Quaternion(qx, qy, qz, qw).invert();
    const gravity = this.gravity.clone().applyQuaternion(quat_inv);

    // Update history
    for (let i = this.gravity_multistep.length - 1; i > 0; i--) {
      this.gravity_multistep[i] = this.gravity_multistep[i - 1];
    }
    this.gravity_multistep[0] = new Float32Array([gravity.x, gravity.y, gravity.z]);

    // Flatten all steps
    const flattened = new Float32Array(this.steps * 3);
    for (let i = 0; i < this.steps; i++) {
      flattened.set(this.gravity_multistep[i], i * 3);
    }
    return flattened;
  }
}

class JointPosMultistep {
  /**
   * 
   * @param {mujoco.Model} model 
   * @param {mujoco.Simulation} simulation 
   * @param {MuJoCoDemo} demo
   * @param {list[string]} joint_names 
   * @param {number} steps 
   */
  constructor(model, simulation, demo, kwargs = {}) {
    this.model = model;
    this.simulation = simulation;
    // Unpack kwargs with defaults
    const { 
      joint_names = [],
      history_steps = 4 
    } = kwargs;
    
    this.steps = history_steps;
    this.joint_names = joint_names;
    this.joint_pos_multistep = new Array(this.steps).fill().map(() => new Float32Array(joint_names.length));
    console.log("joint_names", joint_names);

    this.joint_qpos_adr = [];
    for (let i = 0; i < joint_names.length; i++) {
      const idx = demo.jointNamesMJC.indexOf(joint_names[i]);
      const joinqposadr = model.jnt_qposadr[idx];
      this.joint_qpos_adr.push(joinqposadr);
    }
    console.log("jposadr", this.joint_qpos_adr);
  }

  /**
   * 
   * @param {dict} extra_info
   * @returns {Float32Array}
   */
  compute(extra_info) {
    // Update history
    for (let i = this.joint_pos_multistep.length - 1; i > 0; i--) {
      this.joint_pos_multistep[i] = this.joint_pos_multistep[i - 1];
    }
    const qpos = this.simulation.qpos;
    for (let i = 0; i < this.joint_names.length; i++) {
      this.joint_pos_multistep[0][i] = qpos[this.joint_qpos_adr[i]];
    }

    // Flatten all steps
    const flattened = new Float32Array(this.steps * this.joint_names.length);
    for (let i = 0; i < this.steps; i++) {
      flattened.set(this.joint_pos_multistep[i], i * this.joint_names.length);
    }
    // console.log(flattened.subarray(0, 12));
    return flattened;
  }
}

class JointVelMultistep {
  /**
   * 
   * @param {mujoco.Model} model 
   * @param {mujoco.Simulation} simulation 
   * @param {MuJoCoDemo} demo
   * @param {list[string]} joint_names 
   * @param {number} steps 
   */
  constructor(model, simulation, demo, kwargs = {}) {
    this.model = model;
    this.simulation = simulation;
    // Unpack kwargs with defaults
    const { 
      joint_names = [],
      history_steps = 4 
    } = kwargs;
    
    this.steps = history_steps;
    this.joint_names = joint_names;
    this.numJoints = joint_names.length;
    this.joint_vel_multistep = new Array(this.steps).fill().map(() => new Float32Array(this.numJoints));

    this.joint_qvel_adr = [];
    for (let i = 0; i < joint_names.length; i++) {
      const idx = demo.jointNamesMJC.indexOf(joint_names[i]);
      const jointqveladr = model.jnt_dofadr[idx];
      this.joint_qvel_adr.push(jointqveladr);
    }
    console.log("jveladr", this.joint_qvel_adr);
  }

  /**
   * 
   * @param {dict} extra_info
   * @returns {Float32Array}
   */
  compute(extra_info) {
    // Update history
    for (let i = this.joint_vel_multistep.length - 1; i > 0; i--) {
      this.joint_vel_multistep[i] = this.joint_vel_multistep[i - 1];
    }
    const qvel = this.simulation.qvel;
    for (let i = 0; i < this.joint_names.length; i++) {
      this.joint_vel_multistep[0][i] = qvel[this.joint_qvel_adr[i]];
    }

    // Flatten all steps
    const flattened = new Float32Array(this.steps * this.numJoints);
    for (let i = 0; i < this.steps; i++) {
      flattened.set(this.joint_vel_multistep[i], i * this.numJoints);
    }
    return flattened;
  }
}

class PrevActions {
  /**
   * 
   * @param {mujoco.Model} model 
   * @param {mujoco.Simulation} simulation 
   * @param {MuJoCoDemo} demo
   * @param {number} steps 
   */
  constructor(model, simulation, demo, kwargs = {}) {
    this.model = model;
    this.simulation = simulation;
    // Unpack kwargs with defaults
    const { history_steps = 4 } = kwargs;
    
    this.steps = history_steps;
    this.numActions = demo.numActions;
    this.actionBuffer = demo.actionBuffer;
  }

  /**
   * 
   * @param {dict} extra_info
   * @returns {Float32Array}
   */
  compute(extra_info) {
    const flattened = new Float32Array(this.steps * this.numActions);
    for (let j = 0; j < this.numActions; j++) {
      for (let i = 0; i < this.steps; i++) {
        flattened[j * this.steps + i] = this.actionBuffer[i][j];
      }
    }
    return flattened;
  }
}


class HIMLocoObs {
  constructor(model, simulation, demo, kwargs = {}) {
    this.model = model;
    this.simulation = simulation;
    this.demo = demo;
    this.steps = 6;
    this.defaultJpos = demo.defaultJpos;

    this.commands_scale = [2.0, 2.0, 0.25];
    this.ang_vel_scale = 0.25;
    this.dof_pos_scale = 1.0;
    this.dof_vel_scale = 0.05;

    this.obs_buf = new Float32Array(this.steps * 45);

    const { joint_names } = kwargs;
    this.joint_qpos_adr = [];
    this.joint_qvel_adr = [];
    for (let i = 0; i < joint_names.length; i++) {
      const idx = demo.jointNamesMJC.indexOf(joint_names[i]);
      const joinqposadr = model.jnt_qposadr[idx];
      this.joint_qpos_adr.push(joinqposadr);
      const jointqveladr = model.jnt_dofadr[idx];
      this.joint_qvel_adr.push(jointqveladr);
    }
    console.log("jposadr", this.joint_qpos_adr);
    console.log("jveladr", this.joint_qvel_adr);
  }

  compute(extra_info) {
    // Get current observations similar to Python compute_observations method
    // commands[:, :3] * commands_scale
    const command_vel_x = this.demo.params["command_vel_x"];
    let commands = [command_vel_x, 0.0, 0.0];
    commands = commands.map((cmd, i) => cmd * this.commands_scale[i]);
    // base_ang_vel * obs_scales.ang_vel
    const base_ang_vel = (
      this.simulation.qvel
      .subarray(3, 6)
      .map(vel => vel * this.ang_vel_scale)
    );
    
    const quat = this.simulation.qpos.subarray(3, 7);
    // Create quaternion directly from the array values
    const quat_inv = new THREE.Quaternion(quat[1], quat[2], quat[3], quat[0]).invert();
    const projected_gravity = new THREE.Vector3(0, 0, -1.0).applyQuaternion(quat_inv);
        
    const dof_pos = new Float32Array(12);
    for (let i = 0; i < 12; i++) {
      dof_pos[i] = (this.simulation.qpos[this.joint_qpos_adr[i]] - this.defaultJpos[i]) * this.dof_pos_scale;
    }
    const dof_vel = new Float32Array(12);
    for (let i = 0; i < 12; i++) {
      dof_vel[i] = this.simulation.qvel[this.joint_qvel_adr[i]] * this.dof_vel_scale;
    }
    
    // actions
    const actions = this.demo.lastActions;
    
    // Concatenate all observations
    let current_obs = [
      ...commands,
      ...base_ang_vel,
      ...projected_gravity,
      ...dof_pos,
      ...dof_vel,
      ...actions
    ];
    
    // put current obs into obs_buf's first 45 elements
    // first shift obs_buf's elements forward by 45 elements
    for (let i = this.obs_buf.length - 1; i >= 45; i--) {
      this.obs_buf[i] = this.obs_buf[i - 45];
    }
    // then put current obs into obs_buf's first 45 elements
    this.obs_buf.set(current_obs, 0);
    return this.obs_buf;
  }
}


class DecapObs {
  constructor(model, simulation, demo, kwargs = {}) {
    this.model = model;
    this.simulation = simulation;
    this.demo = demo;
    this.defaultJpos = demo.defaultJpos;

    this.commands_scale = [2.0, 2.0, 0.25];
    this.ang_vel_scale = 0.25;
    this.dof_pos_scale = 1.0;
    this.dof_vel_scale = 0.05;

    const { joint_names } = kwargs;
    this.joint_qpos_adr = [];
    this.joint_qvel_adr = [];
    for (let i = 0; i < joint_names.length; i++) {
      const idx = demo.jointNamesMJC.indexOf(joint_names[i]);
      const joinqposadr = model.jnt_qposadr[idx];
      this.joint_qpos_adr.push(joinqposadr);
      const jointqveladr = model.jnt_dofadr[idx];
      this.joint_qvel_adr.push(jointqveladr);
    }
    console.log("jposadr", this.joint_qpos_adr);
    console.log("jveladr", this.joint_qvel_adr);
  }

  compute(extra_info) {

    const command_vel_x = this.demo.params["command_vel_x"];
    let commands = [command_vel_x, 0.0, 0.0];
    console.log("decap commands", commands);
    commands = commands.map((cmd, i) => cmd * this.commands_scale[i]);

    const quat = this.simulation.qpos.subarray(3, 7);
    // Create quaternion directly from the array values
    const quat_inv = new THREE.Quaternion(quat[1], quat[2], quat[3], quat[0]).invert();
    const projected_gravity = new THREE.Vector3(0, 0, -1.0).applyQuaternion(quat_inv);

    const dof_pos = new Float32Array(12);
    for (let i = 0; i < 12; i++) {
      dof_pos[i] = (this.simulation.qpos[this.joint_qpos_adr[i]] - this.defaultJpos[i]) * this.dof_pos_scale;
    }
    const dof_vel = new Float32Array(12);
    for (let i = 0; i < 12; i++) {
      dof_vel[i] = this.simulation.qvel[this.joint_qvel_adr[i]] * this.dof_vel_scale;
    }

    // actions
    const actions = this.demo.lastActions;

    let obs = [
      ...projected_gravity,
      ...commands,
      ...dof_pos,
      ...dof_vel,
      ...actions
    ];

    return obs;
  }
}


// Export a dictionary of all observation classes
export const Observations = {
  VelocityCommand,
  ImpedanceCommand,
  BaseAngVelMultistep,
  GravityMultistep,
  JointPosMultistep,
  JointVelMultistep,
  PrevActions,
  HIMLocoObs,
  DecapObs
};

