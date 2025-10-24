import * as THREE from 'three';

/**
 * G1 Robot Observation Helper
 * Dynamically builds observations from YAML config for Unitree G1 humanoid
 * Supports: base_ang_vel, projected_gravity, velocity_commands, joint_pos_rel, joint_vel_rel, last_action
 * Each with configurable history length and scaling
 */
class G1YAMLObs {
  constructor(model, simulation, demo, kwargs = {}) {
    this.model = model;
    this.simulation = simulation;
    this.demo = demo;
    
    const { yaml_config, obs_terms, num_joints } = kwargs;
    this.yaml_config = yaml_config;
    this.obs_terms = obs_terms;
    this.num_joints = num_joints;
    
    // Get the floating base joint for base_ang_vel and projected_gravity
    const base_joint_name = "floating_base_joint";
    const joint_idx = demo.jointNamesMJC.indexOf(base_joint_name);
    this.base_qvel_adr = model.jnt_dofadr[joint_idx];
    this.base_qpos_adr = model.jnt_qposadr[joint_idx];
    
    // Get joint addresses for the actuated joints
    this.joint_qpos_adr = [];
    this.joint_qvel_adr = [];
    
    // Use joint_ids_map to determine which joints to read
    const joint_ids_map = yaml_config.joint_ids_map;
    
    for (let i = 0; i < num_joints; i++) {
      const mapped_id = joint_ids_map[i];
      const joint_name = demo.jointNamesIsaac[mapped_id];
      const idx = demo.jointNamesMJC.indexOf(joint_name);
      
      if (idx === -1) {
        console.error(`Joint ${i}: Cannot find joint "${joint_name}" (Isaac idx ${mapped_id}) in MuJoCo model!`);
      }
      
      this.joint_qpos_adr.push(model.jnt_qposadr[idx]);
      this.joint_qvel_adr.push(model.jnt_dofadr[idx]);
    }
    
    // Initialize history buffers for each observation term
    // On first compute(), we'll populate with current values
    this.history = {};
    this.history_initialized = false;
    for (const [name, config] of Object.entries(obs_terms)) {
      const base_size = config.scale.length;
      const history_length = config.history_length;
      this.history[name] = new Array(history_length).fill(null).map(() => new Float32Array(base_size));
    }
  }
  
  compute(extra_info) {
    const obs_buffer = [];
    
    // On first call, initialize history with current values (repeated)
    if (!this.history_initialized) {
      for (const [name, config] of Object.entries(this.obs_terms)) {
        const current_value = this.computeTerm(name, config);
        // Fill all history slots with the current value
        for (let i = 0; i < this.history[name].length; i++) {
          this.history[name][i] = new Float32Array(current_value);
        }
      }
      this.history_initialized = true;
    }
    
    for (const [name, config] of Object.entries(this.obs_terms)) {
      const current_value = this.computeTerm(name, config);
      
      // Update history - shift left (remove oldest) and append new value at end
      this.history[name].shift();
      this.history[name].push(current_value);
      
      // Flatten history and add to observation buffer (OLDEST TO NEWEST to match Isaac Lab)
      // Isaac Lab convention: oldest observations first, then newer history
      for (let i = 0; i < this.history[name].length; i++) {
        obs_buffer.push(...this.history[name][i]);
      }
    }
    
    return new Float32Array(obs_buffer);
  }
  
  computeTerm(name, config) {
    const qpos = this.simulation.qpos;
    const qvel = this.simulation.qvel;
    const scale = config.scale;
    
    if (!scale || scale.length === 0) {
      console.error(`Missing scale for observation term: ${name}`);
      return new Float32Array(0);
    }
    
    switch (name) {
      case 'base_ang_vel': {
        // Read angular velocity from base joint (3D)
        // For free joints in MuJoCo, qvel has 6 DOFs: [lin_vel(3), ang_vel(3)]
        // So angular velocity is at offset +3 from base_qvel_adr
        const ang_vel = new Float32Array(3);
        for (let i = 0; i < 3; i++) {
          ang_vel[i] = qvel[this.base_qvel_adr + 3 + i] * scale[i];
        }
        return ang_vel;
      }
      
      case 'projected_gravity': {
        // Transform gravity vector to body frame
        const qw = qpos[this.base_qpos_adr + 3];
        const qx = qpos[this.base_qpos_adr + 4];
        const qy = qpos[this.base_qpos_adr + 5];
        const qz = qpos[this.base_qpos_adr + 6];
        const quat_inv = new THREE.Quaternion(qx, qy, qz, qw).invert();
        const gravity = new THREE.Vector3(0, 0, -1.0).applyQuaternion(quat_inv);
        
        return new Float32Array([
          gravity.x * scale[0],
          gravity.y * scale[1],
          gravity.z * scale[2]
        ]);
      }
      
      case 'velocity_commands': {
        // Get velocity commands from demo params
        const command_vel_x = this.demo.params["command_vel_x"] || 0.0;
        const command_vel_y = this.demo.params["command_vel_y"] || 0.0;
        const command_vel_yaw = this.demo.params["command_vel_yaw"] || 0.0;
        
        return new Float32Array([
          command_vel_x * scale[0],
          command_vel_y * scale[1],
          command_vel_yaw * scale[2]
        ]);
      }
      
      case 'joint_pos_rel': {
        // Joint positions relative to default (23 joints for G1)
        // default_pos is indexed by action space (i), not joint space
        const joint_pos = new Float32Array(this.num_joints);
        const default_pos = this.yaml_config.default_joint_pos;
        for (let i = 0; i < this.num_joints; i++) {
          const adr = this.joint_qpos_adr[i];
          if (adr === undefined || adr === -1 || adr >= qpos.length) {
            console.error(`joint_pos_rel: Invalid address ${adr} for action ${i} (qpos length: ${qpos.length})`);
            joint_pos[i] = 0.0;
            continue;
          }
          const pos = qpos[adr];
          // Check for extreme values - qpos should be in radians, typically -π to π
          if (!isFinite(pos) || Math.abs(pos) > 100) {
            console.error(`joint_pos_rel[${i}]: Extreme qpos value ${pos} at address ${adr}, joint: ${this.demo.jointNamesIsaac[this.yaml_config.joint_ids_map[i]]}`);
          }
          // default_pos is indexed by action index i
          joint_pos[i] = (pos - default_pos[i]) * scale[i];
        }
        return joint_pos;
      }
      
      case 'joint_vel_rel': {
        // Joint velocities (23 joints for G1)
        const joint_vel = new Float32Array(this.num_joints);
        for (let i = 0; i < this.num_joints; i++) {
          const adr = this.joint_qvel_adr[i];
          if (adr === undefined || adr === -1) {
            console.error(`joint_vel_rel: Invalid address for joint ${i}`);
            joint_vel[i] = 0.0;
            continue;
          }
          joint_vel[i] = qvel[adr] * scale[i];
        }
        return joint_vel;
      }
      
      case 'last_action': {
        // Previous actions from action buffer (23 actions for G1)
        // lastActions stores the normalized network outputs directly
        const last_action = new Float32Array(this.num_joints);
        
        for (let i = 0; i < this.num_joints; i++) {
          // Get the normalized action from the buffer (default to 0)
          const normalized_action = this.demo.lastActions && this.demo.lastActions[i] !== undefined ? this.demo.lastActions[i] : 0.0;
          // Apply the observation scale
          last_action[i] = normalized_action * scale[i];
          
          // Check for NaN
          if (!isFinite(last_action[i])) {
            console.error(`last_action[${i}] is NaN/Inf:`, {
              normalized_action,
              scale: scale[i],
              result: last_action[i]
            });
          }
        }
        return last_action;
      }
      
      default:
        console.warn(`Unknown observation term: ${name}`);
        return new Float32Array(scale.length);
    }
  }
}


// Export only G1 observation helper
export const Observations = {
  G1YAMLObs
};
