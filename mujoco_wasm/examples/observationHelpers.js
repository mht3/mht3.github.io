import * as THREE from 'three';

const X_AXIS = new THREE.Vector3(1, 0, 0);
const Y_AXIS = new THREE.Vector3(0, 1, 0);
const Z_AXIS = new THREE.Vector3(0, 0, 1);

class MotionLoader {
  constructor({ dt, rootPositions, rootQuaternions, jointPositions }) {
    this.dt = dt;
    this.rootPositions = rootPositions;
    this.rootQuaternions = rootQuaternions;
    this.jointPositions = jointPositions;
    this.numFrames = jointPositions.length;
    this.jointCount = jointPositions.length > 0 ? jointPositions[0].length : 0;
    this.maxTime = Math.max(0, (this.numFrames - 1) * this.dt);
    this.index0 = 0;
    this.index1 = 0;
    this.blend = 0;
    this.currentTime = 0;
    this.jointVelocities = this._computeVelocities();
  }

  static fromCSVText(text, { dt = 0.02 } = {}) {
    if (!text || typeof text !== 'string') {
      throw new Error('MotionLoader: CSV text must be a non-empty string');
    }

    const lines = text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0);

    if (!lines.length) {
      throw new Error('MotionLoader: CSV file contains no data rows');
    }

    const frames = lines.map((line, rowIdx) => {
      const values = line.split(',').map((segment) => parseFloat(segment.trim()));
      if (values.some((v) => Number.isNaN(v))) {
        throw new Error(`MotionLoader: Non-numeric value detected in CSV at row ${rowIdx}`);
      }
      return values;
    });

    const rowLength = frames[0].length;
    if (rowLength < 7) {
      throw new Error(`MotionLoader: Expected at least 7 columns per row, found ${rowLength}`);
    }

    const jointCount = rowLength - 7;
    const rootPositions = frames.map((frame) => new Float32Array(frame.slice(0, 3)));
    const rootQuaternions = frames.map((frame) => {
      const quat = new THREE.Quaternion(frame[3], frame[4], frame[5], frame[6]);
      quat.normalize();
      return quat;
    });
    const jointPositions = frames.map((frame) => new Float32Array(frame.slice(7, 7 + jointCount)));

    return new MotionLoader({ dt, rootPositions, rootQuaternions, jointPositions });
  }

  _computeVelocities() {
    if (this.numFrames === 0 || this.dt <= 0) {
      return [];
    }
    const velocities = new Array(this.numFrames);
    for (let i = 0; i < this.numFrames; i++) {
      velocities[i] = new Float32Array(this.jointCount);
    }
    for (let i = 0; i < this.numFrames - 1; i++) {
      const current = this.jointPositions[i];
      const next = this.jointPositions[i + 1];
      const vel = velocities[i];
      for (let j = 0; j < this.jointCount; j++) {
        vel[j] = (next[j] - current[j]) / this.dt;
      }
    }
    if (this.numFrames > 1) {
      velocities[this.numFrames - 1].set(velocities[this.numFrames - 2]);
    }
    return velocities;
  }

  update(timeSeconds) {
    if (this.numFrames === 0) {
      this.index0 = 0;
      this.index1 = 0;
      this.blend = 0;
      this.currentTime = 0;
      return;
    }

    const clampedTime = Math.max(0, Math.min(timeSeconds, this.maxTime));
    this.currentTime = clampedTime;
    const frameFloat = clampedTime / this.dt;
    const idx0 = Math.floor(frameFloat);
    const idx1 = Math.min(idx0 + 1, this.numFrames - 1);
    const blend = Math.min(1, Math.max(0, frameFloat - idx0));

    this.index0 = idx0;
    this.index1 = idx1;
    this.blend = blend;
  }

  joint_pos() {
    if (this.numFrames === 0) {
      return new Float32Array(0);
    }
    if (this.index0 === this.index1 || this.blend === 0) {
      return new Float32Array(this.jointPositions[this.index0]);
    }
    const a = this.jointPositions[this.index0];
    const b = this.jointPositions[this.index1];
    const blend = this.blend;
    const inv = 1 - blend;
    const result = new Float32Array(this.jointCount);
    for (let i = 0; i < this.jointCount; i++) {
      result[i] = a[i] * inv + b[i] * blend;
    }
    return result;
  }

  joint_vel() {
    if (this.numFrames === 0) {
      return new Float32Array(0);
    }
    if (this.index0 === this.index1 || this.blend === 0) {
      return new Float32Array(this.jointVelocities[this.index0]);
    }
    const a = this.jointVelocities[this.index0];
    const b = this.jointVelocities[this.index1];
    const blend = this.blend;
    const inv = 1 - blend;
    const result = new Float32Array(this.jointCount);
    for (let i = 0; i < this.jointCount; i++) {
      result[i] = a[i] * inv + b[i] * blend;
    }
    return result;
  }

  root_quaternion() {
    if (this.numFrames === 0) {
      return new THREE.Quaternion();
    }
    if (this.index0 === this.index1 || this.blend === 0) {
      return this.rootQuaternions[this.index0].clone();
    }
    const quat = this.rootQuaternions[this.index0].clone();
    quat.slerp(this.rootQuaternions[this.index1], this.blend);
    quat.normalize();
    return quat;
  }

  get duration() {
    return this.maxTime;
  }
}

function extractYawQuaternion(quaternion) {
  const q = quaternion.clone().normalize();
  const yaw = Math.atan2(
    2 * (q.w * q.z + q.x * q.y),
    1 - 2 * (q.y * q.y + q.z * q.z)
  );
  const halfYaw = yaw * 0.5;
  return new THREE.Quaternion(0, 0, Math.sin(halfYaw), Math.cos(halfYaw));
}

function quaternionToAnchorColumns(quaternion) {
  const q = quaternion.clone().normalize();
  const x = q.x;
  const y = q.y;
  const z = q.z;
  const w = q.w;

  const xx = x * x;
  const yy = y * y;
  const zz = z * z;
  const xy = x * y;
  const xz = x * z;
  const yz = y * z;
  const wx = w * x;
  const wy = w * y;
  const wz = w * z;

  const m00 = 1 - 2 * (yy + zz);
  const m01 = 2 * (xy - wz);
  const m02 = 2 * (xz + wy);
  const m10 = 2 * (xy + wz);
  const m11 = 1 - 2 * (xx + zz);
  const m12 = 2 * (yz - wx);

  return [m00, m10, m01, m11, m02, m12];
}

/**
 * G1 Robot Observation Helper
 * Dynamically builds observations from YAML config for Unitree G1 humanoid
 * Supports: base_ang_vel, projected_gravity, velocity_commands, joint_pos_rel, joint_vel_rel, last_action
 * Each with configurable history length and scaling
 */
class G1YAMLObs {
  constructor(model, data, demo, kwargs = {}) {
    this.model = model;
    this.data = data;
    this.demo = demo;
    
    const { yaml_config, obs_terms, num_joints } = kwargs;
    this.yaml_config = yaml_config;
    this.obs_terms = obs_terms;
    this.num_joints = num_joints;
    this.policy_dt = yaml_config.step_dt ?? 0.02;
    
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

    this.motionLoader = null;
    this.motionTime = 0;
    this.motionInitQuat = null;
    this.motionLoaderPath = null;
    this.motionWarningIssued = false;
    this.motionTermsPresent = Object.keys(this.obs_terms).some(
      (name) => name.startsWith('motion_')
    );
    this.ready = this.initializeMotionResources();
  }
  
  async initializeMotionResources() {
    if (!this.motionTermsPresent) {
      return;
    }

    const candidates = this.buildMotionDatasetCandidates();
    const baseDir = this.yaml_config.__baseDir || '.';
    const failures = [];

    for (const candidate of candidates) {
      if (!candidate) continue;
      const resolved = this.resolveMotionPath(baseDir, candidate);
      try {
        const response = await fetch(resolved);
        if (!response.ok) {
          failures.push({ path: resolved, reason: `HTTP ${response.status}` });
          continue;
        }
        const text = await response.text();
        const params =
          this.obs_terms.motion_command?.params ||
          this.obs_terms.motion_anchor_ori_b?.params ||
          {};
        const fpsValue = params.fps !== undefined ? parseFloat(params.fps) : NaN;
        const dt = Number.isFinite(fpsValue) && fpsValue > 0 ? 1 / fpsValue : this.policy_dt;
        this.motionLoader = MotionLoader.fromCSVText(text, { dt });
        this.motionLoaderPath = resolved;
        this.motionTime = 0;
        this.motionInitQuat = null;
        return;
      } catch (error) {
        failures.push({ path: resolved, reason: error?.message || error });
      }
    }

    if (failures.length) {
      console.warn(
        '[G1YAMLObs] Unable to locate motion dataset for motion_* observations. Attempts:',
        failures
      );
    } else {
      console.warn('[G1YAMLObs] No candidate motion datasets found for motion_* observations.');
    }
  }

  buildMotionDatasetCandidates() {
    const override = this.yaml_config.__motionDataset;
    if (override) {
      return [override];
    }
    const motionParams = this.obs_terms.motion_command?.params || {};
    const anchorParams = this.obs_terms.motion_anchor_ori_b?.params || {};
    const direct = motionParams.motion_file || anchorParams.motion_file;
    if (direct) {
      return [direct];
    }
    console.warn('[G1YAMLObs] motion_* observations configured without motion dataset override.');
    return [];
  }

  resolveMotionPath(baseDir, candidate) {
    if (/^(?:https?:)?\/\//i.test(candidate)) {
      return candidate;
    }
    if (candidate.startsWith('/') || candidate.startsWith('./') || candidate.startsWith('../')) {
      return candidate;
    }
    return `${baseDir.replace(/\/+$/, '')}/${candidate}`;
  }

  reset() {
    this.history_initialized = false;
    this.motionTime = 0;
    this.motionInitQuat = null;
    if (this.motionLoader) {
      this.motionLoader.update(0);
    }
  }

  async setMotionProgress(normalized) {
    if (!this.motionLoader || !isFinite(normalized)) {
      return;
    }
    const clamped = Math.max(0, Math.min(1, normalized));
    const targetTime = clamped * this.motionLoader.duration;
    this.motionTime = targetTime;
    this.motionLoader.update(targetTime);
    this.motionInitQuat = null;
    this.history_initialized = false;
  }

  async replayMotion() {
    if (this.ready && typeof this.ready.then === 'function') {
      try {
        await this.ready;
      } catch (err) {
        console.warn('[G1YAMLObs] replayMotion aborted; initialization failed', err);
        return false;
      }
    }
    if (!this.motionLoader) {
      console.warn('[G1YAMLObs] replayMotion requested but no motion loader available');
      return false;
    }
    this.motionTime = 0;
    this.motionLoader.update(0);
    this.motionInitQuat = null;
    this.history_initialized = false;
    return true;
  }

  updateMotionState() {
    if (!this.motionLoader) {
      return;
    }
    this.motionLoader.update(this.motionTime);
    if (!this.motionInitQuat) {
      this.motionInitQuat = this.computeMotionInitQuat();
    }
  }

  computeMotionInitQuat() {
    if (!this.motionLoader) {
      return null;
    }
    const torsoQuat = this.computeTorsoQuaternion();
    if (!torsoQuat) {
      return null;
    }
    const rootQuat = this.motionLoader.root_quaternion();
    const robotYaw = extractYawQuaternion(torsoQuat);
    const refYaw = extractYawQuaternion(rootQuat);
    const init = robotYaw.clone().multiply(refYaw.clone().invert());
    init.normalize();
    return init;
  }

  computeTorsoQuaternion() {
    const qpos = this.data.qpos;
    const baseQuat = new THREE.Quaternion(
      qpos[this.base_qpos_adr + 4],
      qpos[this.base_qpos_adr + 5],
      qpos[this.base_qpos_adr + 6],
      qpos[this.base_qpos_adr + 3]
    );
    const qposIsaac = this.demo.qpos_adr_isaac || [];

    const yawIdx = qposIsaac[12];
    const rollIdx = qposIsaac[13];
    const pitchIdx = qposIsaac[14];

    const yaw = yawIdx !== undefined ? qpos[yawIdx] : 0;
    const roll = rollIdx !== undefined ? qpos[rollIdx] : 0;
    const pitch = pitchIdx !== undefined ? qpos[pitchIdx] : 0;

    const yawQuat = new THREE.Quaternion().setFromAxisAngle(Z_AXIS, yaw);
    const rollQuat = new THREE.Quaternion().setFromAxisAngle(X_AXIS, roll);
    const pitchQuat = new THREE.Quaternion().setFromAxisAngle(Y_AXIS, pitch);

    baseQuat.multiply(yawQuat).multiply(rollQuat).multiply(pitchQuat);
    baseQuat.normalize();
    return baseQuat;
  }

  computeReferenceAnchorQuaternion() {
    if (!this.motionLoader) {
      return null;
    }
    const rootQuat = this.motionLoader.root_quaternion();
    const jointPos = this.motionLoader.joint_pos();
    const yaw = jointPos.length > 12 ? jointPos[12] : 0;
    const yawQuat = new THREE.Quaternion().setFromAxisAngle(Z_AXIS, yaw);
    rootQuat.multiply(yawQuat);
    rootQuat.normalize();
    return rootQuat;
  }

  compute(extra_info) {
    const obs_buffer = [];
    if (this.motionLoader) {
      this.updateMotionState();
    }
    
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
    
    if (this.motionLoader) {
      const stepDt = (typeof this.motionLoader.dt === 'number' && isFinite(this.motionLoader.dt) && this.motionLoader.dt > 0)
        ? this.motionLoader.dt
        : this.policy_dt;
      this.motionTime = Math.min(this.motionTime + stepDt, this.motionLoader.duration);
    }

    return new Float32Array(obs_buffer);
  }
  
  computeTerm(name, config) {
    const qpos = this.data.qpos;
    const qvel = this.data.qvel;
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
        return new Float32Array([
          command_vel_x * scale[0],
          0.0 * (scale[1] ?? 1),
          0.0 * (scale[2] ?? 1)
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

      case 'motion_command': {
        if (!this.motionLoader) {
          if (!this.motionWarningIssued) {
            console.warn('motion_command requested but motion dataset not available; returning zeros');
            this.motionWarningIssued = true;
          }
          return new Float32Array(scale.length);
        }
        const result = new Float32Array(scale.length);
        const jointPosFull = this.motionLoader.joint_pos();
        const jointVelFull = this.motionLoader.joint_vel();
        const jointIdsMap = this.yaml_config.joint_ids_map || [];

        for (let i = 0; i < this.num_joints; i++) {
          const isaacIndex = jointIdsMap[i];
          if (isaacIndex === undefined || isaacIndex < 0 || isaacIndex >= jointPosFull.length) {
            console.warn(`motion_command: invalid joint index mapping for action ${i} -> ${isaacIndex}`);
            continue;
          }
          result[i] = jointPosFull[isaacIndex];
          result[i + this.num_joints] = jointVelFull[isaacIndex];
        }

        for (let i = 0; i < result.length; i++) {
          result[i] *= scale[i] ?? 1;
        }
        return result;
      }

      case 'motion_anchor_ori_b': {
        if (!this.motionLoader) {
          if (!this.motionWarningIssued) {
            console.warn('motion_anchor_ori_b requested but motion dataset not available; returning zeros');
            this.motionWarningIssued = true;
          }
          return new Float32Array(scale.length);
        }
        const initQuat = this.motionInitQuat || this.computeMotionInitQuat();
        const realQuat = this.computeTorsoQuaternion();
        const refQuat = this.computeReferenceAnchorQuaternion();

        if (!initQuat || !realQuat || !refQuat) {
          return new Float32Array(scale.length);
        }

        const rotQuat = initQuat.clone().multiply(refQuat);
        rotQuat.invert();
        rotQuat.multiply(realQuat);
        const values = quaternionToAnchorColumns(rotQuat);
        const result = new Float32Array(values.length);
        for (let i = 0; i < values.length; i++) {
          result[i] = values[i] * (scale[i] ?? 1);
        }
        return result;
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
