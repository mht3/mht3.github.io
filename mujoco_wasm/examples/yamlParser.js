/**
 * Simple YAML parser for G1 deployment configuration files
 * Handles the specific structure of balance_deploy_*.yaml files
 */

export async function parseYAMLConfig(yamlPath) {
  const response = await fetch(yamlPath);
  const text = await response.text();
  
  const config = {
    __sourcePath: yamlPath,
    __baseDir: (() => {
      const lastSlash = yamlPath.lastIndexOf('/');
      if (lastSlash === -1) {
        return '.';
      }
      return yamlPath.slice(0, lastSlash);
    })(),
    joint_ids_map: [],
    step_dt: 0.02,
    stiffness: [],
    damping: [],
    default_joint_pos: [],
    commands: {},
    actions: {},
    observations: {}
  };
  
  const lines = text.split('\n');
  let currentSection = null;
  let currentObsName = null;
  let indentStack = [];
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();
    
    if (!trimmed || trimmed.startsWith('#')) continue;
    
    // Parse arrays like joint_ids_map, stiffness, damping, default_joint_pos
    if (trimmed.startsWith('joint_ids_map:')) {
      config.joint_ids_map = parseArrayValue(line, lines, i);
    } else if (trimmed.startsWith('step_dt:')) {
      config.step_dt = parseFloat(trimmed.split(':')[1].trim());
    } else if (trimmed.startsWith('stiffness:')) {
      config.stiffness = parseArrayValue(line, lines, i);
    } else if (trimmed.startsWith('damping:')) {
      config.damping = parseArrayValue(line, lines, i);
    } else if (trimmed.startsWith('default_joint_pos:')) {
      config.default_joint_pos = parseArrayValue(line, lines, i);
    } else if (trimmed === 'commands:') {
      currentSection = 'commands';
    } else if (trimmed === 'actions:') {
      currentSection = 'actions';
    } else if (trimmed === 'observations:') {
      currentSection = 'observations';
    } else if (currentSection === 'observations' && !trimmed.includes(':') && !trimmed.startsWith('-')) {
      // Skip
    } else if (currentSection === 'observations' && trimmed.match(/^[a-z_]+:$/)) {
      // New observation term
      currentObsName = trimmed.slice(0, -1);
      config.observations[currentObsName] = {
        params: {},
        clip: null,
        scale: [],
        history_length: 1
      };
    } else if (currentSection === 'observations' && currentObsName) {
      // Parse observation properties
      if (trimmed.startsWith('params:')) {
        const paramsValue = trimmed.substring(7).trim();
        if (paramsValue === '{}') {
          config.observations[currentObsName].params = {};
        } else if (paramsValue.startsWith('{')) {
          // Parse inline dict like {command_name: base_velocity}
          const match = paramsValue.match(/\{([^:]+):\s*([^}]+)\}/);
          if (match) {
            const key = match[1].trim();
            const val = match[2].trim();
            config.observations[currentObsName].params[key] = val;
          }
        }
      } else if (trimmed.startsWith('clip:')) {
        const val = trimmed.split(':')[1].trim();
        config.observations[currentObsName].clip = val === 'null' ? null : val;
      } else if (trimmed.startsWith('scale:')) {
        config.observations[currentObsName].scale = parseArrayValue(line, lines, i);
      } else if (trimmed.startsWith('history_length:')) {
        config.observations[currentObsName].history_length = parseInt(trimmed.split(':')[1].trim());
      }
    } else if (currentSection === 'actions') {
      if (trimmed === 'JointPositionAction:') {
        config.actions.JointPositionAction = {
          scale: [],
          offset: []
        };
      } else if (trimmed.startsWith('scale:')) {
        config.actions.JointPositionAction.scale = parseArrayValue(line, lines, i);
      } else if (trimmed.startsWith('offset:')) {
        config.actions.JointPositionAction.offset = parseArrayValue(line, lines, i);
      }
    } else if (currentSection === 'commands') {
      // Parse commands section (base_velocity ranges)
      if (trimmed === 'base_velocity:') {
        config.commands.base_velocity = { ranges: {} };
      }
    }
  }
  
  return config;
}

/**
 * Parse array values that can span multiple lines
 */
function parseArrayValue(line, lines, startIdx) {
  const arr = [];
  const firstLine = line.trim();
  
  // Check if array starts on same line
  const colonIdx = firstLine.indexOf(':');
  if (colonIdx >= 0) {
    const afterColon = firstLine.substring(colonIdx + 1).trim();
    if (afterColon.startsWith('[')) {
      // Inline array, possibly multi-line
      let fullArray = afterColon;
      let idx = startIdx;
      
      // If not closed, continue reading lines
      while (!fullArray.includes(']') && idx < lines.length - 1) {
        idx++;
        fullArray += ' ' + lines[idx].trim();
      }
      
      // Parse the array
      const match = fullArray.match(/\[(.*?)\]/);
      if (match) {
        const items = match[1].split(',').map(s => {
          const trimmed = s.trim();
          // Try to parse as number
          const num = parseFloat(trimmed);
          return isNaN(num) ? trimmed : num;
        });
        return items;
      }
    }
  }
  
  return arr;
}

/**
 * Convert YAML config to the format expected by main.js
 * This creates a policy configuration similar to robust.json
 */
export function yamlConfigToPolicyConfig(yamlConfig, onnxPath) {
  // Determine observation size
  let obsSize = 0;
  const obsTermSizes = {};
  
  for (const [name, obs] of Object.entries(yamlConfig.observations)) {
    const baseSize = obs.scale.length;
    const totalSize = baseSize * obs.history_length;
    obsTermSizes[name] = { baseSize, totalSize };
    obsSize += totalSize;
  }
  
  const numJoints = yamlConfig.actions.JointPositionAction.scale.length;
  
  // Map stiffness and damping according to joint_ids_map
  const stiffness_array = [];
  const damping_array = [];
  for (let i = 0; i < numJoints; i++) {
    const joint_idx = yamlConfig.joint_ids_map[i];
    stiffness_array.push(yamlConfig.stiffness[joint_idx]);
    damping_array.push(yamlConfig.damping[joint_idx]);
  }
  
  console.log('YAML Config: obs_size=' + obsSize + ', num_joints=' + numJoints);
  
  return {
    onnx: {
      path: onnxPath,
      meta: {
        in_keys: ["obs"],
        out_keys: ["actions"]
      }
    },
    num_joints: numJoints,
    action_scale: yamlConfig.actions.JointPositionAction.scale, // Full array, not just first element
    stiffness: yamlConfig.stiffness[0], // Will be overridden by array below
    damping: yamlConfig.damping[0], // Will be overridden by array below
    stiffness_array: stiffness_array,
    damping_array: damping_array,
    control_type: "joint_position",
    default_joint_pos: yamlConfig.default_joint_pos,
    action_offset: yamlConfig.actions.JointPositionAction.offset,
    joint_ids_map: yamlConfig.joint_ids_map,
    step_dt: yamlConfig.step_dt,
    obs_config: {
      obs: [{
        name: "G1YAMLObs",
        yaml_config: yamlConfig,
        obs_terms: yamlConfig.observations,
        num_joints: numJoints
      }]
    }
  };
}

