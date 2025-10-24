# Bug Fixes Applied

## Issue 1: Critical Angular Velocity Bug ✅ FIXED
**Problem:** Base angular velocity was reading linear velocity instead
**Location:** `observationHelpers.js` line 106
**Fix:** Changed `qvel[this.base_qvel_adr + i]` to `qvel[this.base_qvel_adr + 3 + i]`
**Impact:** This was causing the sim2sim gap - policies couldn't sense rotational motion!

## Issue 2: Node.js Version Compatibility ✅ FIXED
**Problem:** Vite 7 requires Node 20+, but system has Node 18.19.1
**Location:** `package.json`
**Fix:** Downgraded to Vite 5.4.11 (compatible with Node 18)

## Issue 3: ONNX Runtime WASM Loading ✅ FIXED
**Problem:** ONNX Runtime couldn't find WASM files (wrong path configuration)
**Locations:** 
- `index.html` - Removed import map (Vite handles this)
- `onnxHelper.js` - Fixed WASM path from `/mujoco_wasm/node_modules/...` to `/ort/`
- `vite.config.js` - Added middleware to serve ONNX Runtime files with correct MIME types

**Fix Details:**
```javascript
// Before (WRONG):
ort.env.wasm.wasmPaths = "/mujoco_wasm/node_modules/onnxruntime-web/dist/";

// After (CORRECT):
ort.env.wasm.wasmPaths = isProduction ? './ort/' : '/ort/';
```

## Testing Instructions

### 1. Start Dev Server
```bash
npm run dev
```

Should see:
```
VITE v5.4.21  ready in XXX ms
➜  Local:   http://localhost:3000/
```

### 2. Open in Browser
Navigate to: `http://localhost:3000`

### 3. Expected Results
✅ **No WASM MIME type warnings** in console
✅ **Policy loads successfully** (see "Policy loaded: actions=23..." in console)
✅ **Robot stands stable** in simulation
✅ **Can switch between policies** (Baseline / Ours) in GUI

### 4. What to Check
1. **Console logs** should show:
   - "Policy loaded: actions=23, control_rate=0.02s..."
   - "ONNX model loaded successfully"
   - Base position, velocity, and joint states

2. **Simulation** should show:
   - G1 robot in standing pose
   - No immediate falling
   - Responds to control inputs

3. **No errors** about:
   - ❌ "no available backend found"
   - ❌ "disallowed MIME type"
   - ❌ "error loading dynamically imported module"

## Files Modified

| File | Changes |
|------|---------|
| `observationHelpers.js` | Fixed angular velocity offset bug |
| `index.html` | Removed import map (Vite handles imports) |
| `onnxHelper.js` | Fixed ONNX Runtime WASM paths |
| `vite.config.js` | Added WASM file serving, MIME type handling |
| `package.json` | Downgraded Vite to 5.4.11 for Node 18 compatibility |
| `_headers` | Enhanced for GitHub Pages deployment |

## Clean Code Removed
- ❌ All Go2 quadruped observation classes
- ❌ Go2 mesh file downloads
- ❌ Unused observation helpers (VelocityCommand, ImpedanceCommand, etc.)
- ❌ Verbose logging (kept only errors)

## Production Deployment

When ready to deploy to GitHub Pages:

```bash
# Build optimized bundle
npm run build

# Commit and push
git add .
git commit -m "Fixed WASM loading and sim2sim bugs"
git push origin main
```

The `_headers` file ensures GitHub Pages serves WASM files correctly.

## Troubleshooting

### If policies still don't work:
1. Check console for observation values
2. Verify base_ang_vel is small when standing still (< 0.1)
3. Check projected_gravity is close to [0, 0, -1] when upright

### If ONNX errors persist:
1. Verify `/ort/` directory has WASM files: `ls ort/*.wasm`
2. Check browser console network tab - should see 200 responses for WASM files
3. Try clearing browser cache (Ctrl+Shift+R)

### If Vite won't start:
```bash
# Clean reinstall
rm -rf node_modules package-lock.json
npm install
npm run dev
```

## Performance Notes

- **Control rate:** 50Hz (0.02s policy step)
- **Physics rate:** 200Hz (0.005s simulation step)  
- **Control decimation:** 4:1 (policy runs every 4 physics steps)
- **Observation history:** 5 timesteps
- **Total observation size:** 390 (6 terms × 5 history × various dimensions)

## Success Criteria

✅ Vite dev server starts without errors
✅ Browser loads simulation without WASM errors  
✅ Policy loads and initializes (console shows "Policy loaded")
✅ Robot maintains balance (doesn't immediately fall)
✅ Both policies work (Baseline and Ours)
✅ No MIME type warnings in console

**All issues should now be resolved!** 🎉

