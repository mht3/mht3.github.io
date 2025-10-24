# Setup Guide: Local Development & GitHub Pages Deployment

This guide will help you set up local development with proper WASM support and deploy to GitHub Pages without MIME type issues.

## Problem Summary

- **Jekyll's dev server** doesn't serve WASM files with correct MIME types (`application/wasm`)
- **GitHub Pages** needs special configuration to serve WASM correctly
- **CDN dependencies** don't work offline and create version control issues

## Solution: Vite + Proper Headers

We use **Vite** for local development (fast, proper MIME types) and **_headers** file for GitHub Pages deployment.

---

## Installation

### 1. Check Node.js Version

This project requires **Node.js 18.x or higher**:

```bash
node --version  # Should show v18.x or higher
```

If you need to upgrade Node.js:
- **Using nvm (recommended):** `nvm install 18 && nvm use 18`
- **Direct download:** https://nodejs.org/

### 2. Install Dependencies

```bash
cd mujoco_wasm
npm install
```

This installs:
- `three` - 3D graphics library
- `onnxruntime-web` - Neural network inference
- `vite@5.x` - Build tool (compatible with Node 18)

---

## Local Development

### Option A: Vite Dev Server (Recommended)

**Best for active development:**

```bash
npm run dev
```

**Features:**
- ✅ Runs at `http://localhost:3000`
- ✅ Proper WASM MIME types
- ✅ Hot module replacement
- ✅ Fast rebuilds
- ✅ Source maps for debugging

**Open in browser:** `http://localhost:3000`

### Option B: Jekyll Serve (GitHub Pages Preview)

**Best for testing exact deployment behavior:**

```bash
npm run jekyll:serve
```

**Features:**
- ✅ Runs at `http://localhost:4000/mujoco_wasm/`
- ⚠️ WASM MIME type warnings (expected, fallback works)
- ✅ Matches GitHub Pages environment

**Expected warnings (safe to ignore):**
```
wasm streaming compile failed: TypeError: WebAssembly: Response has unsupported MIME type...
falling back to ArrayBuffer instantiation
```

These warnings occur because Jekyll's dev server doesn't configure MIME types, but the ArrayBuffer fallback works perfectly. **GitHub Pages deployment will have correct MIME types.**

---

## Building for Production

### Build Command

```bash
npm run build
```

This creates optimized production bundles in `build/`:
- Minified JavaScript
- Bundled modules
- Copied WASM files
- Optimized assets

### What Gets Built

```
build/
├── index.html           # Entry point
├── assets/
│   ├── main.js         # Bundled application
│   └── [other chunks]
├── dist/
│   ├── mujoco_wasm.wasm
│   └── mujoco_wasm.js
├── ort/
│   ├── ort-wasm-simd-threaded.wasm
│   └── ort-wasm-simd-threaded.mjs
└── examples/           # Scene files, models, policies
```

---

## GitHub Pages Deployment

### Automatic Headers Configuration

The `_headers` file tells GitHub Pages/Netlify how to serve files:

```
# MIME types for WebAssembly files
/*.wasm
  Content-Type: application/wasm
  
/dist/*.wasm
  Content-Type: application/wasm

/ort/*.wasm
  Content-Type: application/wasm

# Enable SharedArrayBuffer for ONNX Runtime threading
/*
  Cross-Origin-Embedder-Policy: require-corp
  Cross-Origin-Opener-Policy: same-origin
```

### Jekyll Configuration

The `_config.yml` ensures Jekyll doesn't mess with your files:

```yaml
include:
  - _headers

keep_files:
  - dist
  - ort
  - node_modules
```

### Deploy Steps

1. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Update MuJoCo WASM app"
   ```

2. **Push to GitHub:**
   ```bash
   git push origin main
   ```

3. **GitHub Actions** (if configured) or **GitHub Pages** will automatically:
   - Copy files to `_site/`
   - Apply headers from `_headers`
   - Serve with correct MIME types ✅

---

## Troubleshooting

### 1. MIME Type Warnings in Jekyll

**Symptom:**
```
wasm streaming compile failed: TypeError: WebAssembly: Response has unsupported MIME type...
```

**Solution:** This is **expected and harmless** in Jekyll dev server. The fallback to ArrayBuffer works fine. Use `npm run dev` for development if warnings bother you.

### 2. Module Not Found Errors

**Symptom:**
```
Error: Cannot find module 'three'
```

**Solution:**
```bash
rm -rf node_modules package-lock.json
npm install
```

### 3. Vite Build Errors

**Symptom:**
```
Error: Build failed
```

**Solution:** Check that WASM files exist:
```bash
ls dist/mujoco_wasm.wasm
ls ort/ort-wasm-simd-threaded.wasm
```

If missing, regenerate MuJoCo WASM or reinstall onnxruntime-web.

### 4. GitHub Pages 404

**Symptom:** Page loads but resources return 404

**Solution:** 
1. Check GitHub Pages settings (Settings → Pages)
2. Ensure `_headers` file is committed
3. Check `_config.yml` includes necessary paths

### 5. SharedArrayBuffer Errors

**Symptom:**
```
SharedArrayBuffer is not defined
```

**Solution:** Ensure headers are set correctly:
- `Cross-Origin-Embedder-Policy: require-corp`
- `Cross-Origin-Opener-Policy: same-origin`

These are in `_headers` for GitHub Pages and in `vite.config.js` for local dev.

---

## File Structure

```
mujoco_wasm/
├── index.html              # Entry HTML
├── package.json           # Dependencies & scripts
├── vite.config.js         # Vite configuration
├── _headers              # GitHub Pages MIME config
├── _config.yml           # Jekyll configuration
├── .gitignore            # Don't commit build/, node_modules/
│
├── examples/
│   ├── main.js           # Main application
│   ├── observationHelpers.js  # Observations
│   ├── mujocoUtils.js    # MuJoCo utilities
│   ├── yamlParser.js     # Policy config parser
│   ├── onnxHelper.js     # ONNX wrapper
│   ├── scenes/           # Robot models
│   └── checkpoints/      # Neural policies
│
├── dist/                 # MuJoCo WASM (committed)
│   ├── mujoco_wasm.wasm
│   └── mujoco_wasm.js
│
├── ort/                  # ONNX Runtime WASM (committed)
│   ├── ort-wasm-simd-threaded.wasm
│   └── ort-wasm-simd-threaded.mjs
│
└── build/               # Build output (gitignored)
```

---

## Benefits of This Setup

### ✅ Local Development
- Fast Vite dev server with HMR
- Proper WASM MIME types
- No warnings or fallbacks
- Full source maps

### ✅ Version Control
- All dependencies in `package.json`
- No CDN links
- Reproducible builds
- Offline development

### ✅ GitHub Pages
- Automatic deployment
- Correct MIME types via `_headers`
- No streaming compile warnings
- Fast load times

### ✅ Maintainability
- Clear separation of dev/prod
- Easy dependency updates
- Standard npm workflow
- Clean file structure

---

## Quick Reference

```bash
# Development
npm run dev              # Vite dev server (recommended)
npm run jekyll:serve     # Jekyll server (Pages preview)

# Production
npm run build           # Build optimized bundle
npm run preview         # Preview production build

# Deployment
git push origin main    # Deploy to GitHub Pages
```

---

## Next Steps

1. **Start development:** `npm run dev`
2. **Make changes** to code in `examples/`
3. **Test locally** in browser
4. **Build for production:** `npm run build`
5. **Commit and push** to deploy

Your site will be live at: `https://yourusername.github.io/mujoco_wasm/`

**No more MIME type issues! 🎉**

