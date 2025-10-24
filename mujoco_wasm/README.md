# MuJoCo WASM - Unitree G1 Balance Policy

Interactive MuJoCo simulation running in the browser with ONNX Runtime for neural network policies.

## Features

- ✨ Unitree G1 humanoid robot simulation
- 🧠 Real-time ONNX policy inference
- 🌐 Runs entirely in the browser via WebAssembly
- 📦 Self-contained build (no CDN dependencies)

## Development

### Prerequisites

```bash
npm install
```

### Local Development (Vite)

The recommended way for development:

```bash
npm run dev
```

This starts a Vite dev server at `http://localhost:3000` with:
- ✅ Proper WASM MIME types
- ✅ Hot module replacement
- ✅ Fast rebuilds

### Jekyll Development (GitHub Pages preview)

To test exactly as it will appear on GitHub Pages:

```bash
npm run jekyll:serve
```

This serves at `http://localhost:4000/mujoco_wasm/`

**Note:** WASM MIME type warnings are expected with Jekyll's built-in server, but the fallback to ArrayBuffer works fine. GitHub Pages deployment will have proper MIME types via the `_headers` file.

## Building for Production

Build optimized bundle:

```bash
npm run build
```

This creates a `build/` directory with all assets bundled and ready for deployment.

### Deploy to GitHub Pages

The `_headers` file configures Netlify/GitHub Pages to serve WASM files with correct MIME types:
- `application/wasm` for `.wasm` files
- Proper CORS headers for SharedArrayBuffer support

Simply push to GitHub and the site will automatically deploy with correct MIME types.

## Project Structure

```
mujoco_wasm/
├── index.html              # Entry point
├── examples/
│   ├── main.js            # Main application logic
│   ├── observationHelpers.js  # G1 observation processing
│   ├── mujocoUtils.js     # MuJoCo utilities
│   ├── yamlParser.js      # Policy configuration parser
│   ├── onnxHelper.js      # ONNX Runtime wrapper
│   ├── scenes/            # Robot models
│   └── checkpoints/       # Neural network policies
├── dist/                  # MuJoCo WASM binaries
├── ort/                   # ONNX Runtime WASM files
├── _headers              # GitHub Pages MIME configuration
└── vite.config.js        # Build configuration
```

## Policies

Two balance policies are available:

- **Baseline**: Standard MLP policy
- **Ours**: State projection architecture

Both trained in Isaac Lab and deployed to MuJoCo WASM.

## Troubleshooting

### MIME Type Warnings

If you see warnings about MIME types during Jekyll serve:
- ✅ This is expected with Jekyll's development server
- ✅ The fallback to ArrayBuffer instantiation works fine
- ✅ Production deployment will have correct MIME types

### WASM Loading Errors

1. Ensure WASM files exist in `dist/` and `ort/`
2. Check browser console for specific errors
3. Try clearing browser cache

### Build Issues

```bash
# Clean install
rm -rf node_modules package-lock.json
npm install

# Rebuild
npm run build
```

## License

See LICENSE file for details.
