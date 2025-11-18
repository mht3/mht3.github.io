import { defineConfig } from 'vite';
import { copyFileSync, mkdirSync, existsSync, cpSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export default defineConfig({
  base: './',
  root: '.',
  publicDir: false, // We'll manually handle static files
  
  build: {
    outDir: '../humanoid/demo',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: 'index.html'
      },
      output: {
        // Keep clean file names for easier debugging
        entryFileNames: 'assets/[name].js',
        chunkFileNames: 'assets/[name].js',
        assetFileNames: 'assets/[name].[ext]'
      }
    },
    // Required for top-level await in onnxruntime-web
    target: 'esnext',
    // Increase chunk size warning limit for WASM files
    chunkSizeWarningLimit: 10000
  },
  
  server: {
    port: 3000,
    headers: {
      // Required for SharedArrayBuffer support in ONNX Runtime
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin'
    },
    fs: {
      // Allow serving files from project root
      strict: false
    }
  },
  
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
    esbuildOptions: {
      target: 'esnext'
    }
  },
  
  plugins: [
    {
      name: 'copy-wasm-files',
      configureServer(server) {
        // Serve ONNX Runtime WASM files with correct MIME type during dev
        server.middlewares.use((req, res, next) => {
          if (req.url?.includes('/ort/') && req.url?.endsWith('.wasm')) {
            res.setHeader('Content-Type', 'application/wasm');
          }
          if (req.url?.includes('/ort/') && req.url?.endsWith('.mjs')) {
            res.setHeader('Content-Type', 'application/javascript');
          }
          if (req.url?.includes('/dist/') && req.url?.endsWith('.wasm')) {
            res.setHeader('Content-Type', 'application/wasm');
          }
          next();
        });
      },
      closeBundle() {
        const outDir = join(__dirname, '..', 'humanoid', 'demo');
        
        // Copy MuJoCo WASM files
        const distDir = join(__dirname, 'node_modules', 'mujoco-js', 'dist');
        if (!existsSync(join(outDir, 'dist'))) {
          mkdirSync(join(outDir, 'dist'), { recursive: true });
        }
        ['mujoco_wasm.wasm', 'mujoco_wasm.js'].forEach(file => {
          const src = join(distDir, file);
          const dest = join(outDir, 'dist', file);
          if (existsSync(src)) {
            copyFileSync(src, dest);
          }
        });
        
        // Copy ONNX Runtime WASM files from node_modules
        const nodeModulesOrtDir = join(__dirname, 'node_modules', 'onnxruntime-web', 'dist');
        if (!existsSync(join(outDir, 'ort'))) {
          mkdirSync(join(outDir, 'ort'), { recursive: true });
        }
        ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs', 
         'ort-wasm-simd-threaded.jsep.wasm', 'ort-wasm-simd-threaded.jsep.mjs'].forEach(file => {
          const src = join(nodeModulesOrtDir, file);
          const dest = join(outDir, 'ort', file);
          if (existsSync(src)) {
            copyFileSync(src, dest);
          }
        });
        
        // Copy examples directory (scenes, policies, assets)
        const examplesDir = join(__dirname, 'examples');
        const outExamplesDir = join(outDir, 'examples');
        if (existsSync(examplesDir)) {
          cpSync(examplesDir, outExamplesDir, { recursive: true });
        }
        
        // Copy _headers for GitHub Pages MIME type configuration
        const headersFile = join(__dirname, '_headers');
        const outHeadersFile = join(outDir, '_headers');
        if (existsSync(headersFile)) {
          copyFileSync(headersFile, outHeadersFile);
        }
      }
    }
  ]
});

