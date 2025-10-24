import { defineConfig } from 'vite';
import { copyFileSync, mkdirSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export default defineConfig({
  base: './',
  root: '.',
  publicDir: false, // We'll manually handle static files
  
  build: {
    outDir: 'build',
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
        // Copy MuJoCo WASM files to build output
        const distDir = join(__dirname, 'dist');
        const buildDir = join(__dirname, 'build');
        
        if (!existsSync(join(buildDir, 'dist'))) {
          mkdirSync(join(buildDir, 'dist'), { recursive: true });
        }
        
        // Copy mujoco WASM files
        ['mujoco_wasm.wasm', 'mujoco_wasm.js'].forEach(file => {
          const src = join(distDir, file);
          const dest = join(buildDir, 'dist', file);
          if (existsSync(src)) {
            copyFileSync(src, dest);
            console.log(`✓ Copied ${file}`);
          }
        });
        
        // Copy ONNX Runtime WASM files from node_modules
        const nodeModulesOrtDir = join(__dirname, 'node_modules', 'onnxruntime-web', 'dist');
        if (!existsSync(join(buildDir, 'ort'))) {
          mkdirSync(join(buildDir, 'ort'), { recursive: true });
        }
        
        ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs', 
         'ort-wasm-simd-threaded.jsep.wasm', 'ort-wasm-simd-threaded.jsep.mjs'].forEach(file => {
          const src = join(nodeModulesOrtDir, file);
          const dest = join(buildDir, 'ort', file);
          if (existsSync(src)) {
            copyFileSync(src, dest);
            console.log(`✓ Copied ${file}`);
          }
        });
      }
    }
  ]
});

