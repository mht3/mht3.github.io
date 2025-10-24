# Jekyll + Vite Integration Fix

## The Problem

**Before:** Jekyll was serving **source files** from `mujoco_wasm/`, which have bare module specifiers like `import * as THREE from 'three';` that don't work in browsers without bundling.

```
bundle exec jekyll serve → mujoco_wasm/index.html (source)
                         → ❌ "three" was a bare specifier error
```

## The Solution

Jekyll now serves **built files** from `mujoco_wasm/docs/` where Vite has bundled everything.

```
bundle exec jekyll serve → mujoco_wasm/docs/index.html (built)
                         → ✅ Everything bundled, works!
```

---

## What Changed

### 1. Root `_config.yml`

**Excluded** mujoco_wasm source files:
```yaml
exclude:
  - mujoco_wasm/examples
  - mujoco_wasm/dist
  - mujoco_wasm/ort
  - mujoco_wasm/index.html
  - mujoco_wasm/vite.config.js
  # ... etc
```

**Marked** `mujoco_wasm/docs/` as static:
```yaml
defaults:
  - scope:
      path: mujoco_wasm/docs/
    values:
      sitemap: false
```

### 2. Button Link Updated

**Before:**
```html
<a href="/mujoco_wasm/">
```

**After:**
```html
<a href="/mujoco_wasm/docs/">
```

---

## How to Use

### Development (Inside mujoco_wasm/)
```bash
cd mujoco_wasm
npm run dev
# Opens http://localhost:3000/
```

### Build (Inside mujoco_wasm/)
```bash
cd mujoco_wasm
npm run build
# Creates docs/ with bundled files
```

### Test Locally with Jekyll (From root)
```bash
cd /home/mht/Documents/mht3.github.io
bundle exec jekyll serve
# Visit http://localhost:4000/mujoco_wasm/docs/
```

### Deploy to GitHub Pages
```bash
cd /home/mht/Documents/mht3.github.io
git add mujoco_wasm/docs/ _config.yml _featured_categories/humanoid.md
git commit -m "Add MuJoCo WASM demo"
git push
```

**No special GitHub Pages configuration needed!** GitHub Pages will serve from the root, and your files are at `/mujoco_wasm/docs/`.

---

## URL Structure

```
Development:
  npm run dev        → http://localhost:3000/

Local Jekyll:
  jekyll serve       → http://localhost:4000/mujoco_wasm/docs/

GitHub Pages:
  Your site          → https://mht3.github.io/mujoco_wasm/docs/
```

---

## Important Notes

1. **Always build before committing:**
   ```bash
   cd mujoco_wasm
   npm run build
   git add docs/
   ```

2. **Don't commit node_modules:**
   - Already in `.gitignore`

3. **Keep source and build separate:**
   - Source: `mujoco_wasm/index.html`, `mujoco_wasm/examples/`
   - Built: `mujoco_wasm/docs/`
   - Jekyll only serves the built files

4. **Both URLs work:**
   - `/mujoco_wasm/docs/` ← **Recommended** (clean)
   - `/mujoco_wasm/docs/index.html` ← Also works

---

## Troubleshooting

### "bare specifier" error?
→ Make sure you're visiting `/mujoco_wasm/docs/`, not `/mujoco_wasm/`

### Files not updating?
→ Run `npm run build` in the `mujoco_wasm/` directory

### Jekyll serving wrong files?
→ Check that root `_config.yml` excludes source files

### 404 on GitHub Pages?
→ Make sure `docs/` directory is committed and pushed

### WASM MIME type errors with `bundle exec jekyll serve`?
→ **This is expected!** Jekyll's local server doesn't support custom MIME types.

**For local testing:**
- Use `npm run dev` inside `mujoco_wasm/` directory
- Or just deploy and test on GitHub Pages

**GitHub Pages will work correctly** because it respects the `_headers` file in `docs/`.

