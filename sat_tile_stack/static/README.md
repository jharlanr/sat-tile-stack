# `sat_tile_stack/static/`

Static assets served by the labeling GUI (`lakelabel`) at `/static/…`.

## `viewer.js` — the deck.gl image-panel widget

`viewer.js` is a **built, self-contained bundle** (~2.4 MB), committed on
purpose so the labeler runs without a separate JS build step. It renders each
lake's timestack client-side on the GPU (deck.gl + deck.gl-zarr), replacing the
old server-rendered matplotlib PNG panel. `index.html` loads it with a plain
`<script src="/static/viewer.js">` and drives it through
`window.SatTileViewer`:

```js
const v = SatTileViewer.create(el);          // mount into a container
const meta = await v.load("/api/zarr/lake42"); // {nFrames, dates, bands}
v.setFrame(3);
v.setBrightness(1.4);
v.zoomBy(0.5, cursorX, cursorY);             // zoom toward the cursor
v.destroy();
```

The Flask backend serves each lake's timestack as an on-demand GeoZarr store
under `/api/zarr/<lake_id>/…` (see `zarr_export.da_to_geozarr` and the routes
in `labeling.py`). Stores are written **uncompressed** so the bundle needs only
zarrita's built-in `bytes` codec — no decompression WASM to load — and the HTTP
responses are gzipped instead.

## Source and rebuilding

The widget's TypeScript source lives in a **separate repository**, not here:

```
deck.gl-raster/examples/qaanaaq-labeler-widget/src/index.ts
```

`viewer.js` is the Vite library-mode (IIFE) build of that entry. To rebuild
after changing the source:

```bash
cd deck.gl-raster/examples/qaanaaq-labeler-widget
pnpm build
cp dist/viewer.js /path/to/sat-tile-stack/sat_tile_stack/static/viewer.js
```

Then commit the updated `viewer.js`.

### Build notes / gotchas (baked into that example's `vite.config.ts`)

- **`process` shim.** Vite *library* mode does not replace
  `process.env.NODE_ENV` the way app builds do, so deck.gl/luma.gl dev checks
  reference an undefined `process` and throw on load. The config statically
  replaces `process.env.NODE_ENV` and injects a `process` shim banner.
- **Uncompressed stores.** The single-file IIFE can't load the separate
  `zstd_codec.wasm` / `blosc_codec.wasm`, so compressed stores render blank.
  Keep `da_to_geozarr` writing with `compressors=None`.
