# `widget/` — source for the deck.gl image-panel viewer

This folder is the **self-contained source** for
`sat_tile_stack/static/viewer.js`, the GPU image panel the labeling GUI
(`lakelabel`) renders each lake's timestack with. `viewer.js` is a built,
committed bundle so the labeler runs without a JS build step; this folder lets
anyone reproduce that bundle from a plain `git clone`, with **no other
repository checked out**.

The source was originally an example in
[`developmentseed/deck.gl-raster`](https://github.com/developmentseed/deck.gl-raster).
It has been vendored here and its three `@developmentseed/*` dependencies
pinned to the published npm release `0.8.0-beta.2` (instead of the monorepo's
`workspace:^`), so the build needs only the public npm registry.

## Build

```bash
cd widget
npm install
npm run build:install   # builds dist/viewer.js and copies it to ../sat_tile_stack/static/
```

`npm run build` alone leaves the bundle in `widget/dist/viewer.js` without
copying. After rebuilding, commit the updated
`sat_tile_stack/static/viewer.js`.

## How the page uses it

`index.html` loads the bundle with a plain `<script src="/static/viewer.js">`
and drives it through `window.SatTileViewer`:

```js
const v = SatTileViewer.create(el);            // mount into a container
const meta = await v.load("/api/zarr/lake42"); // {nFrames, dates, bands}
v.setFrame(3);
v.setBrightness(1.4);
v.zoomBy(0.5, cursorX, cursorY);               // zoom toward the cursor
v.destroy();
```

## Build notes / gotchas (baked into `vite.config.ts`)

- **`process` shim.** Vite *library* mode does not replace
  `process.env.NODE_ENV` the way app builds do, so deck.gl/luma.gl dev checks
  reference an undefined `process` and throw on load. The config statically
  replaces `process.env.NODE_ENV` and injects a `process` shim banner.
- **Uncompressed stores only.** The single-file IIFE can't load the separate
  `zstd_codec.wasm` / `blosc_codec.wasm`, so compressed GeoZarr stores render
  blank. Keep `sat_tile_stack.zarr_export.da_to_geozarr` writing with
  `compressors=None`.
