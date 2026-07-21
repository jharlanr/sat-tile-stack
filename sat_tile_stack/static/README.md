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

The widget's TypeScript source is vendored in this repo under
[`widget/`](../../widget/) — self-contained, with no other repository required.
`viewer.js` is the Vite library-mode (IIFE) build of `widget/src/index.ts`.
To rebuild after changing the source:

```bash
cd widget
npm install
npm run build:install   # builds and copies dist/viewer.js here
```

Then commit the updated `viewer.js`. See [`widget/README.md`](../../widget/README.md)
for build notes and gotchas (the `process` shim and the uncompressed-store
requirement).
