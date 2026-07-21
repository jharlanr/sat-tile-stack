import { resolve } from "node:path";
import { defineConfig } from "vite";

// Build a single self-contained IIFE bundle exposing `window.SatTileViewer`,
// so the vanilla labeling page can load it with a plain <script> tag.
export default defineConfig({
  // Vite library mode does NOT replace `process.env.NODE_ENV` the way app
  // builds do, so deck.gl/luma.gl's `process.env.NODE_ENV !== "production"`
  // dev checks reference an undefined `process` and throw in the browser.
  // Replace it statically, and shim the remaining Node `process.*` references
  // (version/hrtime/browser, pulled in by a dependency) via a banner.
  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
  },
  build: {
    lib: {
      entry: resolve(__dirname, "src/index.ts"),
      name: "SatTileViewer",
      formats: ["iife"],
      fileName: () => "viewer.js",
    },
    rollupOptions: {
      external: [],
      output: {
        banner:
          'globalThis.process=globalThis.process||{env:{NODE_ENV:"production"},' +
          'browser:true,version:"v18.0.0",versions:{node:"18.0.0"},' +
          "platform:\"browser\",nextTick:function(cb){Promise.resolve().then(cb)}," +
          "hrtime:function(){return[0,0]}};",
      },
    },
    minify: true,
    target: "es2022",
  },
  worker: { format: "es" },
});
