/**
 * SatTileViewer — a self-contained, framework-free deck.gl widget for the
 * sat-tile-stack labeling GUI. It replaces the server-rendered matplotlib PNG
 * image panel with client-side GPU rendering of a per-lake GeoZarr store
 * (deck.gl-zarr), so frame scrubbing is instant and pixels stay crisp
 * (nearest-neighbour, no interpolation) at any zoom.
 *
 * Loaded as an IIFE (`<script src="/static/viewer.js">`), it exposes
 * `window.SatTileViewer.create(container)`:
 *
 *     const v = SatTileViewer.create(el);
 *     const meta = await v.load("/api/zarr/lake42");  // {nFrames, dates, bands}
 *     v.setFrame(3);
 *     v.setBrightness(1.4);
 *     v.destroy();
 */
import { Deck, MapView, WebMercatorViewport } from "@deck.gl/core";
import type { MinimalTileData } from "@developmentseed/deck.gl-raster";
import type { GetTileDataOptions } from "@developmentseed/deck.gl-zarr";
import { ZarrLayer } from "@developmentseed/deck.gl-zarr";
import { parseWkt } from "@developmentseed/proj";
import type { Device, Texture } from "@luma.gl/core";
import proj4 from "proj4";
import * as zarr from "zarrita";

type TileData = MinimalTileData & { image: Texture };

interface ViewState {
  longitude: number;
  latitude: number;
  zoom: number;
  minZoom?: number;
  maxZoom?: number;
}

interface LoadResult {
  nFrames: number;
  dates: string[];
  bands: string[];
}

function toRgbaBytes(
  result: zarr.Chunk<zarr.NumberDataType>,
  width: number,
  height: number,
  brightness: number,
): Uint8Array {
  const { data } = result;
  const scale = (255 / 10000) * brightness;
  const px = width * height;
  const rgba = new Uint8Array(px * 4);
  const gOff = px;
  const bOff = px * 2;
  for (let i = 0; i < px; i++) {
    rgba[i * 4 + 0] = Math.min(255, data[i]! * scale);
    rgba[i * 4 + 1] = Math.min(255, data[gOff + i]! * scale);
    rgba[i * 4 + 2] = Math.min(255, data[bOff + i]! * scale);
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

function toNearestTexture(
  device: Device,
  rgba: Uint8Array,
  width: number,
  height: number,
): Texture {
  return device.createTexture({
    format: "rgba8unorm",
    width,
    height,
    data: rgba,
    mipLevels: 1,
    sampler: {
      minFilter: "nearest",
      magFilter: "nearest",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    },
  });
}

class SatTileViewerInstance {
  private container: HTMLElement;
  private deck: Deck<MapView>;
  private node: zarr.Array<zarr.DataType, zarr.Readable> | null = null;
  private viewState: ViewState = { longitude: 0, latitude: 0, zoom: 1 };
  // Centred "home" view for the current store. Frame changes snap the pan back
  // here (keeping the user's current zoom) so every frame is presented centred.
  private homeLngLat: { longitude: number; latitude: number } = {
    longitude: 0,
    latitude: 0,
  };
  private frame = 0;
  private brightness = 1.2;
  private zarrUrl = "";

  constructor(container: HTMLElement) {
    this.container = container;
    this.deck = new Deck<MapView>({
      parent: container as HTMLDivElement,
      views: new MapView({ repeat: false }),
      // Scroll is reserved for frame-scrubbing by the labeling page, so disable
      // scroll-zoom here; zoom via double-click, pan via drag.
      controller: {
        dragRotate: false,
        scrollZoom: false,
        doubleClickZoom: true,
        dragPan: true,
      },
      initialViewState: this.viewState,
      viewState: this.viewState,
      onViewStateChange: ({ viewState }) => {
        this.viewState = viewState as ViewState;
        this.deck.setProps({ viewState: this.viewState });
      },
      // Dark canvas; deck's transparent clear lets it show around the tile.
      style: { background: "#111" },
    });
  }

  /** Open a GeoZarr store, recentre on it, and render its first frame. */
  async load(zarrUrl: string): Promise<LoadResult> {
    try {
      return await this._load(zarrUrl);
    } catch (e) {
      console.error("SatTileViewer.load failed for", zarrUrl, e);
      this.container.setAttribute(
        "data-viewer-error",
        e instanceof Error ? e.message : String(e),
      );
      throw e;
    }
  }

  private async _load(zarrUrl: string): Promise<LoadResult> {
    // zarrita's FetchStore needs an absolute URL.
    const abs = new URL(zarrUrl, window.location.href).href;
    this.zarrUrl = abs;
    const store = new zarr.FetchStore(abs);
    const arr = await zarr.open(store, { kind: "array" });
    this.node = arr as zarr.Array<zarr.DataType, zarr.Readable>;

    const attrs = arr.attrs as {
      "sts:dates"?: string[];
      "sts:bands"?: string[];
      "spatial:transform": number[];
      "spatial:shape": [number, number];
      "proj:wkt2": string;
    };
    const dates = attrs["sts:dates"] ?? [];
    const bands = attrs["sts:bands"] ?? [];

    this.viewState = this.computeView(
      attrs["spatial:transform"],
      attrs["spatial:shape"],
      attrs["proj:wkt2"],
    );
    this.homeLngLat = {
      longitude: this.viewState.longitude,
      latitude: this.viewState.latitude,
    };
    this.frame = 0;
    this.render();
    return { nFrames: dates.length, dates, bands };
  }

  /** Centre the map on the store and pick a zoom that fits the tile. */
  private computeView(
    transform: number[],
    shape: [number, number],
    wkt: string,
  ): ViewState {
    const [a, , c, , e, f] = transform;
    const [H, W] = shape;
    const cx = a! * (W / 2) + c!;
    const cy = e! * (H / 2) + f!;
    // biome-ignore lint/suspicious/noExplicitAny: proj4 typings don't cover parseWkt output
    const [lon, lat] = proj4(parseWkt(wkt) as any, "EPSG:4326").forward([cx, cy]);

    const tileMeters = Math.abs(a!) * W;
    const viewPx = this.container.clientWidth || 512;
    const mpp = tileMeters / (viewPx * 0.9); // fit ~90% of the panel width
    const worldMpp = (156543.03392 * Math.cos((lat * Math.PI) / 180)) / 1;
    const zoom = Math.log2(worldMpp / mpp);
    return { longitude: lon, latitude: lat, zoom, minZoom: zoom - 3, maxZoom: zoom + 8 };
  }

  private makeLayer(): ZarrLayer<zarr.Readable, zarr.DataType, TileData> | null {
    if (!this.node) return null;
    const brightness = this.brightness;
    return new ZarrLayer<zarr.Readable, zarr.DataType, TileData>({
      // Frame in the id remounts the layer: RasterTileLayer doesn't forward an
      // updateTriggers.getTileData, so changing selection alone won't refetch.
      id: `tile-t${this.frame}-b${brightness.toFixed(2)}-${this.zarrUrl}`,
      node: this.node,
      selection: { time: this.frame, band: null },
      async getTileData(arr, options: GetTileDataOptions) {
        const result = await zarr.get(
          arr as zarr.Array<zarr.NumberDataType, zarr.Readable>,
          options.sliceSpec,
          { signal: options.signal },
        );
        const { width, height, device } = options;
        const rgba = toRgbaBytes(result, width, height, brightness);
        const image = toNearestTexture(device, rgba, width, height);
        return { image, width, height, byteLength: rgba.byteLength };
      },
      renderTile: (d) => ({ image: d.image }),
      onTileError: (err: Error) =>
        console.error("SatTileViewer tile load/decode error:", err),
    });
  }

  private render() {
    const layer = this.makeLayer();
    this.deck.setProps({
      viewState: this.viewState,
      layers: layer ? [layer] : [],
    });
  }

  setFrame(i: number) {
    if (i === this.frame) return;
    this.frame = i;
    // Recentre on the store (dropping any pan) but keep the current zoom, so
    // scrubbing always presents the frame centred instead of wherever the user
    // last dragged to.
    this.viewState = {
      ...this.viewState,
      longitude: this.homeLngLat.longitude,
      latitude: this.homeLngLat.latitude,
    };
    this.render();
  }

  /**
   * Zoom by `dz` levels, keeping the geographic point under screen pixel
   * (px, py) fixed. Called by the page for Ctrl+scroll / pinch, so plain
   * scroll stays free for frame-scrubbing. Omit px/py to zoom to centre.
   */
  zoomBy(dz: number, px?: number, py?: number) {
    const vs = this.viewState;
    const width = this.container.clientWidth || 512;
    const height = this.container.clientHeight || 512;
    const lo = vs.minZoom ?? vs.zoom - 5;
    const hi = vs.maxZoom ?? vs.zoom + 10;
    const newZoom = Math.max(lo, Math.min(hi, vs.zoom + dz));
    if (newZoom === vs.zoom) return;

    let { longitude, latitude } = vs;
    if (px != null && py != null) {
      const before = new WebMercatorViewport({
        width, height, longitude, latitude, zoom: vs.zoom,
      });
      const lngLat = before.unproject([px, py]);
      const after = new WebMercatorViewport({
        width, height, longitude, latitude, zoom: newZoom,
      });
      const [x2, y2] = after.project(lngLat);
      // Re-centre so the point that was under the cursor returns there.
      const c = after.unproject([width / 2 + (x2 - px), height / 2 + (y2 - py)]);
      longitude = c[0];
      latitude = c[1];
    }
    this.viewState = { ...vs, longitude, latitude, zoom: newZoom };
    this.deck.setProps({ viewState: this.viewState });
  }

  setBrightness(b: number) {
    this.brightness = b;
    this.render();
  }

  destroy() {
    this.deck.finalize();
  }
}

/** Public factory — the only entry point the labeling page calls. */
export function create(container: HTMLElement): SatTileViewerInstance {
  return new SatTileViewerInstance(container);
}
