"""Live, read-only scene inference server: six cameras, BEV and 3D occupancy.

Run with the training project's Python. Uses the model and data from
--project-root; does not train or change weights. Binds only to 127.0.0.1.
"""
from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import statistics
import sys
import threading
import time
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image

from visualize_voxels_3d import BOUNDS, NAMES, PALETTE, exposed_faces, top_down

CAMERA_DISPLAY = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
                  "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")
VISUALIZATION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = VISUALIZATION_ROOT.parent


def image_uri(image, format="PNG"):
    buffer = io.BytesIO()
    image.save(buffer, format=format, **({"quality": 75} if format == "JPEG" else {}))
    mime = "image/jpeg" if format == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def pack_mesh(labels, valid):
    """Pack exact outer cube faces as uint8 grid corners (0..200, 0..16).

    No occupied voxel subsampling. The browser restores metres and triangles.
    """
    faces, classes = exposed_faces(labels, valid)
    corners = np.rint((faces - BOUNDS[:3]) / .4).astype(np.uint8)
    return {"corners": base64.b64encode(corners.tobytes()).decode("ascii"),
            "classes": base64.b64encode(classes.astype(np.uint8).tobytes()).decode("ascii"),
            "faces": len(faces)}


def pack_centres(labels, valid):
    keep = valid & (labels < 17)
    indices = np.argwhere(keep).astype(np.uint8)
    return {"indices": base64.b64encode(indices.tobytes()).decode("ascii"),
            "classes": base64.b64encode(labels[keep].astype(np.uint8).tobytes()).decode("ascii"),
            "count": len(indices)}


class SceneEngine:
    def __init__(self, project_root, checkpoint_path, scene_name, device_name="auto"):
        sys.path.insert(0, str(project_root.resolve()))
        import torch
        from data.nuscenes_dataset import NuScenesOccupancyDataset, load_nuscenes
        from foundation4d.registry import build_model
        self.torch = torch
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.config = checkpoint["config"]
        self.epoch = checkpoint["epoch"]
        self.checkpoint_path = str(checkpoint_path.resolve())
        metadata = self.config["data"]
        if scene_name in metadata["validation_scenes"]:
            self.split = "val"
        elif scene_name in metadata["train_scenes"]:
            self.split = "train"
        else:
            raise ValueError("Scene is not in the checkpoint's train or validation split.")
        version = metadata.get('version','v1.0-mini')
        nusc = load_nuscenes(metadata["dataroot"], version=version, verbose=False,
            annotations_json=metadata.get('annotations_json'),image_root=metadata.get('image_root'))
        self.dataset = NuScenesOccupancyDataset(
            dataroot=metadata["dataroot"], occupancy_root=metadata["occupancy_root"],
            split=self.split, image_size=tuple(metadata["image_size"]), nusc=nusc,
            version=version, image_root=metadata.get('image_root'), scene_names=[scene_name],
        )
        if tuple(self.config["model"]["grid_shape"]) != self.dataset.grid_shape:
            raise ValueError("Checkpoint grid shape does not match the dataset.")
        if tuple(self.config["model"]["voxel_bounds"]) != self.dataset.voxel_bounds:
            raise ValueError("Checkpoint grid bounds do not match the dataset.")
        if tuple(metadata["camera_order"]) != self.dataset.camera_order:
            raise ValueError("Checkpoint and dataset camera orders differ.")
        self.indices = [i for i,r in enumerate(self.dataset.records) if r["scene_name"] == scene_name]
        if not self.indices:
            raise ValueError(f"No frames in {scene_name}.")
        timestamps = [nusc.get("sample", self.dataset.records[i]["sample_token"])["timestamp"]
                      for i in self.indices]
        if any(b <= a for a,b in zip(timestamps,timestamps[1:])):
            raise ValueError("Scene timestamps must be strictly increasing.")
        self.times = [(stamp-timestamps[0])/1e6 for stamp in timestamps]
        self.scene_name = scene_name
        self.device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                                   else "cpu" if device_name == "auto" else device_name)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        self.model = build_model(self.config["model"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.lock = threading.Lock()
        self.cache = {}
        self.benchmark = None

    def info(self):
        return {"scene": self.scene_name, "split": self.split, "epoch": self.epoch,
                "frames": len(self.indices), "times": self.times,
                "duration": self.times[-1], "device": str(self.device),
                "camera_order": CAMERA_DISPLAY, "palette": PALETTE.tolist(), "classes": NAMES,
                "benchmark": self.benchmark,
                "scope": "Exact labeled keyframes; camera-visible non-ignore region; ego-centred grid per frame. "
                         "Ground truth is used only for display, never supplied to the model."}

    def frame(self, index, fresh=False, geometry="cubes"):
        if not 0 <= index < len(self.indices):
            raise IndexError("Frame is outside this scene.")
        if geometry not in ("cubes", "centres"):
            raise ValueError("Geometry must be cubes or centres.")
        # One GPU worker; prevent concurrent requests from racing the model/cache.
        with self.lock:
            cache_key = (index, geometry)
            if cache_key in self.cache and not fresh:
                return {**self.cache[cache_key], "cache_hit": True}
            start = time.perf_counter()
            item = self.dataset[self.indices[index]]
            record = self.dataset.records[self.indices[index]]
            torch = self.torch
            inputs = [item[key].unsqueeze(0).to(self.device)
                      for key in ("images", "intrinsics", "ego_to_camera")]
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            forward_start = time.perf_counter()
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"
            ):
                logits = self.model(*inputs)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            model_ms = (time.perf_counter()-forward_start)*1000
            prediction = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            target = item["semantics"].numpy()
            valid = item["valid_mask"].numpy() & (target != 255)
            cameras = {}
            for camera, path in zip(self.dataset.camera_order, record["image_paths"]):
                with Image.open(path) as image:
                    rgb = image.convert("RGB").resize((384,216), Image.Resampling.BILINEAR)
                    cameras[camera] = image_uri(rgb, "JPEG")
            volumes = {}
            for key, labels in (("gt",target),("pred",prediction)):
                bev = PALETTE[top_down(labels,valid)].astype(np.uint8)
                # Row zero is ego X=-40; flip so forward (+X) is at the top.
                volumes[key] = {"bev": image_uri(Image.fromarray(bev[::-1]))}
                if geometry == "cubes":
                    volumes[key]["mesh"] = pack_mesh(labels,valid)
                else:
                    volumes[key]["centres"] = pack_centres(labels,valid)
            result = {"index": index, "time": self.times[index], "sample_token": item["sample_token"],
                      "cameras": cameras, "volumes": volumes, "model_ms": model_ms,
                      "backend_ms": (time.perf_counter()-start)*1000,
                      "valid_voxels": int(valid.sum()),
                      "predicted_occupied": int(((prediction < 17) & valid).sum()),
                      "gt_occupied": int(((target < 17) & valid).sum()), "cache_hit": False}
            result["geometry"] = geometry
            self.cache[cache_key] = result
            return result

    def benchmark_scene(self, destination):
        measurements = []
        for index in range(len(self.indices)):
            frame = self.frame(index, fresh=True)
            measurements.append({key: frame[key] for key in ("index","model_ms","backend_ms")})
            if index % 10 == 0 or index == len(self.indices)-1:
                print(f"Benchmark: {index+1}/{len(self.indices)}; model={frame['model_ms']:.1f} ms, "
                      f"backend={frame['backend_ms']:.1f} ms", flush=True)
        steady = measurements[min(3,len(measurements)-1):]
        average_model = statistics.mean(row["model_ms"] for row in steady)
        average_backend = statistics.mean(row["backend_ms"] for row in steady)
        interval = statistics.median(np.diff(self.times)) if len(self.times)>1 else 0
        self.benchmark = {
            "scene": self.scene_name, "epoch": self.epoch, "frames": len(self.indices),
            "device": str(self.device), "torch_version": str(self.torch.__version__),
            "checkpoint": self.checkpoint_path,
            "warmup_excluded": min(3,len(measurements)-1),
            "model_mean_ms": average_model, "model_fps": 1000/average_model,
            "backend_mean_ms": average_backend, "backend_fps": 1000/average_backend,
            "keyframe_hz": 1/interval if interval else None,
            "backend_includes": "Image/target loading, normalization, forward, argmax, six JPEGs, BEV and exact surface meshes.",
            "excludes": "JSON serialization, HTTP transfer and browser rendering; model-only FPS is not viewer FPS.",
            "measurements": measurements,
        }
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(self.benchmark,indent=2))
        print(json.dumps({k:v for k,v in self.benchmark.items() if k != "measurements"},indent=2),flush=True)


def make_handler(engine, page, plotly_file):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path)
            try:
                if path.path in ("/", "/index.html"):
                    content, mime = page.read_bytes(), "text/html; charset=utf-8"
                elif path.path == "/plotly.min.js":
                    content, mime = plotly_file.read_bytes(), "text/javascript"
                elif path.path == "/api/info":
                    content, mime = json.dumps(engine.info()).encode(), "application/json"
                elif path.path == "/api/frame":
                    query = parse_qs(path.query)
                    index = int(query.get("index",["0"])[0])
                    fresh = query.get("fresh",["0"])[0] == "1"
                    geometry = query.get("geometry",["cubes"])[0]
                    content = json.dumps(engine.frame(index,fresh,geometry),separators=(",",":")).encode()
                    mime = "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type",mime)
                self.send_header("Content-Length",str(len(content)))
                self.send_header("Cache-Control","no-store" if path.path.startswith("/api") else "no-cache")
                self.send_header("X-Content-Type-Options","nosniff")
                self.end_headers()
                self.wfile.write(content)
            except (ValueError,IndexError) as error:
                self.send_error(400,str(error))
            except (ConnectionResetError,BrokenPipeError):
                pass
            except Exception as error:
                print(f"Frame error: {error}",flush=True)
                self.send_error(500,"Could not process frame; check server output.")

        def log_message(self,format,*args):
            if args and str(args[1]) not in ("200","304"):
                super().log_message(format,*args)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=PROJECT_ROOT)
    parser.add_argument("--checkpoint",type=Path,default=PROJECT_ROOT/"foundation4d/outputs/lite_480_geometry_seed0/epoch_009.pt")
    parser.add_argument("--scene",default="scene-0103")
    parser.add_argument("--port",type=int,default=8767)
    parser.add_argument("--device",choices=["auto","cuda","cpu"],default="auto")
    parser.add_argument("--benchmark",action="store_true",help="Measure every scene frame before serving.")
    parser.add_argument("--benchmark-only",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=VISUALIZATION_ROOT/"outputs/scene_player")
    args = parser.parse_args()
    page = VISUALIZATION_ROOT/"scene_player.html"
    plotly = VISUALIZATION_ROOT/"_deps/plotly/package_data/plotly.min.js"
    if not plotly.is_file():
        raise FileNotFoundError("Plotly.js is missing. Install plotly into visualization/_deps.")
    engine = SceneEngine(args.project_root,args.checkpoint,args.scene,args.device)
    print(f"Ready: {engine.scene_name}; epoch {engine.epoch}; {len(engine.indices)} keyframes; {engine.device}",flush=True)
    if args.benchmark or args.benchmark_only:
        engine.benchmark_scene(args.output_dir/args.scene/"benchmark.json")
    if args.benchmark_only:
        return
    server = ThreadingHTTPServer(("127.0.0.1",args.port),make_handler(engine,page,plotly))
    print(f"Scene player: http://127.0.0.1:{args.port}/",flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
