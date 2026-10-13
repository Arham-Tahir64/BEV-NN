"""Timestamp-paced playback of all six nuScenes camera streams."""
from __future__ import annotations

from bisect import bisect_left
from pathlib import Path
import time

CAMERA_ORDER = (
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
)


def build_camera_timeline(nusc, scene_name: str | None = None) -> list[dict]:
    """Index keyframes and sweeps; align cameras to the nearest front timestamp.

    Only metadata is retained in memory. Scenes are ordered by start timestamp.
    """
    scenes = [scene for scene in nusc.scene if scene_name is None or scene["name"] == scene_name]
    if not scenes:
        raise ValueError(f"Scene {scene_name!r} was not found.")
    scenes.sort(key=lambda scene: nusc.get("sample", scene["first_sample_token"])["timestamp"])
    streams = {scene["token"]: {camera: [] for camera in CAMERA_ORDER} for scene in scenes}
    channels = {
        record["token"]: nusc.get("sensor", record["sensor_token"])["channel"]
        for record in nusc.calibrated_sensor
    }
    for record in nusc.sample_data:
        channel = channels[record["calibrated_sensor_token"]]
        if channel not in CAMERA_ORDER:
            continue
        scene_token = nusc.get("sample", record["sample_token"])["scene_token"]
        if scene_token in streams:
            streams[scene_token][channel].append(record)

    timeline = []
    for scene in scenes:
        cameras = streams[scene["token"]]
        for channel, records in cameras.items():
            records.sort(key=lambda record: record["timestamp"])
            if not records:
                raise ValueError(f"Missing {channel} frames in {scene['name']}.")
        timestamps = {
            channel: [record["timestamp"] for record in records]
            for channel, records in cameras.items()
        }
        front = cameras["CAM_FRONT"]
        for index, reference in enumerate(front):
            timestamp = reference["timestamp"]
            tokens = {}
            for channel, records in cameras.items():
                insertion = bisect_left(timestamps[channel], timestamp)
                candidates = {max(0, insertion - 1), min(insertion, len(records) - 1)}
                nearest = min(candidates, key=lambda pos: (abs(timestamps[channel][pos] - timestamp), pos))
                tokens[channel] = records[nearest]["token"]
            timeline.append({
                "scene": scene["name"],
                "timestamp": timestamp,
                "scene_start": front[0]["timestamp"],
                "scene_frame": index + 1,
                "scene_frames": len(front),
                "camera_tokens": tokens,
            })
    return timeline


def read_camera_images(nusc, frame: dict, size: tuple[int, int]):
    """Decode just one six-camera frame, resizing it for the playback window."""
    from PIL import Image

    images = {}
    for channel in CAMERA_ORDER:
        path = Path(nusc.get_sample_data_path(frame["camera_tokens"][channel]))
        with Image.open(path) as source:
            source.draft("RGB", size)
            image = source.convert("RGB")
            image.thumbnail(size, Image.Resampling.BILINEAR)
            images[channel] = image
    return images


class CameraPlayer:
    def __init__(self, root, nusc, frames: list[dict], speed: float = 1.0, loop: bool = False):
        import tkinter as tk
        from tkinter import ttk

        self.root, self.nusc, self.frames = root, nusc, frames
        self.loop, self.index, self.playing, self.timer = loop, 0, False, None
        self.photos = {}
        root.title("nuScenes - six-camera video")
        width = min(480, max(240, (root.winfo_screenwidth() - 100) // 3))
        self.image_size = (width, width * 9 // 16)
        root.minsize(760, 480)
        panel = ttk.Frame(root, padding=8)
        panel.pack(fill="both", expand=True)
        self.labels = {}
        for i, channel in enumerate(CAMERA_ORDER):
            cell = ttk.Frame(panel)
            cell.grid(row=i // 3, column=i % 3, padx=3, pady=3, sticky="nsew")
            ttk.Label(cell, text=channel, anchor="center").pack(fill="x")
            label = ttk.Label(cell, anchor="center")
            label.pack(fill="both", expand=True)
            self.labels[channel] = label
        for i in range(3):
            panel.columnconfigure(i, weight=1)
        for i in range(2):
            panel.rowconfigure(i, weight=1)
        controls = ttk.Frame(root, padding=8)
        controls.pack(fill="x")
        self.play_button = ttk.Button(controls, text="Play", command=self.toggle)
        self.play_button.pack(side="left", padx=4)
        ttk.Button(controls, text="Previous frame", command=lambda: self.step(-1)).pack(side="left", padx=4)
        ttk.Button(controls, text="Next frame", command=lambda: self.step(1)).pack(side="left", padx=4)
        ttk.Button(controls, text="Restart", command=self.restart).pack(side="left", padx=4)
        ttk.Label(controls, text="Speed:").pack(side="left", padx=4)
        self.speed = tk.StringVar(value=f"{speed:g}")
        ttk.Combobox(controls, textvariable=self.speed, values=("0.25", "0.5", "1", "2", "4"),
                     state="readonly", width=6).pack(side="left")
        self.status = tk.StringVar()
        ttk.Label(root, textvariable=self.status, padding=8).pack(fill="x")
        root.bind("<space>", lambda event: self.toggle())
        root.bind("<Left>", lambda event: self.step(-1))
        root.bind("<Right>", lambda event: self.step(1))
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.draw()

    def draw(self):
        from PIL import ImageTk

        frame = self.frames[self.index]
        images = read_camera_images(self.nusc, frame, self.image_size)
        for channel, image in images.items():
            self.photos[channel] = ImageTk.PhotoImage(image, master=self.root)
            self.labels[channel].configure(image=self.photos[channel])
        elapsed = (frame["timestamp"] - frame["scene_start"]) / 1_000_000
        self.status.set(
            f"{frame['scene']} | {elapsed:.2f}s | Scene frame {frame['scene_frame']}/{frame['scene_frames']}"
            f" | Overall {self.index + 1}/{len(self.frames)} | Space: pause/play | Arrows: step"
        )

    def pause(self):
        self.playing = False
        self.play_button.configure(text="Play")
        if self.timer is not None:
            self.root.after_cancel(self.timer)
            self.timer = None

    def toggle(self):
        if self.playing:
            self.pause()
        else:
            if self.index == len(self.frames) - 1:
                self.index = 0
                self.draw()
            self.playing = True
            self.play_button.configure(text="Pause")
            self.schedule()

    def schedule(self, render_seconds: float = 0.0):
        if not self.playing:
            return
        if self.index + 1 < len(self.frames):
            current, following = self.frames[self.index:self.index + 2]
            delta = ((following["timestamp"] - current["timestamp"]) / 1_000_000
                     if following["scene"] == current["scene"] else 1 / 12)
        else:
            delta = 1 / 12
        milliseconds = max(1, round(1000 * (delta / float(self.speed.get()) - render_seconds)))
        self.timer = self.root.after(milliseconds, self.advance)

    def advance(self):
        self.timer = None
        if not self.playing:
            return
        if self.index == len(self.frames) - 1:
            if not self.loop:
                self.pause()
                self.status.set(self.status.get() + " | Playback finished")
                return
            self.index = -1
        self.index += 1
        start = time.perf_counter()
        try:
            self.draw()
        except (OSError, ValueError) as exc:
            self.pause()
            from tkinter import messagebox
            messagebox.showerror("Camera playback error", str(exc), parent=self.root)
            return
        self.schedule(time.perf_counter() - start)

    def step(self, direction: int):
        self.pause()
        self.index = min(len(self.frames) - 1, max(0, self.index + direction))
        self.draw()

    def restart(self):
        self.pause()
        self.index = 0
        self.draw()
        self.toggle()

    def close(self):
        self.pause()
        self.root.destroy()


def stream_camera_frames(nusc, scene_name=None, speed=1.0, loop=False):
    import math
    import tkinter as tk

    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("Playback speed must be a positive finite number.")
    frames = build_camera_timeline(nusc, scene_name)
    print(f"Streaming {len(frames)} camera frames across {len({f['scene'] for f in frames})} scenes.")
    print("Space: pause/play. Left/right arrows: step. Close the window to stop.")
    root = tk.Tk()
    try:
        player = CameraPlayer(root, nusc, frames, speed, loop)
        player.toggle()
        root.mainloop()
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass  # The window may already have been closed by the user.
