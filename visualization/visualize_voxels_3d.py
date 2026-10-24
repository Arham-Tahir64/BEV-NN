"""Render saved occupancy predictions as BEV and actual 0.4 m voxel cubes.

No inference or retraining. Writes standalone interactive HTML plus PNG.
Run with the project's Python; install Plotly into visualization/_deps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

BOUNDS = np.array([-40., -40., -1., 40., 40., 5.4])
SHAPE = (200, 200, 16)
NAMES = ["Others", "Barrier", "Bicycle", "Bus", "Car", "Construction vehicle",
         "Motorcycle", "Pedestrian", "Traffic cone", "Trailer", "Truck", "Road",
         "Other flat", "Sidewalk", "Terrain", "Manmade", "Vegetation", "Free", "Unsupervised"]
PALETTE = np.array([
    [180,180,180], [255,220,75], [255,158,30], [255,158,30], [255,158,30],
    [255,158,30], [255,158,30], [255,75,100], [255,220,75], [255,158,30],
    [255,158,30], [70,215,240], [170,150,105], [245,151,200], [150,195,110],
    [185,115,240], [120,225,75], [49,61,74], [18,23,30],
], dtype=np.uint8)

# Plotly substitutes {plot_id} when embedding this callback in an HTML export.
CAMERA_SYNC_JS = r"""
(function () {
    const plot = document.getElementById('{plot_id}');
    const controls = document.createElement('label');
    controls.style.cssText = 'display:block;text-align:center;color:#e2e8f0;font:15px system-ui;margin:12px;cursor:pointer';
    const toggle = document.createElement('input');
    toggle.type = 'checkbox';
    toggle.checked = true;
    toggle.style.marginRight = '8px';
    controls.append(toggle, document.createTextNode('Sync 3D views after movement stops (rotation, zoom and pan)'));
    plot.parentNode.insertBefore(controls, plot);

    let lastSource = 'scene';
    let updating = false;
    let pending = null;
    let timer = null;
    const clone = value => JSON.parse(JSON.stringify(value));
    const quietMilliseconds = 250;

    function readCamera(source) {
        // The wheel event's camera can lag behind the actual WebGL camera.
        // Read the final rendered camera only after the gesture has settled.
        const scene = plot._fullLayout && plot._fullLayout[source];
        const renderer = scene && scene._scene;
        return clone(renderer && typeof renderer.getCamera === 'function'
            ? renderer.getCamera() : plot.layout[source].camera);
    }

    function schedule() {
        if (timer !== null) clearTimeout(timer);
        timer = null;
        if (updating || !pending || !toggle.checked) return;
        timer = setTimeout(function () {
            timer = null;
            if (!toggle.checked || updating || !pending) return;
            const source = pending;
            pending = null;
            const camera = readCamera(source);
            updating = true;
            // Commit both cameras so a redraw cannot restore stale source state.
            const change = {'scene.camera': clone(camera),
                            'scene2.camera': clone(camera)};
            Promise.resolve(Plotly.relayout(plot, change)).finally(function () {
                updating = false;
                if (pending) schedule();
            });
        }, quietMilliseconds);
    }

    function queue(source) {
        lastSource = source;
        if (!toggle.checked) return;
        pending = source;
        schedule();
    }

    function changedScene(event) {
        return ['scene', 'scene2'].find(source => {
            const key = source + '.camera';
            return Object.keys(event).some(k => k === key || k.startsWith(key + '.'));
        });
    }

    plot.on('plotly_relayout', function (event) {
        if (updating) return; // Ignore feedback from the update to both views.
        const source = changedScene(event);
        if (source) queue(source);
    });
    plot.on('plotly_relayouting', function (event) {
        if (updating || !changedScene(event)) return;
        // A new drag must cancel an older pending sync; wait for its release.
        if (timer !== null) clearTimeout(timer);
        timer = null;
        pending = null;
    });
    for (const source of ['scene', 'scene2']) {
        const scene = plot._fullLayout && plot._fullLayout[source];
        const canvas = scene && scene._scene && scene._scene.glplot.canvas;
        if (canvas) {
            // Capture every wheel tick, including ticks that emit an empty
            // relayout event. Never prevent or intercept Plotly's own zoom.
            canvas.addEventListener('wheel', function () { queue(source); },
                                    {capture: true, passive: true});
        }
    }
    toggle.addEventListener('change', function () {
        if (timer !== null) clearTimeout(timer);
        timer = null;
        pending = null;
        if (toggle.checked) queue(lastSource);
    });
})();
"""


def top_down(labels, valid):
    occupied = valid & (labels < 17)
    z = np.arange(labels.shape[2])[None, None, :]
    highest = np.max(np.where(occupied, z, -1), axis=2)
    selected = np.take_along_axis(labels, np.maximum(highest, 0)[..., None], axis=2)[..., 0]
    return np.where(highest >= 0, selected, np.where(valid.any(2), 17, 18))


def exposed_faces(labels, valid, bounds=BOUNDS):
    """Exact outside faces of occupied cubes. Never sample/downsample voxels.

    Hidden internal faces are culled for rendering; all occupied voxels still
    define the mesh. Unknown/invisible voxels and free space are omitted.
    Returns quadrilateral faces in metres and semantic IDs for their owners.
    """
    occupied = valid & (labels < 17)
    padded = np.pad(occupied, 1)
    vertices, classes = [], []
    size = (bounds[3:] - bounds[:3]) / np.asarray(labels.shape)
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        for direction in (-1, 1):
            slices = [slice(1,-1)] * 3
            slices[axis] = slice(0,-2) if direction == -1 else slice(2,None)
            keep = occupied & ~padded[tuple(slices)]
            indices = np.argwhere(keep)
            if not len(indices):
                continue
            corners = np.zeros((4,3), dtype=np.float32)
            corners[:,axis] = int(direction == 1)
            corners[:,others[0]] = [0,1,1,0]
            corners[:,others[1]] = [0,0,1,1]
            normal = np.cross(corners[1]-corners[0], corners[2]-corners[0])
            if normal[axis] * direction < 0:
                corners = corners[::-1]
            face = bounds[:3] + (indices[:,None,:] + corners[None,:,:]) * size
            vertices.append(face.astype(np.float32))
            classes.append(labels[keep])
    if not vertices:
        return np.empty((0,4,3), np.float32), np.empty(0, np.uint8)
    return np.concatenate(vertices), np.concatenate(classes)


def discrete_scale():
    scale = []
    for index, colour in enumerate(PALETTE):
        rgb = f"rgb({colour[0]},{colour[1]},{colour[2]})"
        scale += [(index/len(PALETTE),rgb), ((index+1)/len(PALETTE),rgb)]
    return scale


def save_html(target, prediction, valid, meshes, title, output, height_scale):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=2, specs=[[{"type":"xy"}]*2, [{"type":"scene"}]*2],
                        row_heights=[.43,.57], vertical_spacing=.08,
                        subplot_titles=["Ground truth — BEV", "Prediction — BEV",
                                        "Ground truth — 3D voxels", "Prediction — 3D voxels"])
    centres_x = -40 + (np.arange(200)+.5)*.4
    centres_y = -40 + (np.arange(200)+.5)*.4
    colours = discrete_scale()
    for col, (labels, (faces, classes)) in enumerate(zip((target,prediction), meshes), start=1):
        bev = top_down(labels, valid)
        fig.add_trace(go.Heatmap(x=centres_y,y=centres_x,z=bev,zmin=-.5,zmax=18.5,
                                colorscale=colours,showscale=False,
                                hovertemplate="Y=%{x:.1f} m<br>X=%{y:.1f} m<br>Class ID=%{z}<extra></extra>"),1,col)
        points = faces.reshape(-1,3)
        base = np.arange(len(faces), dtype=np.int32)*4
        fig.add_trace(go.Mesh3d(
            x=points[:,0],y=points[:,1],z=points[:,2],
            i=np.concatenate([base,base]),j=np.concatenate([base+1,base+2]),
            k=np.concatenate([base+2,base+3]),
            intensity=np.concatenate([classes,classes]), intensitymode="cell",
            colorscale=colours,cmin=-.5,cmax=18.5,showscale=False,
            flatshading=True,lighting=dict(ambient=.7,diffuse=.6,roughness=.8),
            hovertemplate="X=%{x:.1f} m<br>Y=%{y:.1f} m<br>Z=%{z:.1f} m<extra></extra>",
            name="Occupied voxel faces",showlegend=False,
        ),2,col)
    # Exact XYZ coordinates; aspect ratio alone changes with the height buttons.
    z_ratio = (BOUNDS[5]-BOUNDS[2]) / 80
    scene = dict(xaxis=dict(title="X forward (m)",range=[-40,40]),
                 yaxis=dict(title="Y left (m)",range=[-40,40]),
                 zaxis=dict(title="Z up (m)",range=[-1,5.4],dtick=1),
                 aspectmode="manual",aspectratio=dict(x=1,y=1,z=z_ratio*height_scale),
                 camera=dict(eye=dict(x=1.3,y=-1.7,z=1.2)),dragmode="orbit")
    fig.update_layout(scene=scene,scene2=scene)
    for axis in ("xaxis", "xaxis2"):
        fig.layout[axis].update(title="Y left (m)",range=[-40,40],constrain="domain")
    for axis, anchor in (("yaxis","x"),("yaxis2","x2")):
        fig.layout[axis].update(title="X forward (m)",range=[-40,40],scaleanchor=anchor,scaleratio=1)
    for class_id in (11,4,15,16,13,7,1,14,12,0):
        rgb = PALETTE[class_id]
        fig.add_trace(go.Scatter3d(x=[None],y=[None],z=[None],mode="markers",
                                  marker=dict(size=6,color=f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"),
                                  name=NAMES[class_id],showlegend=True),2,1)
    buttons = [dict(label=f"Height ×{scale}",method="relayout",
                    args=[{"scene.aspectratio.z": z_ratio*scale,
                           "scene2.aspectratio.z": z_ratio*scale}]) for scale in (1,3,6)]
    fig.update_layout(
        title=dict(text=title+"<br><sup>Drag 3D to rotate • scroll to zoom • 0.4 m cubes • free/unknown omitted in 3D</sup>",x=.5),
        height=1050,template="plotly_dark",margin=dict(t=110,b=115,l=55,r=30),
        legend=dict(orientation="h",y=-.07,x=.5,xanchor="center"),
        updatemenus=[dict(type="buttons",direction="right",buttons=buttons,
                         active=(1 if height_scale == 3 else 0),x=.5,xanchor="center",y=1.06)],
    )
    explanation = ("<p style='text-align:center;font:15px system-ui;color:#cbd5e1'>"
                   f"Initial height display ×{height_scale}; select Height ×1 for true proportions. "
                   "Z coordinates and tick labels always remain in metres. "
                   "Only camera-visible, non-ignore voxels are compared. "
                   "BEV shows the highest occupied voxel in each XY column; "
                   "3D retains all 16 height layers. Gray BEV = free; black = unknown.</p>")
    html = fig.to_html(full_html=True,include_plotlyjs=True,
                       config={"responsive":True,"scrollZoom":True},post_script=CAMERA_SYNC_JS)
    html = html.replace("<body>","<body style='background:#111827'>"+explanation)
    output.write_text(html,encoding="utf-8")


def save_png(target, prediction, valid, meshes, title, output, height_scale):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(14,12),facecolor="#111827",layout="constrained")
    for col,(labels,(faces,classes)) in enumerate(zip((target,prediction),meshes)):
        ax = fig.add_subplot(2,2,col+1,facecolor="#12171e")
        ax.imshow(top_down(labels,valid),origin="lower",extent=(-40,40,-40,40),
                  cmap=ListedColormap(PALETTE/255),vmin=-.5,vmax=18.5,interpolation="nearest")
        ax.set_title(("Ground truth","Prediction")[col]+" — BEV",color="white")
        ax.set_xlabel("Y left (m)",color="white")
        ax.set_ylabel("X forward (m)",color="white")
        ax.tick_params(colors="#cbd5e1")
        ax.plot(0,0,"w+",markersize=5)
        ax3 = fig.add_subplot(2,2,col+3,projection="3d",facecolor="#111827")
        ax3.add_collection3d(Poly3DCollection(faces,facecolors=PALETTE[classes]/255,
                                             edgecolors=(0,0,0,.09),linewidths=.08,
                                             zsort="average",rasterized=True))
        ax3.set(xlim=(-40,40),ylim=(-40,40),zlim=(-1,5.4),xlabel="X forward (m)",
                ylabel="Y left (m)",zlabel="Z up (m)")
        ax3.set_box_aspect((80,80,6.4*height_scale))
        ax3.view_init(elev=25,azim=-55)
        ax3.set_title(("Ground truth","Prediction")[col]+f" — 3D (height display ×{height_scale})",color="white")
        ax3.tick_params(colors="#cbd5e1",labelsize=8)
        for axis in (ax3.xaxis,ax3.yaxis,ax3.zaxis):
            axis.label.set_color("white")
            axis.set_pane_color((.10,.14,.19,1))
    fig.suptitle(title+"\n0.4 m voxels; 16 height layers; Z = −1 to 5.4 m",
                 color="white",fontsize=16)
    fig.legend(handles=[Patch(color=PALETTE[i]/255,label=NAMES[i]) for i in (11,4,15,16,13,7,1,14)],
               loc="outside lower center",ncol=8,labelcolor="white",frameon=False)
    fig.savefig(output,dpi=150,facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parent
    parser.add_argument("--evaluation-dir",type=Path,default=root/"outputs/train_xyz_full_evaluation")
    parser.add_argument("--checkpoint",choices=["best","latest"],default="latest")
    parser.add_argument("--height-scale",type=int,choices=[1,3,6],default=3)
    parser.add_argument("--plotly-deps",type=Path,default=root/"_deps")
    parser.add_argument("--html-only",action="store_true",help="Refresh interactive viewers without rendering PNGs.")
    args=parser.parse_args()
    if args.plotly_deps.is_dir():
        sys.path.insert(0,str(args.plotly_deps.resolve()))
    paths=sorted((args.evaluation_dir/args.checkpoint).glob("*/*/predictions.npz"))
    if not paths:
        raise FileNotFoundError("No saved predictions found; run evaluate_run.py first.")
    for path in paths:
        with np.load(path,allow_pickle=False) as archive:
            prediction,target,valid = archive["prediction"],archive["semantics"],archive["valid_mask"].astype(bool)
        if prediction.shape != SHAPE or target.shape != SHAPE or valid.shape != SHAPE:
            raise ValueError("Expected Occ3D [200,200,16] arrays.")
        valid &= target != 255
        meshes=[exposed_faces(labels,valid) for labels in (target,prediction)]
        metadata=json.loads((path.parent/"metrics.json").read_text())
        scene=metadata["scene_name"]
        epoch=metadata.get("epoch",100 if args.checkpoint=="latest" else 11)
        title=f"{scene} · {path.parent.parent.name} frame · epoch {epoch}"
        save_html(target,prediction,valid,meshes,title,path.parent/"bev_and_3d.html",args.height_scale)
        if not args.html_only:
            save_png(target,prediction,valid,meshes,title,path.parent/"bev_and_3d.png",args.height_scale)
        print(f"Saved BEV + 3D: {path.parent}; faces GT/pred: {[len(mesh[0]) for mesh in meshes]}",flush=True)


if __name__=="__main__":
    main()
