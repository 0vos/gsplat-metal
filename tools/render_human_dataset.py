# SPDX-FileCopyrightText: Copyright (c) 2026 the Regents of the University of California, Nerfstudio Team and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Render a multi-view dataset of a humanoid FBX model with Blender (headless).

Produces a NeRF-synthetic/Blender-format dataset that ``examples/train_mps.py``
can consume directly::

    data/human/
      transforms_train.json
      transforms_test.json
      train/r_0000.png  ...   (RGBA, transparent background)
      test/r_0000.png   ...

Usage::

    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
        --python tools/render_human_dataset.py -- \
        --fbx ~/Documents/Models/宫野志保/base_basic_pbr.fbx \
        --out data/human --res 400 --views 48 --test-views 12 --samples 48
"""

import argparse
import json
import math
import os
import sys
import time

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fbx",
        default=os.path.expanduser("~/Documents/Models/宫野志保/base_basic_pbr.fbx"),
    )
    ap.add_argument("--out", default="data/human")
    ap.add_argument("--res", type=int, default=400)
    ap.add_argument("--views", type=int, default=48)
    ap.add_argument("--test-views", type=int, default=12)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--lens", type=float, default=50.0)
    ap.add_argument("--sensor", type=float, default=36.0)
    return ap.parse_args(argv)


def scene_bbox():
    mins = Vector((1e18, 1e18, 1e18))
    maxs = Vector((-1e18, -1e18, -1e18))
    dg = bpy.context.evaluated_depsgraph_get()
    for o in bpy.context.scene.objects:
        if o.type != "MESH" or o.hide_render:
            continue
        ob = o.evaluated_get(dg)
        for c in ob.bound_box:
            w = ob.matrix_world @ Vector(c)
            mins = Vector((min(mins.x, w.x), min(mins.y, w.y), min(mins.z, w.z)))
            maxs = Vector((max(maxs.x, w.x), max(maxs.y, w.y), max(maxs.z, w.z)))
    return mins, maxs


def main():
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)

    t0 = time.time()
    bpy.ops.import_scene.fbx(filepath=args.fbx)
    print(f"[human] imported {args.fbx} in {time.time() - t0:.1f}s")

    mins, maxs = scene_bbox()
    center = (mins + maxs) * 0.5
    height = maxs.z - mins.z
    radius = max(maxs.x - mins.x, maxs.y - mins.y)
    print(f"[human] bbox {tuple(round(x, 3) for x in mins)} .. "
          f"{tuple(round(x, 3) for x in maxs)} height={height:.3f}")

    # --- World: white, transparent film (alpha = mask) ---
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 1.0

    # --- Lights: key / fill / rim ---
    def add_light(name, kind, energy, loc, angle=None):
        data = bpy.data.lights.new(name, type=kind)
        data.energy = energy
        if kind == "AREA":
            data.size = 4.0
        if kind == "SUN" and angle is not None:
            data.angle = angle
        obj = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(obj)
        obj.location = loc
        direction = center - obj.location
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        return obj

    d = height
    add_light("key", "AREA", 900.0, center + Vector((1.5 * d, -2.0 * d, 1.2 * d)))
    add_light("fill", "AREA", 350.0, center + Vector((-2.0 * d, -1.2 * d, 0.4 * d)))
    add_light("rim", "AREA", 500.0, center + Vector((0.2 * d, 2.2 * d, 1.0 * d)))

    # --- Camera ---
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = args.lens
    cam_data.sensor_width = args.sensor
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = args.samples
    sc.cycles.device = "CPU"
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 6
    sc.render.resolution_x = args.res
    sc.render.resolution_y = args.res
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.image_settings.color_depth = "8"
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"

    angle_x = 2.0 * math.atan((args.sensor / 2.0) / args.lens)

    out = os.path.abspath(args.out)
    for split in ("train", "test"):
        os.makedirs(os.path.join(out, split), exist_ok=True)

    def render_view(rel_path, theta, elev_frac, dist_frac):
        dist = dist_frac * height
        eye = Vector(
            (
                center.x + dist * math.sin(theta),
                center.y - dist * math.cos(theta),
                center.z + elev_frac * height,
            )
        )
        target = Vector((center.x, center.y, center.z + 0.05 * height))
        cam.location = eye
        cam.rotation_euler = (target - eye).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        path = os.path.join(out, rel_path + ".png")
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        m = cam.matrix_world.copy()
        return {
            "file_path": "./" + rel_path,
            "rotation": 0.0,
            "time": 0.0,
            "transform_matrix": [[m[i][j] for j in range(4)] for i in range(4)],
        }

    train_frames = []
    n = args.views
    for i in range(n):
        theta = 2.0 * math.pi * i / n + 0.07 * math.sin(3.0 * i)
        # alternate two elevation bands so the model is seen from below/above
        elev = 0.10 if i % 2 == 0 else 0.35
        dist = 2.10 if i % 2 == 0 else 2.00
        train_frames.append(
            render_view(f"train/r_{i:04d}", theta, elev, dist)
        )
        print(f"[human] train {i + 1}/{n} theta={theta:.2f} elev={elev} "
              f"({time.time() - t0:.0f}s)")

    test_frames = []
    for i in range(args.test_views):
        theta = 2.0 * math.pi * i / args.test_views + 0.5 * math.pi / args.test_views
        elev = 0.22 + 0.05 * math.sin(2.0 * i)
        test_frames.append(
            render_view(f"test/r_{i:04d}", theta, elev, 2.05)
        )
        print(f"[human] test {i + 1}/{args.test_views} ({time.time() - t0:.0f}s)")

    meta = lambda frames: {"camera_angle_x": angle_x, "frames": frames}
    with open(os.path.join(out, "transforms_train.json"), "w") as f:
        json.dump(meta(train_frames), f, indent=2)
    with open(os.path.join(out, "transforms_test.json"), "w") as f:
        json.dump(meta(test_frames), f, indent=2)
    print(f"[human] done: {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
