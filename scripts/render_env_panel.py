"""Render a three-task PyBullet panel for the poster. Spins up block_pulling,
block_picking, and drawer_opening in DIRECT mode, grabs an RGB per task at a
hand-tuned camera, and composes them into a 1x3 panel. The PyBullet sky
gradient is swapped for the poster's cream so the panel sits flush.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pybullet as pb

from so2_equi_rl.envs.wrapper import EnvWrapper

# Left-to-right poster order, with display labels.
TASKS = [
    ("close_loop_block_pulling", "Block Pulling"),
    ("close_loop_block_picking", "Block Picking"),
    ("close_loop_drawer_opening", "Drawer Opening"),
]

# Eyes sit front-right of the workspace and look slightly down. FOV is
# narrow so the workspace fills the frame.
CAMS = {
    "close_loop_block_pulling": dict(
        eye=[0.95, -0.55, 0.55],
        target=[0.45, 0.00, 0.06],
        fov=34,
    ),
    "close_loop_block_picking": dict(
        eye=[0.95, -0.55, 0.55],
        target=[0.45, 0.00, 0.06],
        fov=34,
    ),
    "close_loop_drawer_opening": dict(
        eye=[1.00, -0.55, 0.58],
        target=[0.52, 0.00, 0.12],
        fov=38,
    ),
}

CAM_UP = [0, 0, 1]
IMG_W, IMG_H = 900, 720
CROP = dict(top=20, bottom=20, left=180, right=60)

# Poster block-body color, replaces the PyBullet sky gradient.
BG_RGB = np.array([245, 245, 240], dtype=np.uint8)

# Poster palette. Objects get recolored so white-on-white isn't invisible against the cream backdrop.
RED = [0.906, 0.435, 0.318, 1.0]  # Coral   #E76F51
YELLOW = [0.914, 0.765, 0.416, 1.0]  # Sand    #E9C46A
TEAL = [0.165, 0.616, 0.561, 1.0]  # Teal    #2A9D8F
WOOD = [0.72, 0.55, 0.38, 1.0]  # Warm wood, drawer body


def _color_body(body_id: int, color) -> None:
    # -1 alone misses articulated parts (drawer slides, robot fingers).
    pb.changeVisualShape(body_id, -1, rgbaColor=color)
    for link in range(pb.getNumJoints(body_id)):
        pb.changeVisualShape(body_id, link, rgbaColor=color)


def setup_scene(env_name: str, env) -> None:
    # Override post-reset poses for reproducibility, recolor for contrast.
    if env_name == "close_loop_block_pulling":
        # Two cubes at fixed offsets near workspace center.
        poses = [
            ([0.40, -0.05, 0.025], RED),
            ([0.50, 0.05, 0.025], YELLOW),
        ]
        for obj, (pos, color) in zip(env.objects, poses):
            pb.resetBasePositionAndOrientation(
                obj.object_id, pos, pb.getQuaternionFromEuler([0, 0, 0])
            )
            _color_body(obj.object_id, color)

    elif env_name == "close_loop_block_picking":
        # Place near the (+x, -y) corner so the cube is close to the camera and reads large.
        pos = [0.35, -0.12, 0.025]
        obj = env.objects[0]
        pb.resetBasePositionAndOrientation(
            obj.object_id, pos, pb.getQuaternionFromEuler([0, 0, 0])
        )
        _color_body(obj.object_id, RED)

    elif env_name == "close_loop_drawer_opening":
        # Rotate the drawer so its handle face points at the camera, and pull it toward the camera.
        drawer_id = env.drawer.id
        drawer_pos = [0.48, 0.03, 0.0]
        drawer_rot = pb.getQuaternionFromEuler([0, 0, np.pi / 2])
        pb.resetBasePositionAndOrientation(drawer_id, drawer_pos, drawer_rot)
        _color_body(drawer_id, WOOD)
        # Handle is a separate multi-body, not a link of the drawer URDF.
        _color_body(env.drawer.handle.id, TEAL)


def grab_rgb(env: EnvWrapper, cam: dict) -> np.ndarray:
    view = pb.computeViewMatrix(cam["eye"], cam["target"], CAM_UP)
    proj = pb.computeProjectionMatrixFOV(
        fov=cam["fov"], aspect=IMG_W / IMG_H, nearVal=0.05, farVal=3.0
    )

    # Prefer hardware OpenGL for shading, fall back to the tiny renderer if no GL context exists.
    try:
        _w, _h, rgba, _depth, _seg = pb.getCameraImage(
            width=IMG_W,
            height=IMG_H,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=pb.ER_BULLET_HARDWARE_OPENGL,
            shadow=1,
            lightDirection=[0.4, -0.6, 1.0],
        )
    except pb.error:
        _w, _h, rgba, _depth, _seg = pb.getCameraImage(
            width=IMG_W,
            height=IMG_H,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=pb.ER_TINY_RENDERER,
        )
    rgba = np.asarray(rgba, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)
    return rgba[..., :3]


def replace_sky(rgb: np.ndarray) -> np.ndarray:
    # PyBullet's sky is a dusty-blue gradient (b > r, b > g). Table,
    # objects, and gripper never sit in that color regime, so a loose
    # blue-channel threshold is enough to mask it out.
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    is_sky = (b > r + 15) & (b > g + 10) & (b > 110)
    out = rgb.copy()
    out[is_sky] = BG_RGB
    return out


def crop_image(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    return rgb[CROP["top"] : h - CROP["bottom"], CROP["left"] : w - CROP["right"]]


def render_one(env_name: str) -> np.ndarray:
    env = EnvWrapper(env_name=env_name, num_processes=0, seed=0, render=False)
    try:
        env.reset()
        # A few resets let physics settle so objects aren't mid-fall.
        for _ in range(3):
            env.reset()
        # Reach through the runner to override poses and colors.
        base_env = env._runner.env
        setup_scene(env_name, base_env)
        # Let physics settle after manual repositioning.
        for _ in range(20):
            pb.stepSimulation()
        rgb = grab_rgb(env, CAMS[env_name])
    finally:
        env.close()
    return crop_image(replace_sky(rgb))


def main() -> None:
    out_dir = Path("report/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for env_name, label in TASKS:
        print(f"  rendering {env_name} ...")
        img = render_one(env_name)
        images.append((img, label))

    # 1x3 panel, generous height, tight margins, no axes. Cream figure
    # background matches the replaced-sky color.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), facecolor="#F5F5F0")
    for ax, (img, label) in zip(axes, images):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#F5F5F0")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(label, fontsize=15, pad=6)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.92, bottom=0.01, wspace=0.02)

    png = out_dir / "env_panel.png"
    pdf = out_dir / "env_panel.pdf"
    fig.savefig(png, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(pdf, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  wrote {png}")
    print(f"  wrote {pdf}")


if __name__ == "__main__":
    main()
