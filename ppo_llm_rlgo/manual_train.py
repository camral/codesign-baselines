"""Interactive PPO plus human-supplied embodiment templates on GPU 0."""

import math
import os
from pathlib import Path
import sys

import numpy as np


def render_boundary(env, base, templates, output, round_id, seed):
    """Render exact prior templates at a fresh robot pose before the next prompt."""
    from PIL import Image, ImageDraw

    designs = np.stack(templates) if templates else np.asarray(base)[None]
    if env.num_envs < len(designs):
        raise ValueError("manual evaluation-worlds must fit all template renders")
    tiled = designs[np.arange(env.num_envs) % len(designs)]
    _, rejected, _ = env.reset(seed=seed, embodiment=tiled)
    rejected = np.asarray(rejected.cpu() if hasattr(rejected, "cpu") else rejected).reshape(-1)
    directory = Path(output) / "manual_renders" / f"round_{round_id:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    images, paths = [], []
    for index in range(len(designs)):
        frame = env.render(env_idx=index)
        if frame is None:
            raise RuntimeError(f"no robot render for template {index}")
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8).copy()).convert("RGB")
        path = directory / ("base.png" if not templates else f"template_{index:02d}.png")
        image.save(path)
        images.append(image)
        paths.append(str(path.resolve()))
    width, height = images[0].size
    columns = min(4, len(images))
    rows = math.ceil(len(images)/columns)
    sheet = Image.new("RGB", (columns*width, rows*(height+28)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, image in enumerate(images):
        x, y = (index % columns)*width, (index // columns)*(height+28)
        sheet.paste(image, (x, y+28))
        label = "base" if not templates else f"template {index}"
        draw.text((x+6, y+7), label+(" REJECTED" if rejected[index] else ""), fill="black")
    contact_sheet = directory / "contact_sheet.png"
    sheet.save(contact_sheet)
    return str(contact_sheet.resolve()), paths


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    if any(flag in sys.argv[1:] for flag in ("--device", "--llm-gateway", "--llm-templates")):
        raise ValueError("manual_train fixes GPU 0, the manual gateway and 16 templates")
    from ppo_ngopt_rlgo.gym_train import main as shared_main
    sys.argv = [sys.argv[0], "--device", "cuda:0", "--num-envs", "16384", "--evaluation-worlds", "128", "--llm-templates", "16", "--llm-gateway", "manual", "--llm-model", "human", *sys.argv[1:]]
    shared_main(method="ppo_llm")


if __name__ == "__main__":
    main()
