#!/usr/bin/env python3
"""Visualize MuJoCo models in interactive viewer."""

import argparse
import sys
from pathlib import Path

import mujoco
from mujoco import viewer


MODELS = {
    "reach": "scene_reach.xml",
    "pick": "pick_and_place_scene.xml",
    "vs050": "vs050.xml",
    "2f85": "vs050_2f85.xml",
}


def main():
    parser = argparse.ArgumentParser(description="Visualize VS050 MuJoCo models")
    parser.add_argument(
        "model",
        choices=list(MODELS.keys()),
        nargs="?",
        default="reach",
        help="Model to load (default: reach)",
    )
    args = parser.parse_args()

    model_dir = Path(__file__).parent
    model_path = model_dir / MODELS[args.model]

    if not model_path.exists():
        print(f"Error: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading: {model_path}")
    print("Controls: [drag] rotate, [scroll] zoom, [shift+drag] pan")
    print("Press [ESC] to exit")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    viewer.launch(model=model, data=data)


if __name__ == "__main__":
    main()