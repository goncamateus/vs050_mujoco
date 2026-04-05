"""
Generate a GIF of a random ReachPose episode for documentation.
"""
import gymnasium as gym
import vs050_mujoco  # noqa: F401
import imageio.v3 as imageio

ENV = "VS050-ReachPose-v0"
OUTPUT = "assets/reach_pose.gif"
MAX_FRAMES = 120  # ~2.4s at 50fps

def main():
    env = gym.make(ENV, render_mode="rgb_array")
    env.reset()

    frames = []
    done = False
    frame_count = 0

    while not done and frame_count < MAX_FRAMES:
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        frame = env.render()
        if frame is not None:
            frames.append(frame)
        frame_count += 1

    env.close()
    imageio.imwrite(OUTPUT, frames, duration=0.02, loop=0)
    print(f"Saved {len(frames)} frames to {OUTPUT}")


if __name__ == "__main__":
    main()
