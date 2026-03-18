"""
Example script: VS050 Pick-and-Place with a Random Agent
"""
import gymnasium as gym
import vs050_mujoco  # noqa: F401
import time

def main():
    # Create the environment with human rendering mode
    print("Initializing environment...")
    env = gym.make("VS050-PickAndPlace-v0", render_mode="human")
    
    obs, info = env.reset()
    print("Environment reset. Starting random agent...")
    
    try:
        for _ in range(1000):
            # Sample a random action from the environment's action space
            action = env.action_space.sample()
            
            # Step the simulation
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Visualize (handled by the environment in 'human' mode)
            # We add a tiny sleep to make it more viewable at human speed if needed, 
            # though MuJoCo's passive viewer usually handles sync well.
            time.sleep(0.01)
            
            if terminated or truncated:
                print("Episode finished. Resetting...")
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("\nShutting down example...")
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()
