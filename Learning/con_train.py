import torch.nn
from stable_baselines3 import PPO
from Env import RunClass
import time
import mujoco.viewer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from typing import Callable

def linear_schedule(initial: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return 5e-5+progress_remaining*(initial - 5e-5)
    return func

checkpoint_callback = CheckpointCallback(
    save_freq= 7500,
    save_path="./checkpoints/",
    name_prefix="walker_",
    save_replay_buffer=True,
    save_vecnormalize=True
)

env = make_vec_env(RunClass, n_envs=8, seed=42,env_kwargs={"robot_path": "robot.xml","total_step":10**6,"ep_c":400})
model = PPO.load("walker_v9_con", env=env, device="cpu",
                 custom_objects={"learning_rate": linear_schedule(1e-4)})
model.learn(total_timesteps=3000000,callback=checkpoint_callback)
model.save("walker_v9_con")
eval_env = RunClass(seed=41,robot_path ="robot.xml")
while True:
    with mujoco.viewer.launch_passive(eval_env.model, eval_env.data) as viewer:
        obs,info = eval_env.reset()
        while True:
            action = model.predict(obs,deterministic=True)[0]
            obs, reward, terminated, truncated, info = eval_env.step(action)
            time.sleep(0.020)
            viewer.sync()
            if terminated or truncated:
                break