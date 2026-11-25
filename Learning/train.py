import torch.nn
from stable_baselines3 import PPO
from Env import RunClass
import time
import mujoco.viewer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import SubprocVecEnv
from typing import Callable
import numpy as np

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

policy_kwargs = dict(
    n_steps=1024,            # x 8 Envs = 8192 Samples pro Update
    batch_size=8192,
    n_epochs=5,
    learning_rate=linear_schedule(3e-4),      # optional: lineare Schedule auf 1e-5
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,            # 0.001-0.005 falls Policy zu früh kollabiert
    vf_coef=0.5,
    max_grad_norm=1.0,
    target_kl=0.02,          # bricht Epochen ab, ersetzt adaptives LR
    policy_kwargs=dict(
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
        activation_fn=torch.nn.ELU,
        log_std_init=-0.5,   # std ≈ 0.61 in Action-Skala
        ortho_init=False,
    ),
)
if __name__ == "__main__":
    env = make_vec_env(RunClass, n_envs=8, seed=42,env_kwargs={"robot_path": "robot.xml"},vec_env_cls=SubprocVecEnv)
    ml = PPO("MlpPolicy",env,verbose=1,device="cpu",**policy_kwargs,seed=42)
    ml.learn(total_timesteps=10000000,callback=checkpoint_callback)
    ml.policy.log_std.data.fill_(np.log(0.15))
    ml.lr_schedule = linear_schedule(1e-4)
    ml.policy.log_std.requires_grad_(False)
    ml.learn(total_timesteps=3000000, callback=checkpoint_callback)
    ml.save("walker_v12")
    eval_env = RunClass(seed=41,robot_path ="robot.xml",ep_c=500,total_step=5_000_000)
    while True:
        with mujoco.viewer.launch_passive(eval_env.model, eval_env.data) as viewer:
            obs,info = eval_env.reset()
            while True:
                action = ml.predict(obs,deterministic=True)[0]
                obs, reward, terminated, truncated, info = eval_env.step(action)
                time.sleep(0.020)
                viewer.sync()
                if terminated or truncated:
                    break