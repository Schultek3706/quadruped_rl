import torch.nn
from OpenGL.raw.GL.SGIX import flush_raster
from stable_baselines3 import PPO
from Env import RunClass
import mujoco
import time
import mujoco.viewer
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.envs.registration import register
policy_kwargs = dict(
    n_steps=1024,            # x 32 Envs = 32768 Samples pro Update
    batch_size=8192,
    n_epochs=5,
    learning_rate=3e-4,      # optional: lineare Schedule auf 1e-5
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
        log_std_init=-0.5,   # std ≈ 0.37 in Action-Skala
        ortho_init=False,
    ),
)
env = make_vec_env(RunClass, n_envs=8, seed=42,env_kwargs={"robot_path": "robot.xml"})
model = PPO("MlpPolicy",env,verbose=1,device="cpu",**policy_kwargs)
model.learn(total_timesteps=1000000)
eval_env = RunClass(seed=42,robot_path ="robot.xml")
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