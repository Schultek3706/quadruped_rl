from stable_baselines3 import PPO
from Env import RunClass
import time
import mujoco.viewer as view
import mujoco

model = PPO.load("models/walker_v9", device="cpu")
eval_env = RunClass(seed=41,robot_path ="robot.xml",ep_c=1000,total_step=10**6)

base_id = mujoco.mj_name2id(eval_env.model, mujoco.mjtObj.mjOBJ_BODY, "base")
"""Using qvel with index 0 at the watch box in the viewer shows the foreward velocity
For a fixed cam you have to comment out the viewer.cam lines"""
while True:
    with view.launch_passive(eval_env.model, eval_env.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = base_id
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -30
        obs,info = eval_env.reset()
        while True:
            action = model.predict(obs,deterministic=True)[0]
            obs, reward, terminated, truncated, info = eval_env.step(action)
            time.sleep(0.020)
            viewer.sync()
            if terminated or truncated:
                break