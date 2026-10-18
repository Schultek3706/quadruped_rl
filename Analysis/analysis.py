from stable_baselines3 import PPO
from Learning.Env import RunClass
import time
import mujoco.viewer as view
import mujoco
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

current_model = "walker_v10"
here = Path(__file__).resolve().parent
project_root = here.parent
model_path = project_root / "Learning/models" / current_model

model = PPO.load(model_path, device="cpu")
eval_env = RunClass(seed=41,robot_path ="robot.xml",ep_c=0,total_step=0)

def data_collection(col_env,col_model):
    obs,info = col_env.reset()
    tp = 70 #testing-period for analysis
    feet_touches = np.zeros((4,tp))
    feet_height = np.zeros((4,tp))
    feet_vel = np.zeros((4,tp))
    ac_des_force = np.zeros((12,tp))
    for i in range(50*10):
        action = col_model.predict(obs,deterministic=True)[0]
        obs, reward, terminated, truncated, info = col_env.step(action)

    for i in range(tp):
        action = col_model.predict(obs,deterministic=True)[0]
        obs, reward, terminated, truncated, info = col_env.step(action)
        feet_touches[:,i] = col_env.handler.feet_data(col_env.data,col_env.model)[0]
        feet_height[:, i] = col_env.handler.feet_data(col_env.data,col_env.model)[1]
        feet_vel[:, i] = col_env.handler.feet_data(col_env.data,col_env.model)[2]
        ac_des_force[:,i] = col_env.handler.force_of(col_env.data,col_env.model,action)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0,0].plot(feet_touches[0,:],label="foot_1")
    ax[0,0].plot(feet_touches[1,:],label="foot_2")
    ax[0,0].plot(feet_touches[2,:],label="foot_3")
    ax[0,0].plot(feet_touches[3,:],label="foot_4")
    ax[0,0].set_xlabel("steps")
    ax[0,0].set_ylabel("binary")
    ax[0,0].set_title("touches")
    ax[0,0].legend()

    ax[0,1].plot(feet_height[0,:],label="foot_1")
    ax[0,1].plot(feet_height[1,:],label="foot_2")
    ax[0,1].plot(feet_height[2,:],label="foot_3")
    ax[0,1].plot(feet_height[3,:],label="foot_4")
    ax[0,1].set_xlabel("steps")
    ax[0,1].set_ylabel("height")
    ax[0,1].set_title("height")
    ax[0,1].legend()

    ax[1,0].plot(feet_vel[0,:],label="foot_1")
    ax[1,0].plot(feet_vel[1,:],label="foot_2")
    ax[1,0].plot(feet_vel[2,:],label="foot_3")
    ax[1,0].plot(feet_vel[3,:],label="foot_4")
    ax[1,0].set_xlabel("steps")
    ax[1,0].set_ylabel("velocity")
    ax[1,0].set_title("velocity")
    ax[1,0].legend()

    ax[1,1].plot(ac_des_force.T,label="diff desired vs usable ac_force")
    ax[1, 1].set_xlabel("steps")
    ax[1, 1].set_ylabel("force_diff")
    ax[1, 1].set_title("ac_force")
    ax[1, 1].legend()

    plt.savefig("plots/analysis_v10.png")
    plt.show()

data_collection(eval_env,model)