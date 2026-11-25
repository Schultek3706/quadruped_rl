from stable_baselines3 import PPO
from Learning.Env import RunClass
import time
import mujoco.viewer as view
import mujoco
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

def force_test(col_env,col_ml,rng):
    obs,info = col_env.reset()
    tp = 500 #testing-period for analysis
    force_point = np.linspace(100,119,20)
    print(force_point)
    force = rng.uniform((-1,-10,-1),(1,10,1),size = (3,))
    force_norm = -20*force/np.linalg.norm(force)
    point = rng.uniform((-0.08,-0.08,-0.02),(0.08,0.08,0.02),size = (3,))
    torque = point*force_norm
    yaw_errors = np.zeros((tp,))
    avg_speeds = np.zeros((tp,))

    with view.launch_passive(eval_env.model, eval_env.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = col_env.handler.base_id
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -30
        for i in range(50*10):
            action = col_ml.predict(obs,deterministic=True)[0]
            obs, reward, terminated, truncated, info = col_env.step(action)
            time.sleep(0.020)
            viewer.sync()

        for i in range(tp):
            action = col_ml.predict(obs,deterministic=True)[0]
            if i in force_point:
                col_env.data.xfrc_applied[col_env.handler.base_id] = np.zeros(6)
                col_env.data.xfrc_applied[col_env.handler.base_id] = np.concat((force_norm, torque))
                print("applied")
            if i - max(force_point) == 1:
                col_env.data.xfrc_applied[col_env.handler.base_id] = np.zeros(6)
            obs, reward, terminated, truncated, info = col_env.step(action)
            yaw_errors[i] = col_env.yaw_err.copy()
            avg_speeds[i] = col_env.avg_speed.copy()
            time.sleep(0.020)
            print(min(force_point)-i)
            viewer.sync()

    fig, ax = plt.subplots(1, 2, figsize=(13, 8))
    ax[0].plot(yaw_errors)
    ax[0].set_title("yaw error")
    ax[0].set_ylabel("rad")
    ax[0].set_xlabel("steps")
    ax[0].vlines(force_point,ymin= yaw_errors.min(),ymax= yaw_errors.max(),colors="red")
    ax[0].legend()

    ax[1].plot(avg_speeds)
    ax[1].set_title("avg speed")
    ax[1].set_ylabel("m/s")
    ax[1].set_xlabel("steps")
    ax[1].vlines(force_point,ymin= avg_speeds.min(),ymax= avg_speeds.max(),colors="red")
    ax[1].legend()

    plt.savefig("plots/force_" + current_model  + ".png")
    plt.show()

def tilt_test(col_env,col_ml,rng):
    phi = rng.uniform(-10,size = (3,))
    phi[2] = 0
    rot = R.from_euler('xyz',phi,degrees=True)
    grav = rot.apply(col_env.model.opt.gravity.copy())
    col_env.model.opt.gravity = grav
    obs, info = col_env.reset()
    tp = 500  # testing-period for analysis
    yaw_errors = np.zeros((tp,))
    avg_speeds = np.zeros((tp,))

    with view.launch_passive(eval_env.model, eval_env.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = col_env.handler.base_id
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -30
        for i in range(50*10):
            action = col_ml.predict(obs,deterministic=True)[0]
            obs, reward, terminated, truncated, info = col_env.step(action)
            time.sleep(0.020)
            viewer.sync()

        for i in range(tp):
            action = col_ml.predict(obs,deterministic=True)[0]
            obs, reward, terminated, truncated, info = col_env.step(action)
            yaw_errors[i] = col_env.yaw_err.copy()
            avg_speeds[i] = col_env.avg_speed.copy()
            time.sleep(0.020)
            viewer.sync()
            if terminated or truncated:
                break
    fig, ax = plt.subplots(1, 2, figsize=(13, 8))
    ax[0].plot(yaw_errors)
    ax[0].set_title("yaw error")
    ax[0].set_ylabel("rad")
    ax[0].set_xlabel("steps")
    ax[0].legend()

    ax[1].plot(avg_speeds)
    ax[1].set_title("avg speed")
    ax[1].set_ylabel("m/s")
    ax[1].set_xlabel("steps")
    ax[1].legend()

    plt.savefig("plots/tilt_" + current_model  + ".png")
    plt.show()

def stuck_test(col_env,col_ml,rng):
    sel_servo = np.random.randint(12)
    print("sel_servo", sel_servo)
    obs, info = col_env.reset()
    tp = 500  # testing-period for analysis
    stuck_point = np.linspace(100, 499, 400)
    yaw_errors = np.zeros((tp,))
    avg_speeds = np.zeros((tp,))

    with view.launch_passive(eval_env.model, eval_env.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = col_env.handler.base_id
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -30
        for i in range(50*10):
            action = col_ml.predict(obs,deterministic=True)[0]
            obs, reward, terminated, truncated, info = col_env.step(action)
            time.sleep(0.020)
            viewer.sync()
        stuck_action = action[sel_servo]
        for i in range(tp):
            action = col_ml.predict(obs,deterministic=True)[0]
            if i in stuck_point:
                action[sel_servo] = stuck_action
            obs, reward, terminated, truncated, info = col_env.step(action)
            yaw_errors[i] = col_env.yaw_err.copy()
            avg_speeds[i] = col_env.avg_speed.copy()
            time.sleep(0.020)
            viewer.sync()
            if terminated or truncated:
                break
    fig, ax = plt.subplots(1, 2, figsize=(13, 8))
    ax[0].plot(yaw_errors)
    ax[0].set_title("yaw error")
    ax[0].set_ylabel("rad")
    ax[0].set_xlabel("steps")
    ax[0].legend()

    ax[1].plot(avg_speeds)
    ax[1].set_title("avg speed")
    ax[1].set_ylabel("m/s")
    ax[1].set_xlabel("steps")
    ax[1].legend()

    plt.savefig("plots/tilt_" + current_model  + ".png")
    plt.show()

if __name__ == "__main__":
    current_model = "walker_v11"
    here = Path(__file__).resolve().parent
    project_root = here.parent
    model_path = project_root / "Learning/models" / current_model

    seed = 42
    rngs = np.random.default_rng(seed)
    ml = PPO.load(model_path, device="cpu")
    eval_env = RunClass(seed=seed, robot_path="robot.xml", ep_c=0, total_step=0)
    stuck_test(col_env=eval_env,col_ml=ml,rng=rngs)