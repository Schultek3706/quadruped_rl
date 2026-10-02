from data_handler.Sim_data import DataHandler
import gymnasium as gym
import numpy as np
from pathlib import Path
import mujoco
import matplotlib.pyplot as plt
import pandas as pd

class RunClass(gym.Env):
    def __init__(self, robot_path, seed = None,ep_c:int = 0, total_step:int = 0):
        super().__init__()
        here = Path(__file__).resolve().parent
        project_root = here.parent
        xml = project_root / "Robot" / robot_path
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        frames_stacked = 30
        self.observation_space = gym.spaces.Box(low = -10*np.ones((44*frames_stacked,)),high = 10*np.ones((44*frames_stacked,)),shape = (44*frames_stacked,))
        self.obs = np.ones(44*frames_stacked)
        self.action_space = gym.spaces.Box(low=-3*np.ones(12),high=3*np.ones(12),shape = (12,))
        self.orientation = np.ones(50) #1 second
        self.orientation = self.orientation.astype(bool)
        self.touchdown = np.ones(3)
        self.touchdown = self.touchdown.astype(bool)
        self.seed = seed
        self.last_action = np.ones(12)
        self.last_action2 = np.ones(12)
        self.last_servo_vel = np.ones(12)
        self.last_v_world = np.zeros(3)
        self.avg_speed = 0
        self.avg_reward = 0
        self.max_reward = 0
        self.speed_goal = 0
        self.yaw_base = 0

        self.rng = np.random.default_rng(self.seed)
        self.handler = DataHandler(self.model,self.seed)
        self.ep_c = ep_c
        self.ep_total = 1000
        self.episode_step = 0
        self.total_step = total_step

        self.cols = ["vel_reward", "base_rew", "yaw_reward", "align_reward","pen_vel", "touch_pen",
                     "pen_rot_vel", "pen_grav", "pen_servo_pos", "pen_servo_vel",
                     "pen_force", "pen_ac", "pen_ac_acc","pen_action","pen_track","pen_body_acc"]
        self.log = pd.DataFrame(columns=self.cols)
        self.ep_log = np.zeros((len(self.cols)))
        self.gen_metrics = {"avg_speed": np.array([]), "avg_reward": np.array([]), "max_reward": np.array([]),"speed_goal":np.array([])}
        self.reset()
    def reset(self, seed=None, options = None):
        super().reset(seed=seed)
        if seed != None:
            self.seed = seed
            self.rng = np.random.default_rng(self.seed)
            self.handler = DataHandler(self.model, self.seed)

        s = self.ep_c / self.ep_total
        self.model, self.data = self.handler.reset_world_random(self.model,self.data,s)
        self.ep_c += 1
        self.yaw_base = self.rng.uniform(-0.3, 0.3)

        for i in range(30):
            servo_pos = self.data.sensordata[self.handler.sensor_pos_adr].copy()
            self.last_action2 = self.last_action.copy()
            self.last_action = servo_pos
            goal_pos = servo_pos + s*self.rng.uniform(-0.02,0.02,servo_pos.shape)
            self.data.ctrl[:] = goal_pos
            mujoco.mj_step(self.model, self.data, 10)
            obs = self.handler.get_obs(self.data,s)
            yaw = obs[-1].copy()
            obs = np.append(obs[:-1],1)
            obs = np.append(obs,goal_pos.copy())
            err = self.limit_angle(yaw-self.yaw_base)
            obs = np.append(obs, err)
            self.obs = np.append(self.obs, obs)
            self.obs = np.delete(self.obs, np.s_[0:44])
        reward_data = self.handler.reward_data(self.data)
        self.orientation = np.ones(50)  # 1 seconds
        self.orientation = self.orientation.astype(bool)
        self.touchdown = np.ones(3)
        self.touchdown = self.touchdown.astype(bool)
        self.episode_step = 0
        self.avg_reward = 0
        self.ep_log = np.zeros((len(self.cols)))
        self.last_servo_vel = reward_data[4]
        self.last_v_world = self.handler.reward_data(self.data)[0].copy()
        info = {"speed": reward_data[0][0], "gravity_down": reward_data[1][2]}
        return self.obs.copy(), info
    def step(self, action): # 50hz control loop
        s = self.ep_c/self.ep_total
        s = np.clip(s,0,1)

        servo_goal = self.handler.set_servos(np.clip(action,-1,1))
        self.data.ctrl[:] = servo_goal
        mujoco.mj_step(self.model,self.data,10)
        obs_data = self.handler.get_obs(self.data,s)
        yaw = obs_data[-1].copy()
        err = self.limit_angle(yaw-self.yaw_base)
        reward_data = self.handler.reward_data(self.data)

        self.orientation = np.append(self.orientation,reward_data[1][2] < 0)[1:]
        self.touchdown = np.append(self.touchdown,reward_data[7] == 0)[1:]

        reward = self.reward(reward_data,action,s)
        self.avg_reward = self.avg_reward * self.episode_step / (self.episode_step + 1) + reward / (
                    self.episode_step + 1)
        obs_data = np.append(obs_data[:-1], 1)
        obs_data = np.append(obs_data, action)
        obs_data = np.append(obs_data, err)
        self.obs = np.append(self.obs,obs_data)
        self.obs = np.delete(self.obs,np.s_[0:44])
        info = {"speed": reward_data[0][0],"gravity_down": reward_data[1][2]}
        self.last_action2 = self.last_action.copy()
        self.last_action = action

        if np.all(self.orientation == False) or np.all(self.touchdown == False):
            terminated = True
            if self.avg_reward > self.max_reward - 0.1 and self.episode_step > 30*50 and self.avg_speed > self.speed_goal - 0.025:
                self.speed_goal = self.speed_goal + 0.05
                self.max_reward = self.avg_reward
                print(self.avg_speed)
            self.logging()
        else:
            terminated = False
        if self.episode_step >= 50*60:
            truncated = True
            if self.avg_reward > self.max_reward - 0.1 and self.avg_speed > self.speed_goal - 0.025:
                self.speed_goal = self.speed_goal + 0.05
                self.max_reward = self.avg_reward
            self.logging()
        else:
            truncated = False
        reward *= 0.002
        reward = np.clip(reward,0,None)
        self.episode_step += 1
        self.total_step += 1
        return self.obs.copy(), reward, terminated, truncated, info
    def reward(self,reward_data,action,s):
        v_world, grav, height, servo_pos, servo_vel, rot_vel, ac_force,base_touch, head_touch, yaw, knee_touch = reward_data
        base_pose =np.array([0, 0.6, -1.1, 0, 0.6, -1.1, 0, 0.6, -1.1, 0, 0.6, -1.1])
        servo_goal = self.handler.set_servos(np.clip(action,-1,1))
        k = min(self.total_step/100_000,1.0)
        temp_rews = {}

        yaw_tri = [np.cos(self.yaw_base),np.sin(self.yaw_base)]
        vel_fwd = yaw_tri[0]*v_world[0] + yaw_tri[1]*v_world[1]
        vel_lat = -yaw_tri[1]*v_world[0] + yaw_tri[0]*v_world[1]
        acc = (v_world - self.last_v_world) / 0.02

        yaw_foreward = [np.cos(yaw),np.sin(yaw)]
        v_base = yaw_foreward[0]*v_world[0] + yaw_foreward[1]*v_world[1]

        SIGMA = 0.25
        SIGMA_YAW = 0.1
        err = self.limit_angle(yaw - self.yaw_base)
        self.avg_speed = (self.avg_speed*self.episode_step + vel_fwd)/(self.episode_step + 1)

        temp_rews["base_rew"] = 0.1*np.exp(v_base)
        temp_rews["vel_reward"]= vel_rew = 5*np.exp(-(self.avg_speed - vel_fwd) ** 2 / SIGMA)*max(0,vel_fwd)
        temp_rews["yaw_reward"] = np.exp(-err ** 2 / SIGMA)*vel_rew

        alive_reward = max(-0.1*grav[2],0)
        temp_rews["align_reward"] = 0.25*(1+np.cos(err))

        head_touch = -head_touch*1
        base_touch = -base_touch*1
        knee_touch = -np.sum(knee_touch)*1
        temp_rews["touch_pen"] = head_touch + base_touch + knee_touch
        temp_rews["pen_vel"] = -0.05*(vel_lat**2 + v_world[2]**2)
        temp_rews["pen_rot_vel"] = -0.002 * (2*rot_vel[0] ** 2 + 3*rot_vel[1] ** 2 + 4*rot_vel[2]**2)
        temp_rews["pen_grav"] = -0.5 * (grav[0] ** 2 + grav[1] ** 2)

        ac_vel = action - self.last_action
        ac_acc = action - 2 * self.last_action + self.last_action2
        temp_rews["pen_servo_pos"] = -0.001 * np.sum((base_pose - servo_pos) ** 2)
        temp_rews["pen_servo_vel"] = -2e-4*np.sum(servo_vel ** 2)/(40*temp_rews["base_rew"])*k
        temp_rews["pen_force"] = -0.01*np.sum((ac_force/0.65)**2)*k
        temp_rews["pen_ac"] = -0.1*np.sum((ac_vel) ** 2)*k
        temp_rews["pen_ac_acc"] = -0.01*np.sum((ac_acc) ** 2)*k
        temp_rews["pen_action"] = -2*np.sum(np.maximum(np.abs(action) - 1.0, 0.0) ** 2)
        temp_rews["pen_track"] = -0.02 * np.sum((servo_goal - servo_pos) ** 2)*k
        temp_rews["pen_body_acc"] = -0.005 * (acc[0] ** 2 + acc[1] ** 2 + 0.5 * acc[2] ** 2) * k

        rew_list = [temp_rews[terms] for terms in self.cols]
        total_reward = sum(rew_list) + alive_reward
        #print("total_reward",total_reward)
        self.last_v_world = v_world.copy()
        self.ep_log = (self.ep_log * self.episode_step + rew_list) / (self.episode_step + 1)
        return total_reward

    def logging(self):
        self.gen_metrics["avg_reward"] = np.append(self.gen_metrics["avg_reward"],self.avg_reward)
        self.gen_metrics["max_reward"] = np.append(self.gen_metrics["max_reward"],self.max_reward)
        self.gen_metrics["avg_speed"] = np.append(self.gen_metrics["avg_speed"],self.avg_speed)
        self.gen_metrics["speed_goal"] = np.append(self.gen_metrics["speed_goal"],self.speed_goal)
        self.log.loc[self.ep_c] = self.ep_log.copy()
        rew = [name for name in self.cols if "rew" in name]
        pen = [name for name in self.cols if "pen" in name]
        fig, ax = plt.subplots(2,2,figsize=(13,8))
        ax[0,0].plot(self.gen_metrics["avg_reward"],label="avg_reward")
        ax[0,0].plot(self.gen_metrics["max_reward"],label="max_reward")
        ax[0,0].set_title("Rewards")
        ax[0,0].set_xlabel("Episodes")
        ax[0,0].set_ylabel("Rewards")
        ax[0,1].plot(self.gen_metrics["avg_speed"],label="avg_speed")
        ax[0,1].plot(self.gen_metrics["speed_goal"],label="speed_goal")
        ax[0,1].set_title("Speed")
        ax[0,1].set_xlabel("Episodes")
        ax[0,1].set_ylabel("Velocity")

        ax[1,0].plot(self.log.loc[:,rew],label=rew)
        ax[1,0].set_title("Reward Components")
        ax[1,0].set_xlabel("Episodes")
        ax[1,0].set_ylabel("Rewards")
        ax[1,1].plot(np.abs(self.log.loc[:,pen]),label=pen)
        ax[1,1].set_title("Pen Components")
        ax[1,1].set_xlabel("Episodes")
        ax[1,1].set_ylabel("Penalty")
        ax[0,0].legend()
        ax[0,1].legend()
        ax[1,0].legend()
        ax[1,1].legend()
        plt.savefig("log_" + str(self.seed) + ".png")
        plt.close()
        pass
    @staticmethod
    def limit_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi