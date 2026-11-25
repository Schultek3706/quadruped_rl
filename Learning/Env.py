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
        frames_stacked = 10
        self.observation_space = gym.spaces.Box(low = -10*np.ones((44*frames_stacked,)),high = 10*np.ones((44*frames_stacked,)),shape = (44*frames_stacked,))
        self.obs = np.ones(44*frames_stacked)
        self.action_space = gym.spaces.Box(low=-2*np.ones(12),high=2*np.ones(12),shape = (12,))
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
        self.max_reward = 1
        self.k_reward = 0
        self.speed_goal = 0
        self.speed_command = 0

        self.yaw_base = 0
        self.yaw_err = 0

        self.rng = np.random.default_rng(self.seed)
        self.handler = DataHandler(self.model,self.seed)
        self.ep_c = ep_c
        self.ramp_step = 5000000/8
        self.episode_step = 0
        self.total_step = total_step
        self.plot_step = 10_000

        self.cols = ["vel_reward", "base_rew", "yaw_reward", "align_reward","pen_vel", "touch_pen",
                     "pen_rot_vel", "pen_grav","pen_action","pen_track","pen_body_acc","pen_force_ac",
                     "pen_sliding","pen_slam"]
        self.log = pd.DataFrame(columns=self.cols)
        self.ep_log = np.zeros((len(self.cols)))
        self.gen_metrics = {"avg_speed": np.array([]), "avg_reward": np.array([]),
                            "max_reward": np.array([]),"speed_goal":np.array([]),
                            "speed_command": np.array([]),"k_reward": np.array([])}
        self.reset()
    def reset(self, seed=None, options = None):
        super().reset(seed=seed)
        if seed != None:
            self.seed = seed
            self.rng = np.random.default_rng(self.seed)
            self.handler = DataHandler(self.model, self.seed)

        s = self.total_step / self.ramp_step
        s = np.clip(s, 0, 1)

        self.model, self.data = self.handler.reset_world_random(self.model,self.data,s)
        self.ep_c += 1
        self.k_reward += (self.max_reward - self.k_reward)*0.25
        self.yaw_base = self.rng.uniform(-np.pi, np.pi)*s
        self.speed_command = self.rng.uniform(0,1)
        for i in range(30):
            servo_pos = self.data.sensordata[self.handler.sensor_pos_adr].copy()
            self.last_action2 = self.last_action.copy()
            self.last_action = servo_pos
            goal_pos = servo_pos + s*self.rng.uniform(-0.02,0.02,servo_pos.shape)
            self.data.ctrl[:] = goal_pos
            max_speeds = self.handler.sub_step(self.model,self.data,10,np.zeros(3),np.zeros(3))
            #mujoco.mj_step(self.model, self.data, 10)
            obs = self.handler.get_obs(self.data,s)
            yaw = obs[-1].copy()
            obs = np.append(obs[:-1],self.speed_command)
            obs = np.append(obs,goal_pos.copy())
            self.yaw_err = self.limit_angle(yaw-self.yaw_base)
            obs = np.append(obs, self.yaw_err)
            self.obs = np.append(self.obs, obs)
            self.obs = np.delete(self.obs, np.s_[0:44])
        reward_data = self.handler.reward_data(self.model,self.data,goal_pos,max_speeds)
        self.orientation = np.ones(50)  # 1 seconds
        self.orientation = self.orientation.astype(bool)
        self.touchdown = np.ones(3)
        self.touchdown = self.touchdown.astype(bool)
        self.episode_step = 0
        self.avg_reward = 0
        self.ep_log = np.zeros((len(self.cols)))
        self.last_servo_vel = reward_data[4]
        self.last_v_world = reward_data[0].copy()
        info = {"speed": reward_data[0][0], "gravity_down": reward_data[1][2]}
        return self.obs.copy(), info
    def step(self, action): # 50hz control loop
        s = self.total_step/self.ramp_step
        s_push = self.total_step/self.ramp_step
        s = np.clip(s,0,1)
        s_push = np.clip(s_push,0,2)

        force = np.zeros(3)
        torque = np.zeros(3)
        if self.rng.uniform(0,1) < s*2e-3:
            force = self.rng.uniform([-10,-10,-1],[10,10,1])
            force = force/np.linalg.norm(force)*s_push*self.handler.body_mass.sum()/0.02
        ac_clipped = np.clip(action,-1,1)

        servo_goal = self.handler.set_servos(ac_clipped)
        self.data.ctrl[:] = servo_goal

        max_speed = self.handler.sub_step(self.model,self.data,10,force,torque)

        obs_data = self.handler.get_obs(self.data,s)
        yaw = obs_data[-1].copy()
        self.yaw_err = self.limit_angle(yaw-self.yaw_base)
        reward_data = self.handler.reward_data(self.model,self.data,ac_clipped,max_speed)
        touch = reward_data[7] + reward_data[8] + np.sum(reward_data[10])

        self.orientation = np.append(self.orientation,reward_data[1][2] < 0)[1:]
        self.touchdown = np.append(self.touchdown,touch == 0)[1:]

        reward = self.reward(reward_data,action,s)
        self.avg_reward = self.avg_reward * self.episode_step / (self.episode_step + 1) + reward / (
                    self.episode_step + 1)
        obs_data = np.append(obs_data[:-1], self.speed_command)
        obs_data = np.append(obs_data, action)
        obs_data = np.append(obs_data, self.yaw_err)
        self.obs = np.append(self.obs,obs_data)
        self.obs = np.delete(self.obs,np.s_[0:44])
        info = {"speed": reward_data[0][0],"gravity_down": reward_data[1][2]}
        self.last_action2 = self.last_action.copy()
        self.last_action = action

        if np.all(self.orientation == False) or np.all(self.touchdown == False):
            terminated = True
            if self.episode_step > 10*50 and self.avg_speed > self.speed_goal - 0.05:
                self.speed_goal = self.speed_goal + 0.05
                self.max_reward = self.avg_reward
                print(self.avg_speed)
            self.logging()
        else:
            terminated = False
        if self.episode_step >= 50*60:
            truncated = True
            if self.avg_speed > self.speed_goal - 0.05:
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
        v_world, grav, height, servo_pos, servo_vel, rot_vel, ac_force,base_touch, head_touch, yaw, knee_touch,force_over, feet_height, feet_vel, vel_vert,max_speeds = reward_data

        servo_goal = self.handler.set_servos(np.clip(action,-1,1))
        k = 1 - np.exp(-self.k_reward/2)

        temp_rews = {}

        yaw_tri = [np.cos(self.yaw_base),np.sin(self.yaw_base)]
        vel_fwd = yaw_tri[0]*v_world[0] + yaw_tri[1]*v_world[1]
        vel_lat = -yaw_tri[1]*v_world[0] + yaw_tri[0]*v_world[1]
        acc = (v_world - self.last_v_world) / 0.02

        yaw_foreward = [np.cos(yaw),np.sin(yaw)]
        v_base = yaw_foreward[0]*v_world[0] + yaw_foreward[1]*v_world[1]


        err = self.limit_angle(yaw - self.yaw_base)
        self.avg_speed = (self.avg_speed*self.episode_step + vel_fwd)/(self.episode_step + 1)


        temp_rews["base_rew"] = 10*max(0,vel_fwd)*(1-k)

        temp_rews["vel_reward"] = 10 * (1 - np.abs(self.speed_command - vel_fwd))
        temp_rews["yaw_reward"] = 2.5 * np.clip((1-3*np.abs(err)),-1,None)

        alive_reward = max(-1*grav[2],0)
        temp_rews["align_reward"] = 0.1*(1+np.cos(err))

        head_touch = -head_touch*3
        base_touch = -base_touch*3
        knee_touch = -np.sum(knee_touch)*3
        temp_rews["touch_pen"] = head_touch + base_touch + knee_touch
        temp_rews["pen_vel"] = -np.sqrt((vel_lat**2 + v_world[2]**2))
        temp_rews["pen_rot_vel"] = -0.002 * (2*rot_vel[0] ** 2 + 3*rot_vel[1] ** 2 + 4*rot_vel[2]**2)
        temp_rews["pen_grav"] = -0.5 * (grav[0] ** 2 + grav[1] ** 2)

        temp_rews["pen_action"] = -2*np.sum(np.maximum(np.abs(action) - 1.0, 0.0))*k
        temp_rews["pen_track"] = -0 * np.sum((servo_goal - servo_pos) ** 2)*k
        temp_rews["pen_body_acc"] = -0.01 * (acc[0] ** 2 + acc[1] ** 2 + 0.5 * acc[2] ** 2) * k
        force_ac = np.abs(self.handler.force_demand(self.data,action))/self.handler.ac_max_force - 0.65
        temp_rews["pen_force_ac"] = -0.1*self.func(np.mean(np.where(force_ac > 0, force_ac,0)))*k

        rel_feet_vel = np.where(np.array([feet_height])<0.0025,np.abs(feet_vel),0)
        temp_rews["pen_sliding"] = -0.25*np.sum(rel_feet_vel)*k
        feet_slam = np.where(max_speeds < 0, max_speeds, 0)**2
        temp_rews["pen_slam"] = -0.35*np.sum(feet_slam)*k

        rew_list = [temp_rews[terms] for terms in self.cols]
        total_reward = sum(rew_list) + alive_reward
        self.last_v_world = v_world.copy()
        self.ep_log = (self.ep_log * self.episode_step + rew_list) / (self.episode_step + 1)
        return total_reward

    def logging(self):
        self.gen_metrics["avg_reward"] = np.append(self.gen_metrics["avg_reward"],self.avg_reward)
        self.gen_metrics["max_reward"] = np.append(self.gen_metrics["max_reward"],self.max_reward)
        self.gen_metrics["avg_speed"] = np.append(self.gen_metrics["avg_speed"],self.avg_speed)
        self.gen_metrics["speed_goal"] = np.append(self.gen_metrics["speed_goal"],self.speed_goal)
        self.gen_metrics["speed_command"] = np.append(self.gen_metrics["speed_command"],self.speed_command)
        self.gen_metrics["k_reward"] = np.append(self.gen_metrics["k_reward"], self.k_reward)
        self.log.loc[self.ep_c] = self.ep_log.copy()
        if (self.seed == 41 or self.seed == 43) and self.plot_step - self.total_step < 0:
            self.plotting()
        pass

    def plotting(self):
        self.plot_step += 10_000
        rew = [name for name in self.cols if "rew" in name]
        full_rew = self.log.loc[:, rew].sum(axis=1).to_numpy()
        rew_pct = self.log.loc[:, rew].to_numpy() / full_rew[:, np.newaxis]
        pen = [name for name in self.cols if "pen" in name]
        full_pen = self.log.loc[:, pen].sum(axis=1).to_numpy()
        pen_pct = self.log.loc[:, pen].to_numpy() / full_pen[:, np.newaxis]

        fig, ax = plt.subplots(2, 3, figsize=(20, 8))
        ax[0, 0].plot(self.gen_metrics["avg_reward"], label="avg_reward")
        ax[0, 0].plot(self.gen_metrics["max_reward"], label="max_reward")
        ax[0, 0].plot(self.gen_metrics["k_reward"], label="k_reward")
        ax[0, 0].set_title("Rewards")
        ax[0, 0].set_xlabel("Episodes")
        ax[0, 0].set_ylabel("Rewards")
        ax[0, 1].plot(self.gen_metrics["avg_speed"], label="avg_speed")
        ax[0, 1].plot(self.gen_metrics["speed_goal"], label="speed_goal")
        ax[0,1].plot(self.gen_metrics["speed_command"], label="speed_command")
        ax[0, 1].set_title("Speed")
        ax[0, 1].set_xlabel("Episodes")
        ax[0, 1].set_ylabel("Velocity")

        ax[1, 0].plot(self.log.loc[:, rew], label=rew)
        ax[1, 0].set_title("Reward Components")
        ax[1, 0].set_xlabel("Episodes")
        ax[1, 0].set_ylabel("Rewards")
        ax[1, 1].plot(np.clip(np.abs(self.log.loc[:, pen]),-5,5), label=pen)
        ax[1, 1].set_title("Pen Components")
        ax[1, 1].set_xlabel("Episodes")
        ax[1, 1].set_ylabel("Penalty")

        ax[0, 2].plot(np.clip(rew_pct,-1,1), label=rew)
        ax[0, 2].set_title("Reward Components Pct")
        ax[0, 2].set_xlabel("Episodes")
        ax[1, 2].plot(np.clip(pen_pct,-1,1), label=pen)
        ax[1, 2].set_title("Penalty Components Pct")
        ax[1, 2].set_xlabel("Episodes")

        ax[0, 0].legend()
        ax[0, 1].legend()
        ax[1, 0].legend()
        ax[1, 1].legend()
        ax[0, 2].legend()
        ax[1, 2].legend()
        plt.savefig(fname="log_" + str(self.seed) + ".png")
        plt.close()
        pass
    @staticmethod
    def limit_angle(a):
        return (a + np.pi) % (2 * np.pi) - np.pi
    @staticmethod
    def func(a):
        return a + a**2