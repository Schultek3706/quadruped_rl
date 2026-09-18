from data_handler.Sim_data import DataHandler
import gymnasium as gym
import numpy as np
from pathlib import Path
import mujoco
import matplotlib.pyplot as plt

class RunClass(gym.Env):
    def __init__(self, robot_path, seed = None):
        super().__init__()
        here = Path(__file__).resolve().parent
        project_root = here.parent
        xml = project_root / "Robot" / robot_path
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        frames_stacked = 30
        self.observation_space = gym.spaces.Box(low = -10*np.ones((43*frames_stacked,)),high = 10*np.ones((43*frames_stacked,)),shape = (43*frames_stacked,),dtype=np.float32)
        self.obs = np.ones(43*frames_stacked)
        self.action_space = gym.spaces.Box(low=-1*np.ones(12),high=1*np.ones(12),shape = (12,))
        self.orientation = np.ones(100) #1 second
        self.orientation = self.orientation.astype(bool)
        self.seed = seed
        self.last_action = np.ones(12)
        self.last_action2 = np.ones(12)
        self.last_servo_vel = np.ones(12)
        self.avg_speed = 0
        self.avg_reward = 0
        self.max_reward = 0
        self.speed_goal = 0
        self.yaw_base = 0
        self.log = {"avg_reward": np.array([]), "max_reward": np.array([]), "avg_speed": np.array([]), "speed_goal": np.array([])}
        self.rng = np.random.default_rng(self.seed)
        self.handler = DataHandler(self.model,self.seed)
        self.ep_c = 0
        self.ep_total = 1000
        self.episode_step = 0
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
        for i in range(30):
            servo_pos = self.data.sensordata[self.handler.sensor_pos_adr].copy()
            self.last_action2 = self.last_action.copy()
            self.last_action = servo_pos
            goal_pos = servo_pos + s*self.rng.uniform(-0.05,0.05,servo_pos.shape)
            self.data.ctrl[:] = goal_pos
            mujoco.mj_step(self.model, self.data, 10)
            obs = self.handler.get_obs(self.data,s)
            obs = np.append(obs,self.speed_goal)
            obs = np.append(obs,goal_pos.copy())
            self.obs = np.append(self.obs, obs)
            self.obs = np.delete(self.obs, np.s_[0:43])
        reward_data = self.handler.reward_data(self.data)
        self.orientation = np.ones(100)  # 2 seconds
        self.orientation = self.orientation.astype(bool)
        self.episode_step = 0
        self.avg_reward = 0
        self.last_servo_vel = reward_data[4]
        self.yaw_base = reward_data[9]
        info = {"speed": reward_data[0][0], "gravity_down": reward_data[1][2]}
        return self.obs.copy(), info
    def step(self, action): # 50hz control loop
        s = self.ep_c/self.ep_total
        s = np.clip(s,0,1)

        servo_goal = self.handler.set_servos(action)
        self.data.ctrl[:] = servo_goal
        mujoco.mj_step(self.model,self.data,10)
        obs_data = self.handler.get_obs(self.data,s)
        reward_data = self.handler.reward_data(self.data)
        self.avg_speed = self.avg_speed*self.episode_step/(self.episode_step+1) + reward_data[0][0]/(self.episode_step+ 1)

        if reward_data[1][2] < 0:
            self.orientation = np.append(self.orientation,True)
        else:
            self.orientation = np.append(self.orientation,False)
        self.orientation = np.delete(self.orientation,0)
        reward = self.reward(reward_data,action,s)
        self.avg_reward = self.avg_reward * self.episode_step / (self.episode_step + 1) + reward / (
                    self.episode_step + 1)
        obs_data = np.append(obs_data, self.speed_goal)
        obs_data = np.append(obs_data, action)
        self.obs = np.append(self.obs,obs_data)
        self.obs = np.delete(self.obs,np.s_[0:43])
        info = {"speed": reward_data[0][0],"gravity_down": reward_data[1][2]}
        self.last_action2 = self.last_action.copy()
        self.last_action = action
        self.episode_step += 1
        if np.all(self.orientation == False):
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
            #print(self.avg_speed)
        else:
            truncated = False
        reward *= 0.002
        reward = np.clip(reward,0,None)
        return self.obs.copy(), reward, terminated, truncated, info
    def reward(self,reward_data,action,s):
        v_base, grav, height, servo_pos, servo_vel, rot_vel, ac_force,base_touch, head_touch, yaw = reward_data
        base_pose =np.array([0, 0.6, -1.1, 0, 0.6, -1.1, 0, 0.6, -1.1, 0, 0.6, -1.1])
        SIGMA = 0.25

        vel_reward = 5*v_base[0]*np.exp(-(self.avg_speed - v_base[0]) ** 2 / SIGMA)
        yaw_reward = 1*np.exp(-(yaw - self.yaw_base) ** 2 / SIGMA)
        if grav[2] < 0:
            alive_reward = 1
        else:
            alive_reward = 0
        head_touch = -head_touch*2
        base_touch = -base_touch*2
        pen_vel = -0.5*(v_base[1]**2 + v_base[2]**2)
        pen_rot_vel = -0.05 * (rot_vel[0] ** 2 + rot_vel[1] ** 2 + 2*rot_vel[2]**2)
        pen_grav = -0.1 * (grav[0] ** 2 + grav[1] ** 2)
        pen_height = -0*(height - 0.12)

        ac_vel = action - self.last_action
        ac_acc = action - 2 * self.last_action + self.last_action2
        pen_servo_pos = -0.05 * np.sum((base_pose - servo_pos) ** 2)
        pen_servo_vel = -2e-4*np.sum(servo_vel ** 2)
        pen_force = -1e-4*np.sum((ac_force ** 2))
        pen_ac = -0.01*np.sum((ac_vel) ** 2)
        pen_ac_acc = -0.001*np.sum((ac_acc) ** 2)
        total_reward = (vel_reward + alive_reward + (pen_vel + pen_rot_vel + pen_servo_pos + pen_grav + pen_height + pen_force +
                        pen_ac + pen_ac_acc + yaw_reward + pen_servo_vel + head_touch + base_touch))
        #print("total_reward",total_reward)
        return total_reward

    def logging(self):
        self.log["avg_reward"] = np.append(self.log["avg_reward"],self.avg_reward)
        self.log["max_reward"] = np.append(self.log["max_reward"],self.max_reward)
        self.log["avg_speed"] = np.append(self.log["avg_speed"],self.avg_speed)
        self.log["speed_goal"] = np.append(self.log["speed_goal"],self.speed_goal)
        fig, ax = plt.subplots(2,1)
        ax[0].plot(self.log["avg_reward"])
        ax[0].plot(self.log["max_reward"])
        ax[1].plot(self.log["avg_speed"])
        ax[1].plot(self.log["speed_goal"])
        plt.savefig("log_" + str(self.seed) + ".png")
        plt.close()
        pass