#General Modules
import os
import time
import ctypes


#Environment and Deep Learning Related Modules
import ray

from ray import tune

from ray.rllib.agents import dqn
import copy
import gym
from gym import spaces

#Input Related Modules
import pydirectinput

#Image Modules
import cv2
import d3dshot

#Memory Reading Modules

from offsets import Offsets
from pymeow import *

#Open Process for memory Reading
mem = process_by_name("starwarsbattlefrontii.exe")


#Math Modules
import math
import numpy as np


class Entity:
    def __init__(self, addr):
        #Initial Values
        self.addr = addr
        self.team = 0
        self.health = 0
        self.height = 0.0
        self.pos3d = None
        self.pos2d = None
        self.visible = False
        self.alive = False
        self.float_health = 150

    def read(self):
        #Read values of pointers and offsets
        self.team = read_int(mem, self.addr + Offsets.Team)

        controlled = read_int64(mem, self.addr + Offsets.ControlledControllable)
        if controlled < 0:
            return

        try:
            health_comp = read_int64(mem, controlled + Offsets.HealthComponent)
        except:
            return

        self.height = read_float(mem, controlled + Offsets.Height)
        self.prev = self.float_health
        self.health = read_float(mem, health_comp + Offsets.Health)
        self.max_health = read_float(mem, health_comp + Offsets.MaxHealth)
        self.float_health = self.health
        #Is Enemy/Self Alive
        self.alive = self.health >= 0
        if not self.alive:
            return

        try:
            soldier = read_int64(mem, controlled + Offsets.SoldierPrediction)
            self.pos3d = read_vec3(mem, soldier + Offsets.Position)
        except:
            return
        #Is enemmy Visible
        self.visible = read_byte(mem, controlled + Offsets.Occluded) == 0

        return self

class CustomEnv(gym.Env):
    def __init__(self, config):
        # Initialize Default Gym Varibles
        self.num_envs = 1
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(0, 255, (42, 42, 3),dtype=np.uint8)
        #Enemy Pos Observation Parameter
        # Health Observation Parameter
        #space_1 = spaces.Box(-np.inf, np.inf, (1, ))
        #self.observation_space = spaces.Tuple((space_0, space_1))

        #Init Input Module Parameters
        pydirectinput.FAILSAFE = False
        pydirectinput.PAUSE = 0

        # Initial Pointers
        self.render_view = read_int(mem, read_int(mem, Offsets.GameRenderer) + Offsets.RenderView)
        game_context = read_int(mem, Offsets.ClientGameContext)
        self.player_manager = read_int(mem, game_context + Offsets.PlayerManager)
        self.client_array = read_int(mem, self.player_manager + Offsets.ClientArray)

        #Switch Statements
        self.enemy_vis = False
        self.sprint_list = [self.s_keyDown, self.s_keyUp]
        self.a_list = [self.s_keyDown, self.s_keyUp]
        self.d_list = [self.s_keyDown, self.s_keyUp]


        #Inital Varibles
        self.current_n_actions_taken = 0
        self.tup_current_pos3d = (0, 0, 0)
        self.tup_past_pos3d = self.tup_current_pos3d
        self.heat_ammo_wasted = 0
        self.time_still = 0
        self.dist_walked = 0

        #Init D3DShot for Screenshots
        self.d = d3dshot.create(capture_output="numpy")
        self.d.display = self.d.displays[0]

    def reset(self):
        # Reset Env/Get New Screenshot
        img = self.d.screenshot().astype(np.uint8)
        resized = cv2.resize(img, (42, 42), interpolation = cv2.INTER_NEAREST)
        reshaped = np.reshape(resized, (42,42, 3))
        self.current_n_actions_taken = 0
        self.heat_ammo_wasted = 0
        self.time_still = 0
        self.enemy_vis = False
        self.dist_walked = 0
        #obs = []
        #obs.append(reshaped)
        #obs.append([150])
        return reshaped

    def step(self, action):
        img = self.d.screenshot().astype(np.uint8)
        resized = cv2.resize(img, (42, 42), interpolation = cv2.INTER_NEAREST)
        self.tup_past_pos3d = self.tup_current_pos3d
        centerx, centery, done, health, prev, self.current_pos3d  = self.prediction()
        self.tup_current_pos3d = (self.current_pos3d["x"], self.current_pos3d["y"], self.current_pos3d["z"])
        reshaped = np.reshape(resized, (42,42, 3))
        reward = self.get_reward(action, centerx, centery, health, prev, done)
        if done:
            #Reset Game if needed and respawn
            time.sleep(11)
            pydirectinput.moveTo(470, 1025)
            pydirectinput.click()
            time.sleep(3.5)
            pydirectinput.keyDown("space")
            time.sleep(0.05)
            pydirectinput.keyUp("space")
            time.sleep(2)
        else:
            self.do_action(action)
        #Reset Enemy Visiblity for next step
        self.enemy_vis = False

        #Append OBS Parameter inputs
        #obs = []
        #obs.append(reshaped)
        #obs.append([health])
        return reshaped, reward, done, {}

    def render(self, mode="human", close=False):
        #Not Much can do here
        pass

    def wts(self, pos, vm):
        #Convert 3D Location to pixels on screen (relative to crosshair)
        w = vm[3] * pos["x"] + vm[7] * pos["y"] + vm[11] * pos["z"] + vm[15]
        if w < 0.3:
            raise Exception("WTS")

        x = vm[0] * pos["x"] + vm[4] * pos["y"] + vm[8] * pos["z"] + vm[12]
        y = vm[1] * pos["x"] + vm[5] * pos["y"] + vm[9] * pos["z"] + vm[13]

        return vec2(
            960 + 960 * x / w,
            540 + 540 * y / w,
        )


    def ent_loop(self):
        #Loop through entity list
        if self.client_array:
            clients = read_ints64(mem, self.client_array, 64 * 2)
            for ent_addr in clients:
                if ent_addr:
                    try:
                        e = Entity(ent_addr).read()
                    except:
                        continue

                    if e:
                        yield e

    def prediction(self):
        #Part of OBS Space and Reward about enemies
        local_player = read_int64(mem, self.player_manager + Offsets.LocalPlayer)
        local_player = Entity(local_player).read()
        centerX_list = [0]
        centerY_list = [0]
        if local_player:
            for ent in self.ent_loop():
                if ent.team == local_player.team:
                    continue

                vm = read_floats(mem, self.render_view + Offsets.ViewProj, 16)

                try:
                    ent.pos2d = self.wts(ent.pos3d, vm)
                except:
                    continue
                if ent.visible:
                    centerX_list.append(ent.pos2d["x"])
                    centerY_list.append(ent.pos2d["y"])
                    self.enemy_vis = True
            centerX_final = min(centerX_list, key=lambda x:abs(x-960))


            centerY_final = min(centerY_list, key=lambda x:abs(x-540))
            if self.enemy_vis:
                return centerX_final, centerY_final, False, local_player.float_health, local_player.prev, local_player.pos3d
            else:
                return 0, 0, False, local_player.float_health, local_player.prev, local_player.pos3d
        else:
            return 0, 0, True, 0, 0, {'x': 0, 'y': 0, 'z': 0}

    def get_reward(self, choice, centerx, centery, health, prev, done):
        #Calc Reward
        if done:
            reward = -1.0
            self.current_n_actions_taken = 0
            self.heat_ammo_wasted = 0
            self.time_still = 0
            self.dist_walked = 0
            return reward
        else:
            self.current_n_actions_taken += 1
            reward = -0.000075 * self.current_n_actions_taken

        if health < prev:
            reward -= (prev - health) * 0.05
        self.dist_walked = math.sqrt(sum([(a - b) ** 2 for a, b in zip(self.tup_current_pos3d, self.tup_past_pos3d)]))
        if abs(self.dist_walked) <= 0.15:
            self.time_still += 1
            reward -= 0.01 * self.time_still
        else:
            self.time_still = 0
        combat_action = int(choice)
        if self.enemy_vis:
            if (abs(abs(centerx) - 960) <= 250):
                reward += abs(abs(abs(centerx) - 960) - 250) * 0.002
                if combat_action == 0:
                    reward += 6.5
            if (abs(abs(centery) - 540)) <= 30:
                reward += 0.075
                if combat_action == 0:
                    reward += 1.0
        else:
            if combat_action == 0:
                self.heat_ammo_wasted += 1
                reward -= 0.05 * self.heat_ammo_wasted
        return reward

    #Key Functions for input switches
    def s_keyDown(self, key):
        pydirectinput.keyDown(key)

    def s_keyUp(self, key):
        pydirectinput.keyUp(key)

    #Keyboard and Mouse Actions
    def do_action(self, choice):
        #Convert Actions to list of actions
        '''
        actions = list(choice)

        #Convert list to each action
        movement_actions = int(actions[0])
        combat_actions = int(actions[1])
        mouse_action_type = int(actions[2])

        #Movement Actions
        if movement_actions == 0:
            func_to_call_sprint = self.sprint_list[0]
            func_to_call_sprint("w")
            func_to_call_sprint("shift")
            self.sprint_list.reverse()

        #Combat Actions
        if combat_actions == 0:
            pydirectinput.press("p")

        #Mouse Actions
        if mouse_action_type == 0:
            pydirectinput.moveRel(-200, 0, relative=True)
        elif mouse_action_type == 1:
            pydirectinput.moveRel(200, 0, relative=True)
        elif mouse_action_type == 2:
            pydirectinput.moveRel(0, -60, relative=True)
        elif mouse_action_type == 3:
            pydirectinput.moveRel(0, 60, relative=True)
        '''
        action = int(choice)
        if action == 0:
            pydirectinput.press("p")
        elif action == 1:
            func_to_call_sprint = self.sprint_list[0]
            func_to_call_sprint("w")
            func_to_call_sprint("shift")
            self.sprint_list.reverse()
        elif action == 2:
            pydirectinput.moveRel(-200, 0, relative=True)
        elif action == 3:
            pydirectinput.moveRel(200, 0, relative=True)
        elif action == 4:
            pydirectinput.moveRel(0, -60, relative=True)
        elif action == 5:
            pydirectinput.moveRel(0, 60, relative=True)


#Init Actor, Parameters, and Ray
ray.init(include_dashboard=False, num_gpus=1, local_mode=True)
'''
config = copy.deepcopy(ppo.DEFAULT_CONFIG)
config["framework"] = "tf2"
config["env"] = CustomEnv

config["num_workers"] = 1
config["num_envs_per_worker"] = 1

config["model"]["use_lstm"] = True
config["model"]["lstm_cell_size"] = 1024

config["batch_mode"] = "complete_episodes"

config["num_gpus"] = .575

config["lr"] = tune.choice([5e-2, 5e-3, 5e-4, 5e-5, 5e-6, 5e-7])
config["lambda"] = tune.choice([.9, .975, .99, 1.0])

results = tune.run(
    "PPO",
    scheduler = tune.schedulers.ASHAScheduler(
        time_attr='training_iteration',
        metric="episode_reward_mean",
        mode="max",
        max_t=3
        ),
    config=config
    )

'''
config = copy.deepcopy(dqn.DEFAULT_CONFIG)

config["framework"] = "torch"

#config["model"]["use_lstm"] = True
#config["model"]["lstm_cell_size"] = 1024

config["lr"] = 0.0000625

config["hiddens"] = [512]
config["model"]["fcnet_hiddens"] = [1024, 1024]

config["num_atoms"] = 51

config["noisy"] = True

config["exploration_config"]["epsilon_timesteps"] = 30000

config["learning_starts"] =  10000

config["num_gpus"] = .575


agent = dqn.DQNTrainer(env=CustomEnv, config=config)
# Continue Training
#agent.restore("C:/Users/Fidgety/ray_results/DQN_CustomEnv_2021-07-03_18-43-53syq_tsq2/checkpoint_22/checkpoint-22")


#Start Training and save model and actor/agent
i = 0
while True:
    if i % 1 == 0: #save every 10th training iteration
        checkpoint_path = agent.save()
        print(checkpoint_path)
    print(agent.train())
    i+=1
