import time

from math import sqrt
import numpy as np

from sb3_contrib import RecurrentPPO

import gym
from gym import spaces

import d3dshot

from offsets import Offsets
from pymeow import *

mem = process_by_name("starwarsbattlefrontii.exe")

center_pos = (960, 540)

import mouse
import keyboard


class Entity:
    def __init__(self, addr):
        self.addr = addr
        self.team = 0
        self.health = 0
        self.height = 0.0
        self.pos3d = None
        self.headpos3d = None
        self.pos2d = None
        self.headpos2d = None
        self.visible = False
        self.alive = False

    def read(self):
        self.team = read_int(mem, self.addr + Offsets.Team)

        controlled = read_int64(mem, self.addr + Offsets.ControlledControllable)
        if controlled < 0:
            return

        try:
            health_comp = read_int64(mem, controlled + Offsets.HealthComponent)
        except:
            return

        self.height = read_float(mem, controlled + Offsets.Height)
        self.health = read_float(mem, health_comp + Offsets.Health)
        self.alive = self.health > 0
        if not self.alive:
            return

        try:
            soldier = read_int64(mem, controlled + Offsets.SoldierPrediction)
            self.pos3d = read_vec3(mem, soldier + Offsets.Position)
            self.headpos3d = vec3(
                self.pos3d["x"], self.pos3d["y"] + self.height - 18.5, self.pos3d["z"]
            )
        except:
            return

        self.visible = read_byte(mem, controlled + Offsets.Occluded) == 0

        return self


class StarWars_v0(gym.Env):
    def __init__(self):
        self.action_space = spaces.MultiDiscrete((4, 3))
        self.observation_space = spaces.Box(low=0, high=255, shape=(90, 160, 3), dtype=np.uint8)
        
        self.render_view = read_int(mem, read_int(mem, Offsets.GameRenderer) + Offsets.RenderView)
        game_context = read_int(mem, Offsets.ClientGameContext)
        self.player_manager = read_int(mem, game_context + Offsets.PlayerManager)
        self.client_array = read_int(mem, self.player_manager + Offsets.ClientArray)

        
        self.current_dist = 600
        self.hp = 150
        self.is_moving = False
        
        self.d = d3dshot.create(capture_output="numpy")

        self.reset()
    def wts(self, pos, vm):
        w = vm[3] * pos["x"] + vm[7] * pos["y"] + vm[11] * pos["z"] + vm[15]
        if w < 0.3:
            return False

        x = vm[0] * pos["x"] + vm[4] * pos["y"] + vm[8] * pos["z"] + vm[12]
        y = vm[1] * pos["x"] + vm[5] * pos["y"] + vm[9] * pos["z"] + vm[13]

        return vec2(
            960 + 960 * x / w,
            540 + 540 * y / w,
        )

    def ent_loop(self):
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

    def get_distance(self, p, q):
        p1_x, p1_y = p
        p2_x, p2_y = q

        distance = sqrt((p1_x-p2_x)**2 + (p1_y-p2_y)**2)
        return distance

    def calc_info(self):
        local_player = read_int64(mem, self.player_manager + Offsets.LocalPlayer)
        local_player = Entity(local_player).read()
        nearest = 10000000
        if local_player:
            vm = read_floats(mem, self.render_view + Offsets.ViewProj, 16)
            for ent in self.ent_loop():
                if ent.team == local_player.team:
                    continue

                ent.pos2d = self.wts(ent.pos3d, vm)
                ent.headpos2d = self.wts(ent.headpos3d, vm)

                if ent.pos2d and ent.headpos2d:
                    if ent.visible:
                        head = ent.headpos2d["y"] - ent.pos2d["y"]

                        x_final = ent.pos2d["x"]
                        y_final = ent.pos2d["y"] + head

                        headpos = (x_final, y_final)

                        dis = self.get_distance(center_pos, headpos)
                        
                        if nearest > dis:
                            nearest = dis
            return abs(nearest), local_player.health, False
        else:
            return abs(nearest), 150, True           
    def rew_func(self, action_choice):
        self.past_dist = self.current_dist
        self.past_hp = self.hp

        self.current_dist, self.hp, done = self.calc_info()

        reward = -0.05

        if self.past_dist > self.current_dist:
            enemy_gain = 600 - self.current_dist
            reward += enemy_gain * 0.05
        
        if action_choice == 1:
            if self.current_dist < 200:
                calc_action_reward = 200 - self.current_dist * .75
                reward += calc_action_reward

            else:
                reward -= 50
        
        if self.past_dist < self.current_dist:
            self.current_dist = 600
            self.past_dist = 600
        
        if self.hp < self.past_hp:
            # 150 is max health for assault class
            health_loss_r = self.past_hp - self.hp
            reward -= health_loss_r * .8
        return reward, done

    def do_action(self, choice_camera, choice_move):
        if choice_camera == 0:
            keyboard.press('7')
            keyboard.release('8')
        elif choice_camera == 1:
            keyboard.press('8')
            keyboard.release('7')
        else:
            keyboard.release('7')
            keyboard.release('8')

        if choice_move == 0:
            if self.is_moving:
                keyboard.release('shift, w')
                self.is_moving = False
            else:
                keyboard.press('shift, w')
                self.is_moving = True
        elif choice_move == 1:
            keyboard.press("4")
            keyboard.release("4")

    def reset(self):
        self.is_moving = False
        obs = np.zeros([90, 160, 3], dtype=np.uint8)
        obs.fill(255)
        return obs

    def step(self, action):
        l_choices = list(action)

        camera_choice = int(l_choices[1])
        move_choice = int(l_choices[0])

        reward, done = self.rew_func(move_choice)

        if done:
            keyboard.release('shift, w')
            
            keyboard.release('7')
            keyboard.release('8')
            
            time.sleep(11)
            mouse.move(470, 1025)
            mouse.click()
            time.sleep(3.5)
            keyboard.press('space')
            time.sleep(0.075)
            keyboard.release('space')
            time.sleep(1.5)
            
            reward = -250

            obs = np.zeros([90, 160, 3], dtype=np.uint8)
            obs.fill(255)
        else:
            self.do_action(camera_choice, move_choice)

            scr = self.d.screenshot()
            obs = scr[::12, ::12]
        return obs, reward, done, {}

if __name__ == "__main__":
    log_dir = "D:/Projects/StarWarsAI/logs"
    policy_kwargs = dict(lstm_hidden_size=512)
    
    model = RecurrentPPO("CnnLstmPolicy", StarWars_v0(), verbose=1, learning_rate=0.00008, gamma=0.8, ent_coef=0.04, tensorboard_log=log_dir, policy_kwargs=policy_kwargs)
    model.learn(1_250_000)
    model.save("model")

    
