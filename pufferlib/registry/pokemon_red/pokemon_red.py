from pdb import set_trace as T
import sys
import uuid 
import os
from math import floor, sqrt
import json
from pathlib import Path

import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from skimage.transform import resize
from pyboy import PyBoy
import hnswlib
import mediapy as media
import pandas as pd
from io import BytesIO

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent


MAP_COORDS = {
    0: {"name": "Pallet Town", "coordinates": np.array([70, 7])},
    1: {"name": "Viridian City", "coordinates": np.array([60, 79])},
    2: {"name": "Pewter City", "coordinates": np.array([60, 187])},
    3: {"name": "Cerulean City", "coordinates": np.array([240, 205])},
    62: {"name": "Invaded house (Cerulean City)", "coordinates": np.array([290, 227])},
    63: {"name": "trade house (Cerulean City)", "coordinates": np.array([290, 212])},
    64: {"name": "Pokémon Center (Cerulean City)", "coordinates": np.array([290, 197])},
    65: {"name": "Pokémon Gym (Cerulean City)", "coordinates": np.array([290, 182])},
    66: {"name": "Bike Shop (Cerulean City)", "coordinates": np.array([290, 167])},
    67: {"name": "Poké Mart (Cerulean City)", "coordinates": np.array([290, 152])},
    35: {"name": "Route 24", "coordinates": np.array([250, 235])},
    36: {"name": "Route 25", "coordinates": np.array([270, 267])},
    12: {"name": "Route 1", "coordinates": np.array([70, 43])},
    13: {"name": "Route 2", "coordinates": np.array([70, 151])},
    14: {"name": "Route 3", "coordinates": np.array([100, 179])},
    15: {"name": "Route 4", "coordinates": np.array([150, 197])},
    33: {"name": "Route 22", "coordinates": np.array([20, 71])},
    37: {"name": "Red house first", "coordinates": np.array([61, 9])},
    38: {"name": "Red house second", "coordinates": np.array([61, 0])},
    39: {"name": "Blues house", "coordinates": np.array([91, 9])},
    40: {"name": "oaks lab", "coordinates": np.array([91, 1])},
    41: {"name": "Pokémon Center (Viridian City)", "coordinates": np.array([100, 54])},
    42: {"name": "Poké Mart (Viridian City)", "coordinates": np.array([100, 62])},
    43: {"name": "School (Viridian City)", "coordinates": np.array([100, 79])},
    44: {"name": "House 1 (Viridian City)", "coordinates": np.array([100, 71])},
    47: {"name": "Gate (Viridian City/Pewter City) (Route 2)", "coordinates": np.array([91,143])},
    49: {"name": "Gate (Route 2)", "coordinates": np.array([91,115])},
    50: {"name": "Gate (Route 2/Viridian Forest) (Route 2)", "coordinates": np.array([91,115])},
    51: {"name": "viridian forest", "coordinates": np.array([35, 144])},
    52: {"name": "Pewter Museum (floor 1)", "coordinates": np.array([60, 196])},
    53: {"name": "Pewter Museum (floor 2)", "coordinates": np.array([60, 205])},
    54: {"name": "Pokémon Gym (Pewter City)", "coordinates": np.array([49, 176])},
    55: {"name": "House with disobedient Nidoran♂ (Pewter City)", "coordinates": np.array([51, 184])},
    56: {"name": "Poké Mart (Pewter City)", "coordinates": np.array([40, 170])},
    57: {"name": "House with two Trainers (Pewter City)", "coordinates": np.array([51, 184])},
    58: {"name": "Pokémon Center (Pewter City)", "coordinates": np.array([45, 161])},
    59: {"name": "Mt. Moon (Route 3 entrance)", "coordinates": np.array([153, 234])},
    60: {"name": "Mt. Moon Corridors", "coordinates": np.array([168, 253])},
    61: {"name": "Mt. Moon Level 2", "coordinates": np.array([197, 253])},
    68: {"name": "Pokémon Center (Route 3)", "coordinates": np.array([135, 197])},
    193: {"name": "Badges check gate (Route 22)", "coordinates": np.array([0, 87])}, # TODO this coord is guessed, needs to be updated
    230: {"name": "Badge Man House (Cerulean City)", "coordinates": np.array([290, 137])}
}

# addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
# https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
HP_ADDR =  [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
MAX_HP_ADDR = [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
PARTY_SIZE_ADDR = 0xD163
PARTY_ADDR = [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
PARTY_LEVEL_ADDR = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
POKE_XP_ADDR = [0xD179, 0xD1A5, 0xD1D1, 0xD1FD, 0xD229, 0xD255]
CAUGHT_POKE_ADDR = range(0xD2F7, 0xD309)
SEEN_POKE_ADDR = range(0xD30A, 0xD31D)
OPPONENT_LEVEL_ADDR = [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]
X_POS_ADDR = 0xD362
Y_POS_ADDR = 0xD361
MAP_N_ADDR = 0xD35E
BADGE_1_ADDR = 0xD356
OAK_PARCEL_ADDR = 0xD74E
OAK_POKEDEX_ADDR = 0xD74B
OPPONENT_LEVEL = 0xCFF3
ENEMY_POKE_COUNT = 0xD89C
EVENT_FLAGS_START_ADDR = 0xD747
EVENT_FLAGS_END_ADDR = 0xD761
MUSEUM_TICKET_ADDR = 0xD754
MONEY_ADDR_1 = 0xD347
MONEY_ADDR_100 = 0xD348
MONEY_ADDR_10000 = 0xD349

ACTIONS = [
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
]

EXTRA_ACTIONS = [
    WindowEvent.PRESS_BUTTON_START,
    WindowEvent.PASS,
]

RELEASE_ARROW = [
    WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.RELEASE_ARROW_UP
]

RELEASE_BUTTON = [
    WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.RELEASE_BUTTON_B
]



def local_to_global_coord(x, y, map_idx):
    map_x, map_y = MAP_COORDS[map_idx]["coordinates"]
    return x + map_x, y + (375 - map_y)

class PokemonRed(Env):
    def __init__(
            self,
            headless=True,
            save_final_state=False,
            early_stop=False,
            action_freq=24,
            max_steps=2048*10, 
            print_rewards=False,
            save_video=False,
            fast_video=True,
            debug=False,
            sim_frame_dist=2_000_000.0, 
            use_screen_explore=True,
            reward_scale=4,
            extra_buttons=False,
            explore_weight=3, # 2.5
            init_state='has_pokedex_nballs.state',
        ):
        self.s_path = Path(f'session_{str(uuid.uuid4())[:8]}')
        self.gb_path=str(Path(__file__).parent / 'pokemon_red.gb')
        self.init_state=str(Path(__file__).parent / init_state)
        
        self.debug = debug
        self.save_final_state = save_final_state
        self.print_rewards = print_rewards
        self.vec_dim = 4320 #1000
        self.headless = headless
        self.num_elements = 20000 # max
        self.act_freq = action_freq
        self.max_steps = max_steps
        self.early_stopping = early_stop
        self.save_video = save_video
        self.fast_video = fast_video
        self.video_interval = 256 * action_freq
        self.downsample_factor = 2
        self.frame_stacks = 3
        self.explore_weight = explore_weight
        self.use_screen_explore = use_screen_explore
        self.similar_frame_dist = sim_frame_dist
        self.reward_scale = reward_scale
        self.extra_buttons = extra_buttons
        self.instance_id = str(uuid.uuid4())[:8]

        self.s_path.mkdir(exist_ok=True)
        self.reset_count = 0
        self.all_runs = []

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.output_shape = (36, 40, 3)
        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 16
        self.output_full = (
            self.output_shape[0] * self.frame_stacks + 2 * (self.mem_padding + self.memory_height),
            self.output_shape[1],
            self.output_shape[2]
        )

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8)

        head = 'headless' if headless else 'SDL2'

        self.pyboy = PyBoy(
            self.gb_path,
            debugging=False,
            disable_input=False,
            window_type=head,
            hide_window='--quiet' in sys.argv,
        )

        self.screen = self.pyboy.botsupport_manager().screen()

        if not self.headless:
            self.pyboy.set_emulation_speed(6)

        with open(self.init_state, 'rb') as f:
            self.initial_state = BytesIO(f.read())
            
        self.counts_map = np.zeros((375, 500))
        self.reset()

    def reset(self, seed=None):
        self.seed = seed
        # restart game, skipping credits
        self.initial_state.seek(0)
        self.pyboy.load_state(self.initial_state)
        
        if self.use_screen_explore:
            self.init_knn()
        else:
            self.init_map_mem()

        self.recent_memory = np.zeros((self.output_shape[1]*self.memory_height, 3), dtype=np.uint8)
        
        self.recent_frames = np.zeros(
            (self.frame_stacks, self.output_shape[0], 
             self.output_shape[1], self.output_shape[2]),
            dtype=np.uint8)

        self.seen_maps = set()
        self.agent_stats = []
        
        if self.save_video:
            base_dir = self.s_path / Path('rollouts')
            base_dir.mkdir(exist_ok=True)
            full_name = Path(f'full_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            model_name = Path(f'model_reset_{self.reset_count}_id{self.instance_id}').with_suffix('.mp4')
            self.full_frame_writer = media.VideoWriter(base_dir / full_name, (144, 160), fps=60)
            self.full_frame_writer.__enter__()
            self.model_frame_writer = media.VideoWriter(base_dir / model_name, self.output_full[:2], fps=60)
            self.model_frame_writer.__enter__()
       
        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        return self.render(), {}
    
    def init_knn(self):
        # Declaring index
        self.knn_index = hnswlib.Index(space='l2', dim=self.vec_dim) # possible options are l2, cosine or ip
        # Initing index - the maximum number of elements should be known beforehand
        self.knn_index.init_index(
            max_elements=self.num_elements, ef_construction=100, M=16)
        
    def init_map_mem(self):
        self.seen_coords = {}

    def render(self, reduce_res=True, add_memory=True, update_mem=True):
        game_pixels_render = self.screen.screen_ndarray() # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (255*resize(game_pixels_render, self.output_shape)).astype(np.uint8)

            x_pos = self.read_m(X_POS_ADDR)
            y_pos = self.read_m(Y_POS_ADDR)
            map_n = self.read_m(MAP_N_ADDR)

            x, y = local_to_global_coord(x_pos, y_pos, map_n)
            try:
                self.counts_map[y, x] += 1
            except:
                pass #TODO: Ensure dims correct

            if update_mem:
                self.recent_frames[0] = game_pixels_render
            if add_memory:
                pad = np.zeros(
                    shape=(self.mem_padding, self.output_shape[1], 3), 
                    dtype=np.uint8)
                game_pixels_render = np.concatenate((
                    self.create_exploration_memory(), 
                    pad,
                    self.create_recent_memory(),
                    pad,
                    rearrange(self.recent_frames, 'f h w c -> (f h) w c')
                ), axis=0)
        return game_pixels_render
    
    def step(self, action):

        self.run_action_on_emulator(action)
        self.append_agent_stats(action)

        self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        obs_memory = self.render()

        # trim off memory from frame for knn index
        frame_start = 2 * (self.memory_height + self.mem_padding)
        obs_flat = obs_memory[
            frame_start:frame_start+self.output_shape[0], ...].flatten().astype(np.float32)

        if self.use_screen_explore:
            self.update_frame_knn_index(obs_flat)
        else:
            self.update_seen_coords()
            
        self.update_heal_reward()
        self.party_size = self.read_m(PARTY_SIZE_ADDR)

        new_reward, new_prog = self.update_reward()
        
        self.last_health = self.read_hp_fraction()

        # shift over short term reward memory
        self.recent_memory = np.roll(self.recent_memory, 3)
        self.recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        self.recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        self.recent_memory[0, 2] = min(new_prog[2] * 128, 255)

        step_limit_reached = self.check_if_done()
        info = {}
        if step_limit_reached:
            info = self.agent_stats[-1]

        #self.save_and_print_info(step_limit_reached, obs_memory)
        if self.save_video and step_limit_reached:
            self.full_frame_writer.close()
            self.model_frame_writer.close()

        self.step_count += 1

        return obs_memory, new_reward*0.1, step_limit_reached, step_limit_reached, info

    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(ACTIONS[action])
        # disable rendering when we don't need it
        if self.headless:
            self.pyboy._rendering(False)
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(RELEASE_ARROW[action])
                if action > 3 and action < 6:
                    # release button 
                    self.pyboy.send_input(RELEASE_BUTTON[action - 4])
                if ACTIONS[action] == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
            if self.save_video and not self.fast_video:
                self.add_video_frame()
            if i == self.act_freq-1:
                self.pyboy._rendering(True)
            self.pyboy.tick()
        if self.save_video and self.fast_video:
            self.add_video_frame()
    
    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render(reduce_res=False, update_mem=False))
        self.model_frame_writer.add_image(self.render(reduce_res=True, update_mem=False))
    
    def append_agent_stats(self, action):
        x_pos = self.read_m(X_POS_ADDR)
        y_pos = self.read_m(Y_POS_ADDR)
        map_n = self.read_m(MAP_N_ADDR)
        self.seen_maps.add(map_n)

        levels = [self.read_m(a) for a in PARTY_LEVEL_ADDR]
        if self.use_screen_explore:
            expl = ('frames', self.knn_index.get_current_count())
        else:
            expl = ('coord_count', len(self.seen_coords))
        self.agent_stats.append({
            'step': self.step_count,
            'x': x_pos,
            'y': y_pos,
            'map': map_n,
            'map_n': len(self.seen_maps),
            'exploration_map': self.counts_map,
            'last_action': action,
            'pcount': self.read_m(PARTY_SIZE_ADDR),
            'levels': levels,
            'level': sum(levels),
            'ptypes': self.read_party(),
            'hp': self.read_hp_fraction(),
            expl[0]: expl[1],
            'deaths': self.died_count,
            'badge': self.get_badges(),
            'badge_2': float(self.get_badges() > 1),
            'event': self.progress_reward['event'],
            'healr': self.total_healing_rew
        })

    def update_frame_knn_index(self, frame_vec):
        
        if self.get_levels_sum() >= 22 and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = self.knn_index.get_current_count()
            self.init_knn()

        if self.knn_index.get_current_count() == 0:
            # if index is empty add current frame
            self.knn_index.add_items(
                frame_vec, np.array([self.knn_index.get_current_count()])
            )
        else:
            # check for nearest frame and add if current 
            labels, distances = self.knn_index.knn_query(frame_vec, k = 1)
            if distances[0][0] > self.similar_frame_dist:
                # print(f"distances[0][0] : {distances[0][0]} similar_frame_dist : {self.similar_frame_dist}")
                self.knn_index.add_items(
                    frame_vec, np.array([self.knn_index.get_current_count()])
                )
    
    def update_seen_coords(self):
        x_pos = self.read_m(X_POS_ADDR)
        y_pos = self.read_m(Y_POS_ADDR)
        map_n = self.read_m(MAP_N_ADDR)
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        if self.get_levels_sum() >= 22 and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = len(self.seen_coords)
            self.seen_coords = {}
        
        self.seen_coords[coord_string] = self.step_count

    def update_reward(self):
        # compute reward
        old_prog = self.group_rewards()
        self.progress_reward = self.get_game_state_reward()
        new_prog = self.group_rewards()
        new_total = sum([val for _, val in self.progress_reward.items()]) #sqrt(self.explore_reward * self.progress_reward)
        new_step = new_total - self.total_reward
        if new_step < 0 and self.read_hp_fraction() > 0:
            #print(f'\n\nreward went down! {self.progress_reward}\n\n')
            self.save_screenshot('neg_reward')
    
        self.total_reward = new_total
        return (new_step, 
                   (new_prog[0]-old_prog[0], 
                    new_prog[1]-old_prog[1], 
                    new_prog[2]-old_prog[2])
               )
    
    def group_rewards(self):
        prog = self.progress_reward
        # these values are only used by memory
        return (prog['level'] * 100 / self.reward_scale, 
                self.read_hp_fraction()*2000, 
                prog['explore'] * 150 / (self.explore_weight * self.reward_scale))
               #(prog['events'], 
               # prog['levels'] + prog['party_xp'], 
               # prog['explore'])

    def create_exploration_memory(self):
        w = self.output_shape[1]
        h = self.memory_height
        
        def make_reward_channel(r_val):
            col_steps = self.col_steps
            max_r_val = (w-1) * h * col_steps
            # truncate progress bar. if hitting this
            # you should scale down the reward in group_rewards!
            r_val = min(r_val, max_r_val)
            row = floor(r_val / (h * col_steps))
            memory = np.zeros(shape=(h, w), dtype=np.uint8)
            memory[:, :row] = 255
            row_covered = row * h * col_steps
            col = floor((r_val - row_covered) / col_steps)
            memory[:col, row] = 255
            col_covered = col * col_steps
            last_pixel = floor(r_val - row_covered - col_covered) 
            memory[col, row] = last_pixel * (255 // col_steps)
            return memory
        
        level, hp, explore = self.group_rewards()
        full_memory = np.stack((
            make_reward_channel(level),
            make_reward_channel(hp),
            make_reward_channel(explore)
        ), axis=-1)
        
        if self.get_badges() > 0:
            full_memory[:, -1, :] = 255

        return full_memory

    def create_recent_memory(self):
        return rearrange(
            self.recent_memory, 
            '(w h) c -> h w c', 
            h=self.memory_height)

    def check_if_done(self):
        if self.early_stopping:
            done = False
            if self.step_count > 128 and self.recent_memory.sum() < (255 * 1):
                done = True
        else:
            done = self.step_count >= self.max_steps
        #done = self.read_hp_fraction() == 0
        return done

    def save_and_print_info(self, done, obs_memory):
        if self.print_rewards:
            prog_string = f'step: {self.step_count:6d}'
            for key, val in self.progress_reward.items():
                prog_string += f' {key}: {val:5.2f}'
            prog_string += f' sum: {self.total_reward:5.2f}'
            print(f'\r{prog_string}', end='', flush=True)
        
        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f'curframe_{self.instance_id}.jpeg'), 
                self.render(reduce_res=False))

        if self.print_rewards and done:
            print('', flush=True)
            if self.save_final_state:
                fs_path = self.s_path / Path('final_states')
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_small.jpeg'), 
                    obs_memory)
                plt.imsave(
                    fs_path / Path(f'frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg'), 
                    self.render(reduce_res=False))

        if self.save_video and done:
            self.full_frame_writer.close()
            self.model_frame_writer.close()

        if done:
            self.all_runs.append(self.progress_reward)
            with open(self.s_path / Path(f'all_runs_{self.instance_id}.json'), 'w') as f:
                json.dump(self.all_runs, f)
            pd.DataFrame(self.agent_stats).to_csv(
                self.s_path / Path(f'agent_stats_{self.instance_id}.csv.gz'), compression='gzip', mode='a')
    
    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit-1] == '1'
    
    def get_levels_sum(self):
        poke_levels = [max(self.read_m(a) - 2, 0) for a in PARTY_LEVEL_ADDR]
        return max(sum(poke_levels) - 4, 0) # subtract starting pokemon level
    
    def get_levels_reward(self):
        explore_thresh = 22
        scale_factor = 4
        level_sum = self.get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum-explore_thresh) / scale_factor + explore_thresh
        self.max_level_rew = max(self.max_level_rew, scaled)
        return self.max_level_rew
    
    def get_knn_reward(self):
        
        pre_rew = self.explore_weight * 0.005
        post_rew = self.explore_weight * 0.01
        cur_size = self.knn_index.get_current_count() if self.use_screen_explore else len(self.seen_coords)
        base = (self.base_explore if self.levels_satisfied else cur_size) * pre_rew
        post = (cur_size if self.levels_satisfied else 0) * post_rew
        return base + post
    
    def get_badges(self):
        return self.bit_count(self.read_m(BADGE_1_ADDR))

    def read_party(self):
        return [self.read_m(addr) for addr in PARTY_ADDR]
    
    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        if (cur_health > self.last_health and
                self.read_m(PARTY_SIZE_ADDR) == self.party_size):
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                if heal_amount > 0.5:
                    print(f'healed: {heal_amount}')
                    self.save_screenshot('healing')
                self.total_healing_rew += heal_amount * 4
            else:
                self.died_count += 1
                
    def get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        event_flags_start = EVENT_FLAGS_START_ADDR
        event_flags_end = EVENT_FLAGS_END_ADDR
        museum_ticket = (MUSEUM_TICKET_ADDR, 0)
        base_event_flags = 13
        return max(
            sum(
                [
                    self.bit_count(self.read_m(i))
                    for i in range(event_flags_start, event_flags_end)
                ]
            )
            - base_event_flags
            - int(self.read_bit(museum_ticket[0], museum_ticket[1])),
        0,
    )

    def get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        '''
        num_poke = self.read_m(PARTY_SIZE_ADDR)
        poke_xps = [self.read_triple(a) for a in POKE_XP_ADDR]
        #money = self.read_money() - 975 # subtract starting money
        seen_poke_count = sum([self.bit_count(self.read_m(i)) for i in SEEN_POKE_ADDR])
        all_events_score = sum([self.bit_count(self.read_m(i)) for i in range(EVENT_FLAGS_START_ADDR, EVENT_FLAGS_END_ADDR)])
        oak_parcel = self.read_bit(OAK_PARCEL_ADDR, 1) 
        oak_pokedex = self.read_bit(OAK_POKEDEX_ADDR, 5)
        opponent_level = self.read_m(OPPONENT_LEVEL) # What is this?
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        enemy_poke_count = self.read_m(ENEMY_POKE_COUNT)
        self.max_opponent_poke = max(self.max_opponent_poke, enemy_poke_count)
        
        if print_stats:
            print(f'num_poke : {num_poke}')
            print(f'poke_levels : {poke_levels}')
            print(f'poke_xps : {poke_xps}')
            #print(f'money: {money}')
            print(f'seen_poke_count : {seen_poke_count}')
            print(f'oak_parcel: {oak_parcel} oak_pokedex: {oak_pokedex} all_events_score: {all_events_score}')
        '''
        
        state_scores = {
            'event': self.reward_scale*self.update_max_event_rew(),  
            #'party_xp': self.reward_scale*0.1*sum(poke_xps),
            'level': self.reward_scale*self.get_levels_reward(), 
            'heal': self.reward_scale*self.total_healing_rew,
            'op_lvl': self.reward_scale*self.update_max_op_level(),
            'dead': self.reward_scale*-0.1*self.died_count,
            'badge': self.reward_scale*self.get_badges() * 5,
            #'op_poke': self.reward_scale*self.max_opponent_poke * 800,
            #'money': self.reward_scale* money * 3,
            #'seen_poke': self.reward_scale * seen_poke_count * 400,
            'explore': self.reward_scale * self.get_knn_reward()
        }
        
        return state_scores
    
    def save_screenshot(self, name):
        return # disable for now
        ss_dir = self.s_path / Path('screenshots')
        ss_dir.mkdir(exist_ok=True)
        plt.imsave(
            ss_dir / Path(f'frame{self.instance_id}_r{self.total_reward:.4f}_{self.reset_count}_{name}.jpeg'), 
            self.render(reduce_res=False))
    
    def update_max_op_level(self):
        opponent_level = max([self.read_m(a) for a in OPPONENT_LEVEL_ADDR])
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level * 0.2
    
    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def read_hp_fraction(self):
        hp_sum = sum([self.read_hp(add) for add in HP_ADDR])
        max_hp_sum = sum([self.read_hp(add) for add in MAX_HP_ADDR])
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start+1)

    # built-in since python 3.10
    def bit_count(self, bits):
        return bin(bits).count('1')

    def read_triple(self, start_add):
        return 256*256*self.read_m(start_add) + 256*self.read_m(start_add+1) + self.read_m(start_add+2)
    
    def read_bcd(self, num):
        return 10 * ((num >> 4) & 0x0f) + (num & 0x0f)
    
    def read_money(self):
        return (100 * 100 * self.read_bcd(self.read_m(MONEY_ADDR_1)) + 
                100 * self.read_bcd(self.read_m(MONEY_ADDR_100)) +
                self.read_bcd(self.read_m(MONEY_ADDR_10000)))

