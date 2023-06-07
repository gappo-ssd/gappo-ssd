import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np


class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class CellEntity(Enum):
    # entity encodings for grid observations
    OUT_OF_BOUNDS = 0
    EMPTY = 1
    FOOD = 2
    AGENT = 3


class Player:
    def __init__(self):
        self.player_id = None
        self.controller = None
        self.position = None
        self.level = None
        self.field_size = None
        self.score = None
        self.reward = 0
        self.history = None
        self.current_step = None
        self.last_action = None

    def setup(self, player_id, position, level, field_size):
        self.player_id = player_id
        self.history = []
        self.position = position
        self.level = level
        self.field_size = field_size
        self.score = 0

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"


class ForagingEnv(Env):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata = {"render.modes": ["human"]}

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["player_id", "position", "level", "history", "reward", "is_self"]
    )  # reward is available only if is_self

    def seed(self, seed=None):
        # set Python random seed
        np.random.seed(seed)
        # set Pytorch seed
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count
        """
        if not self._grid_observation:
            field_x = self.field.shape[1]
            field_y = self.field.shape[0]
            # field_size = field_x * field_y

            max_food = self.max_food
            max_food_level = self.max_player_level * len(self.players)

            min_obs = [-1, -1, 0] * max_food + [-1, -1, 0] * len(self.players)
            max_obs = [field_x - 1, field_y - 1, max_food_level] * max_food + [
                field_x - 1,
                field_y - 1,
                len(self.players),
            ] * len(self.players)
        else:
            # todo: observation_space definition
            # grid observation space
            grid_shape = (1 + 2 * self._old_global_sight, 1 + 2 * self._old_global_sight)

            # agents layer: agent levels
            agents_min = np.zeros(grid_shape, dtype=np.float32)
            agents_max = np.ones(grid_shape, dtype=np.float32) * self.max_player_level

            # foods layer: foods level
            max_food_level = self.max_player_level * len(self.players)
            foods_min = np.zeros(grid_shape, dtype=np.float32)
            foods_max = np.ones(grid_shape, dtype=np.float32) * max_food_level

            # access layer: i the cell available
            access_min = np.zeros(grid_shape, dtype=np.float32)
            access_max = np.ones(grid_shape, dtype=np.float32)

            # total layer
            min_obs = np.stack([agents_min, foods_min, access_min]).flatten()
            max_obs = np.stack([agents_max, foods_max, access_max]).flatten()

        return gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.float32)

    def __init__(
            self,
            players,
            max_player_level,
            field_size,
            food,
            sight,
            max_episode_steps,
            force_coop,
            food_respawn_interval,
            energy_penalty,
            seed=None,
            food_respawn=True,
            normalize_reward=False,
            grid_observation=False,
            penalty=0,
    ):
        if seed:
            self.seed(seed)
        self.logger = logging.getLogger(__name__)
        self.players = [Player() for _ in range(players)]
        self.field = np.zeros(field_size, np.int32)
        self.penalty = penalty
        self.max_food = sum(list(map(lambda f: f[0], food.values())))
        self.food = food
        self.max_food_lvl = int(max(list(food.keys()), key=lambda x: int(x)))
        self.food_respawn = food_respawn
        self.to_respawn = dict()
        self.food_respawn_interval = food_respawn_interval
        self._food_spawned = 0.0
        self.max_player_level = max_player_level
        self.sight = sight
        self._old_global_sight = self.sight
        self.force_coop = force_coop
        self._game_over = None

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        self._normalize_reward = normalize_reward
        self._grid_observation = grid_observation

        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(6)] * len(self.players)))
        self.observation_space = gym.spaces.Tuple(tuple([self._get_observation_space()] * len(self.players)))

        self.viewer = None

        self.n_agents = len(self.players)
        self.energy_penalty = energy_penalty

    @classmethod
    def from_obs(cls, obs):
        players = []
        for p in obs.players:
            player = Player()
            player.setup(0, p.position, p.level, obs.field.shape)
            player.score = p.score if p.score else 0
            players.append(player)

        env = cls(players, None, None, None, None)
        env.field = np.copy(obs.field)
        env.current_step = obs.current_step
        env.sight = obs.sight
        env._gen_valid_moves()

        return env

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                   max(row - distance, 0): min(row + distance + 1, self.rows),
                   max(col - distance, 0): min(col + distance + 1, self.cols),
                   ]

        return (
                self.field[
                max(row - distance, 0): min(row + distance + 1, self.rows), col
                ].sum()
                + self.field[
                  row, max(col - distance, 0): min(col + distance + 1, self.cols)
                  ].sum()
        )

    def adjacent_food(self, row, col):
        return (
                self.field[max(row - 1, 0), col]
                + self.field[min(row + 1, self.rows - 1), col]
                + self.field[row, max(col - 1, 0)]
                + self.field[row, min(col + 1, self.cols - 1)]
        )

    def adjacent_food_location(self, row, col):
        if row > 1 and self.field[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.field[row + 1, col] > 0:
            return row + 1, col
        elif col > 1 and self.field[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.field[row, col + 1] > 0:
            return row, col + 1

    def adjacent_players(self, row, col):
        return [
            player
            for player in self.players
            if abs(player.position[0] - row) == 1
               and player.position[1] == col
               or abs(player.position[1] - col) == 1
               and player.position[0] == row
        ]

    def spawn_food(self, food, max_level):
        min_level = max_level if self.force_coop else 1

        for food_level, f in food.items():
            food_amount = f[0]
            attempts = 0
            food_count = 0
            while food_count < food_amount and attempts < 1000:
                attempts += 1
                row = self.np_random.integers(1, self.rows - 1)
                col = self.np_random.integers(1, self.cols - 1)

                # check if it has neighbors:
                if (
                        self.neighborhood(row, col).sum() > 0
                        or self.neighborhood(row, col, distance=2, ignore_diag=True) > 0
                        or not self._is_empty_location(row, col)
                ):
                    continue

                self.field[row, col] = food_level
                food_count += 1
        self._food_spawned = self.field.sum()

    def respawn_food(self, food_lvl):
        attempts = 0

        while attempts < 1000:
            attempts += 1
            row = self.np_random.integers(1, self.rows - 1)
            col = self.np_random.integers(1, self.cols - 1)

            # check if it has neighbors:
            if (
                    self.neighborhood(row, col).sum() > 0
                    or self.neighborhood(row, col, distance=2, ignore_diag=True) > 0
                    or not self._is_empty_location(row, col)
            ):
                continue

            self.field[row, col] = food_lvl
            break
        self._food_spawned = self.field.sum()

    def _is_empty_location(self, row, col):
        if self.field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True

    def spawn_players(self, max_player_level):
        p_id = 0
        for player in self.players:

            attempts = 0
            player.reward = 0

            while attempts < 1000:
                row = self.np_random.integers(0, self.rows)
                col = self.np_random.integers(0, self.cols)
                if self._is_empty_location(row, col):
                    player.setup(
                        p_id,
                        (row, col),
                        self.np_random.integers(1, max_player_level + 1),
                        self.field_size,
                    )
                    p_id += 1
                    break
                attempts += 1

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                    player.position[0] > 0
                    and self.field[player.position[0] - 1, player.position[1]] == 0
            )
        elif action == Action.SOUTH:
            return (
                    player.position[0] < self.rows - 1
                    and self.field[player.position[0] + 1, player.position[1]] == 0
            )
        elif action == Action.WEST:
            return (
                    player.position[1] > 0
                    and self.field[player.position[0], player.position[1] - 1] == 0
            )
        elif action == Action.EAST:
            return (
                    player.position[1] < self.cols - 1
                    and self.field[player.position[0], player.position[1] + 1] == 0
            )
        elif action == Action.LOAD:
            return self.adjacent_food(*player.position) > 0

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        return (
            position[0] + (sight - center[0]),
            position[1] + (sight - center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player):
        food_in_sight = []
        for f in list(zip(*np.nonzero(self.field))):
            if 0 <= abs(player.position[0] - f[0]) <= self._old_global_sight and 0 <= abs(player.position[1] - f[1]) <= self._old_global_sight:
                # local position
                # local_position = self._transform_to_neighborhood(player.position, player.sight, f)
                # regular position
                position = f
                food_lvl = self.field[f[0]][f[1]]
                food_in_sight.append([position[0], position[1], food_lvl])
        return self.Observation(
            actions=self._valid_actions[player],
            players=[
                self.PlayerObservation(
                    player_id=a.player_id,
                    # local position
                    # position=self._transform_to_neighborhood(player.position, player.sight, a.position),
                    # regular position
                    position=a.position,
                    level=a.level,
                    is_self=a == player,
                    history=a.history,
                    reward=a.reward if a == player else None,
                )
                for a in self.players
                if 0 <= abs(player.position[0] - a.position[0]) <= self._old_global_sight and
                   0 <= abs(player.position[1] - a.position[1]) <= self._old_global_sight
            ],
            field=food_in_sight,
            game_over=self.game_over,
            sight=self._old_global_sight,
            current_step=self.current_step,
        )

    def _make_gym_obs(self, eaten_foods=None):
        def make_obs_array(observation):
            obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
            # obs[: observation.field.size] = observation.field.flatten()
            # self player is always first
            seen_players = [p for p in observation.players if p.is_self] + [
                p for p in observation.players if not p.is_self
            ]

            for i in range(self.max_food):
                obs[3 * i] = -1
                obs[3 * i + 1] = -1
                obs[3 * i + 2] = 0

            field_x, field_y = self.field_size

            for i, (y, x, l) in enumerate(observation.field):
                obs[3 * i] = y
                obs[3 * i + 1] = x
                obs[3 * i + 2] = l

            for i in range(len(self.players)):
                obs[self.max_food * 3 + 3 * i] = -1
                obs[self.max_food * 3 + 3 * i + 1] = -1
                obs[self.max_food * 3 + 3 * i + 2] = 0

            for i, p in enumerate(seen_players):
                obs[self.max_food * 3 + 3 * i] = p.position[0]
                obs[self.max_food * 3 + 3 * i + 1] = p.position[1]
                obs[self.max_food * 3 + 3 * i + 2] = p.player_id

            return obs

        def make_global_grid_arrays():
            """
            Create global arrays for grid observation space
            """
            # todo: creation of grids
            grid_shape_x, grid_shape_y = self.field_size
            grid_shape_x += 2 * self._old_global_sight
            grid_shape_y += 2 * self._old_global_sight
            grid_shape = (grid_shape_x, grid_shape_y)

            # agents layer
            agents_layer = np.ones(grid_shape, dtype=np.float32) * -1
            for player in self.players:
                player_x, player_y = player.position
                agents_layer[player_x + self._old_global_sight, player_y + self._old_global_sight] = player.player_id

            # food layer
            foods_layer = np.ones(grid_shape, dtype=np.float32) * -1
            field_copy = self.field.copy()
            field_copy[field_copy == 0] = -1
            foods_layer[self._old_global_sight:-self._old_global_sight, self._old_global_sight:-self._old_global_sight] = field_copy

            # access layer
            access_layer = np.ones(grid_shape, dtype=np.float32) * -1
            # out of bounds not accessible
            access_layer[:self._old_global_sight, :] = 0.0
            access_layer[-self._old_global_sight:, :] = 0.0
            access_layer[:, :self._old_global_sight] = 0.0
            access_layer[:, -self._old_global_sight:] = 0.0
            # agent locations are not accessible
            for player in self.players:
                player_x, player_y = player.position
                access_layer[player_x + self._old_global_sight, player_y + self._old_global_sight] = 0.0
            # food locations are not accessible
            foods_x, foods_y = self.field.nonzero()
            for x, y in zip(foods_x, foods_y):
                access_layer[x + self._old_global_sight, y + self._old_global_sight] = 0.0
            return np.stack([agents_layer, foods_layer, access_layer])

        def get_agent_grid_bounds(agent_x, agent_y):
            return agent_x, agent_x + 2 * self._old_global_sight + 1, agent_y, agent_y + 2 * self._old_global_sight + 1
        def get_player_reward(observation):
            for p in observation.players:
                if p.is_self:
                    return p.reward

        observations = [self._make_obs(player) for player in self.players]
        if self._grid_observation:
            layers = make_global_grid_arrays()
            agents_bounds = [get_agent_grid_bounds(*player.position) for player in self.players]
            nobs = np.array([layers[:, start_x:end_x, start_y:end_y] for start_x, end_x, start_y, end_y in agents_bounds])
        else:
            nobs = tuple([make_obs_array(obs) for obs in observations])
        nreward = [get_player_reward(obs) for obs in observations]
        ndone = [obs.game_over for obs in observations]
        # ninfo = [{'observation': obs} for obs in observations]
        ninfo = {} if eaten_foods is None else {"eaten_foods": eaten_foods}
        


        flat_nobs = []
        for i in range(self.n_agents):
            flat_nobs.append(nobs[i].flatten())
        flat_nobs = np.array(flat_nobs)

        player_positions = [p.position for p in self.players]

        return flat_nobs, player_positions, nreward, ndone, ninfo

    def reset(self):
        self.field = np.zeros(self.field_size, np.int32)
        self.spawn_players(self.max_player_level)
        player_levels = sorted([player.level for player in self.players])

        self.spawn_food(
            self.food, max_level=sum(player_levels[:3])
        )
        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()

        nobs, player_positions, _, _, _ = self._make_gym_obs()
        return nobs, player_positions

    def step(self, actions):
        # save received last actions
        for i in range(self.n_agents):
            self.players[i].last_action = actions[i]
        self.current_step += 1

        # keep track of type of food eaten
        eaten_food = dict()
        for lvl in self.food.keys():
            eaten_food[lvl] = 0

        for p in self.players:
            p.reward = self.energy_penalty

        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        # check if actions are valid
        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                actions[i] = Action.NONE

        loading_players = set()

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)
            elif action == Action.LOAD:
                collisions[player.position].append(player)
                loading_players.add(player)

        # and do movements for non colliding players

        # check for collisions if players are colliding they are staying in place, check for collisions with players
        # that might be willing to move to their old position
        all_collisions = collisions.copy()
        for k, v in collisions.items():
            if len(v) > 1:  # make sure no more than an player will arrive at location
                for i in range(len(v)):
                    all_collisions[(v[i].position[0], v[i].position[1])].append(v[i])
                continue
        # now move the players if they are not colliding
        for k, v in all_collisions.items():
            if len(v) > 1:
                continue
            v[0].position = k

        # finally process the loadings:
        while loading_players:
            # find adjacent food
            player = loading_players.pop()
            frow, fcol = self.adjacent_food_location(*player.position)
            food = self.field[frow, fcol]

            adj_players = self.adjacent_players(frow, fcol)
            adj_players = [
                p for p in adj_players if p in loading_players or p is player
            ]

            adj_player_level = sum([a.level for a in adj_players])

            loading_players = loading_players - set(adj_players)

            if adj_player_level < food:
                # failed to load
                for a in adj_players:
                    a.reward -= self.penalty
                continue

            # else the food was loaded and each player scores points
            for a in adj_players:
                # divide the reward of the food by the amount of loading players to introduce competition for food
                a.reward = self.food[str(food)][1] / len(adj_players)
                if self._normalize_reward:
                    a.reward = a.reward / float(
                        adj_player_level * self._food_spawned
                    )  # normalize reward
            # and the food is removed
            self.field[frow, fcol] = 0

            if self.food_respawn:
                # when food is eaten the food will respawn after 'food_respawn_interval' steps
                respawn_t = self.current_step + self.food_respawn_interval
                if respawn_t in self.to_respawn:
                    self.to_respawn[respawn_t].append(food)
                else:
                    self.to_respawn[respawn_t] = []
                    self.to_respawn[respawn_t].append(food)

            eaten_food[str(food)] += 1

        if self.food_respawn:
            # respawn food that needs to respawn in this level
            if self.current_step in self.to_respawn:
                food_lvls = self.to_respawn[self.current_step]
                for lvl in food_lvls:
                    self.respawn_food(lvl)
                del self.to_respawn[self.current_step]

        if self.food_respawn:
            self._game_over = (
                    self._max_episode_steps <= self.current_step
            )
        else:
            self._game_over = (
                    self.field.sum() == 0 or self._max_episode_steps <= self.current_step
            )
        self._gen_valid_moves()

        for p in self.players:
            p.score += p.reward

        return self._make_gym_obs(eaten_food)

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
