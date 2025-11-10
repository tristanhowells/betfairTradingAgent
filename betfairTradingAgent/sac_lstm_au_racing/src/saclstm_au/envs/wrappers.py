
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

class FlattenObs(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
    def observation(self, observation):
        return observation

class FrameStack1D(ObservationWrapper):
    def __init__(self, env, k: int = 4):
        super().__init__(env)
        self.k = k
        low = np.repeat(env.observation_space.low, k, axis=0)
        high = np.repeat(env.observation_space.high, k, axis=0)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        self.frames = None
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs.copy() for _ in range(self.k)]
        return self._get_obs(), info
    def observation(self, observation):
        self.frames.pop(0)
        self.frames.append(observation)
        return self._get_obs()
    def _get_obs(self):
        return np.concatenate(self.frames, axis=0)
