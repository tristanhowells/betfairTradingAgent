
import math
import numpy as np
from gymnasium import Env, spaces

class AUPreOffEnvV7TradeCost(Env):
    """Minimal, self-contained pre-off trading env stub."""
    metadata = {"render_modes": []}

    def __init__(self,
                 n_runners: int = 10,
                 episode_seconds: int = 120,
                 tick: float = 0.01,
                 commission: float = 0.05,
                 per_market_loss_cap_bps: float = 150.0,
                 reject_unaffordable: bool = True,
                 rng_seed: int = 0):
        super().__init__()
        self.rng = np.random.default_rng(rng_seed)
        self.n_runners = n_runners
        self.episode_seconds = episode_seconds
        self.tick = tick
        self.commission = commission
        self.loss_cap_frac = per_market_loss_cap_bps / 1e4
        self.reject_unaffordable = reject_unaffordable

        # latent price state
        self._p = np.log(self.rng.uniform(1.5, 10.0, size=n_runners))

        # observation space
        self._obs_per_runner = 4
        self._obs_agg = 4
        self._obs_dim = self.n_runners * self._obs_per_runner + self._obs_agg
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)

        # ---- Option A: Single Box action (runner, template, size) ----
        # a[0] -> runner index (continuous) mapped to [0, n_runners-1] then floored to int
        # a[1] -> template (0: enter_back, 1: enter_lay, 2: exit_back, 3: exit_lay)
        # a[2] -> size in [0, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([float(self.n_runners - 1), 3.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.t = None
        self.positions = np.zeros((self.n_runners,), dtype=np.float32)
        self.cash = 0.0
        self.entry_prices = np.zeros((self.n_runners,), dtype=np.float32) + np.nan
        self._build_ladder()

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def _build_ladder(self):
        prices = np.exp(self._p)
        spreads = self.rng.uniform(self.tick, 3 * self.tick, size=self.n_runners)
        best_back = np.maximum(1.01, prices - spreads)
        best_lay = prices + spreads
        sizes_back = self.rng.uniform(50, 500, size=self.n_runners)
        sizes_lay = self.rng.uniform(50, 500, size=self.n_runners)
        self._ladder = {
            "best_back": best_back,
            "best_lay": best_lay,
            "size_back": sizes_back,
            "size_lay": sizes_lay,
        }

    def _step_prices(self):
        drift = -0.002
        noise = self.rng.normal(0, 0.02, size=self.n_runners)
        self._p = self._p + drift + noise
        self._build_ladder()

    def _market_aggregates(self):
        inv_back = 1.0 / np.maximum(1.01, self._ladder["best_back"])
        inv_lay = 1.0 / np.maximum(1.01, self._ladder["best_lay"])
        back_over = float(inv_back.sum())
        lay_over = float(inv_lay.sum())
        spread_ticks_mean = float(np.mean((self._ladder["best_lay"] - self._ladder["best_back"]) / self.tick))
        return spread_ticks_mean, back_over, lay_over

    def _obs(self):
        ob = []
        for i in range(self.n_runners):
            ob.extend([
                self._ladder["best_back"][i], self._ladder["size_back"][i],
                self._ladder["best_lay"][i], self._ladder["size_lay"][i],
            ])
        spread_ticks_mean, back_over, lay_over = self._market_aggregates()
        t_norm = 1.0 - (self.t / self.episode_seconds)
        ob.extend([spread_ticks_mean, back_over, lay_over, t_norm])
        return np.asarray(ob, dtype=np.float32)

    def _mtm_pnl(self):
        pnl = 0.0
        for i in range(self.n_runners):
            pos = self.positions[i]
            if pos > 0:
                pnl += pos * (self._ladder["best_lay"][i] - self.entry_prices[i])
            elif pos < 0:
                pnl += (-pos) * (self.entry_prices[i] - self._ladder["best_back"][i])
        pnl -= abs(pnl) * self.commission
        return pnl

    def _worst_roi(self):
        bankroll_unit = 100.0
        per_runner = []
        for i in range(self.n_runners):
            pos = self.positions[i]
            if pos > 0:
                pnl = pos * (self._ladder["best_lay"][i] - self.entry_prices[i])
            elif pos < 0:
                pnl = (-pos) * (self.entry_prices[i] - self._ladder["best_back"][i])
            else:
                pnl = 0.0
            per_runner.append(pnl / bankroll_unit)
        return float(min(per_runner)) if per_runner else 0.0

    def _all_green(self):
        for i in range(self.n_runners):
            pos = self.positions[i]
            if pos > 0 and self._ladder["best_lay"][i] < self.entry_prices[i]:
                return False
            if pos < 0 and self._ladder["best_back"][i] > self.entry_prices[i]:
                return False
        return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        self.t = 0
        self.positions[:] = 0.0
        self.cash = 0.0
        self.entry_prices[:] = np.nan
        self._p = np.log(self.rng.uniform(1.5, 10.0, size=self.n_runners))
        self._build_ladder()
        obs = self._obs()
        info = {}
        return obs, info

    def step(self, action):
        # Map continuous action -> discrete runner/template + continuous size
        runner_cont = float(action[0])
        template_cont = float(action[1])
        size = float(np.clip(action[2], 0.0, 1.0))

        ridx = int(np.clip(np.floor(runner_cont + 1e-6), 0, self.n_runners - 1))
        template = int(np.clip(np.floor(template_cont + 1e-6), 0, 3))

        orders_submitted = 0
        fills_step = 0
        penalty = 0.0

        best_back = self._ladder["best_back"][ridx]
        best_lay = self._ladder["best_lay"][ridx]

        new_pos = self.positions[ridx]
        if template == 0:        # enter_back
            new_pos += size
            entry_price = best_back
        elif template == 1:      # enter_lay
            new_pos -= size
            entry_price = best_lay
        elif template == 2:      # exit_back
            new_pos = max(0.0, new_pos - size)
            entry_price = best_lay
        else:                    # exit_lay
            new_pos = min(0.0, new_pos + size)
            entry_price = best_back

        orders_submitted += 1
        if abs(new_pos) > 1.0 and self.reject_unaffordable:
            penalty -= 0.0005
        else:
            self.positions[ridx] = new_pos
            if math.isnan(self.entry_prices[ridx]):
                self.entry_prices[ridx] = entry_price
            else:
                self.entry_prices[ridx] = 0.5 * self.entry_prices[ridx] + 0.5 * entry_price
            fills_step += 1

        self._step_prices()
        self.t += 1
        terminated = self.t >= self.episode_seconds

        mtm = self._mtm_pnl()
        worst_roi = self._worst_roi()
        reward = 0.3 * (mtm / 100.0) - 0.8 * max(0.0, -worst_roi) + penalty

        if worst_roi < -self.loss_cap_frac:
            terminated = True

        obs = self._obs()
        info = {
            "orders_submitted_step": orders_submitted,
            "fills_step": fills_step,
            "worst_roi_step": worst_roi,
            "green_all_step": self._all_green(),
        }
        truncated = False
        return obs, reward, terminated, truncated, info
