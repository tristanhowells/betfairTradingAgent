from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from gymnasium import Env, spaces


# ----------------------------- Utilities -----------------------------

def _nan_to_num(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, copy=False, posinf=0.0, neginf=0.0)


def _safe_float(x) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return 0.0


# ----------------------------- Replay Dataset (v5_3b wide) -----------------------------

_L1_PATTERNS = [
    r"^(?P<field>bbp|bbs|blp|bls)[_\-]*L?1[_\-]*r(?P<runner>\d+)$",
    r"^(?P<field>bbp|bbs|blp|bls)[_\-]*r(?P<runner>\d+)[_\-]*L?1$",
    r"^r(?P<runner>\d+)[_\-]*(?P<field>bbp|bbs|blp|bls)[_\-]*L?1$",
    r"^(?P<field>bbp|bbs|blp|bls)[_\-]*L?1[_\-]*(?P<runner>\d+)$",
]


def _match_l1_map(columns: List[str]) -> Dict[Tuple[str, int], str]:
    """
    Build a mapping: (field, runner_index) -> column_name for L1 best prices/sizes.
    field in {bbp,bbs,blp,bls}, runner_index is 0-based (normalized later).
    """
    out: Dict[Tuple[str, int], str] = {}
    for col in columns:
        c = col.strip()
        for pat in _L1_PATTERNS:
            m = re.match(pat, c, flags=re.IGNORECASE)
            if m:
                field = m.group("field").lower()
                ridx = int(m.group("runner"))
                out[(field, ridx)] = col
                break
    return out


@dataclass
class MarketEpisode:
    df: pd.DataFrame
    market_id: str
    scheduled_off_ts: Optional[float]


class V53bWideReplay:
    """
    Reader for v5_3b 'wide' parquet files.

    Required columns:
      - 'ts_unix'           (seconds since epoch)
      - 'file_market_id'    (string id per market)
      - L1 columns for each runner: (bbp|bbs|blp|bls) with an 'r<idx>' suffix/prefix
    """

    def __init__(self, root: str | Path, file_glob: str = "**/*.parquet", forward_fill: bool = True):
        self.root = Path(root)
        self.file_glob = file_glob
        self.forward_fill = forward_fill
        self.files: List[Path] = sorted(self.root.glob(self.file_glob))
        if not self.files:
            raise FileNotFoundError(f"No parquet files found under {self.root} with glob '{self.file_glob}'")

    def _normalize_runner_indexing(self, colmap: Dict[Tuple[str, int], str]) -> Dict[Tuple[str, int], str]:
        raw = sorted({r for (_, r) in colmap.keys()})
        if not raw:
            return {}
        # Collapse to contiguous 0..N-1 preserving order
        mapping = {r: i for i, r in enumerate(raw)}
        norm: Dict[Tuple[str, int], str] = {}
        for (field, r), col in colmap.items():
            norm[(field, mapping[r])] = col
        return norm

    def _build_frame(self, df: pd.DataFrame) -> MarketEpisode:
        cols = list(df.columns)
        ts_col = "ts_unix" if "ts_unix" in cols else None
        if ts_col is None:
            raise KeyError("Expected 'ts_unix' in replay parquet.")

        mid_col = "file_market_id" if "file_market_id" in cols else None
        if mid_col is None:
            raise KeyError("Expected 'file_market_id' in replay parquet.")

        sched_col = None
        for cand in ["scheduled_off_unix", "scheduled_off_ts", "scheduled_off", "market_start_ts_unix"]:
            if cand in cols:
                sched_col = cand
                break

        l1_map_raw = _match_l1_map(cols)
        if not l1_map_raw:
            raise KeyError("Could not find any L1 columns matching (bbp|bbs|blp|bls).*r<idx> in replay parquet.")
        l1_map = self._normalize_runner_indexing(l1_map_raw)

        use = {ts_col, mid_col}
        if sched_col:
            use.add(sched_col)
        use.update(l1_map.values())
        df2 = df[list(use)].copy()
        df2 = df2.sort_values(ts_col).reset_index(drop=True)

        if self.forward_fill:
            df2[list(l1_map.values())] = df2[list(l1_map.values())].ffill()
        for c in l1_map.values():
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
        df2[list(l1_map.values())] = df2[list(l1_map.values())].ffill().fillna(0.0)

        market_id = str(df2[mid_col].iloc[0])
        scheduled_off_ts = float(df2[sched_col].iloc[0]) if sched_col is not None and pd.notnull(df2[sched_col].iloc[0]) else None

        df2.attrs["ts_col"] = ts_col
        df2.attrs["sched_col"] = sched_col
        df2.attrs["l1_map"] = l1_map
        df2.attrs["market_id"] = market_id

        return MarketEpisode(df=df2, market_id=market_id, scheduled_off_ts=scheduled_off_ts)

    def sample_episode(self) -> MarketEpisode:
        fn = random.choice(self.files)
        df = pd.read_parquet(fn)
        return self._build_frame(df)


# ----------------------------- Environment -----------------------------

class AUPreOffEnvV9Replay(Env):
    """
    v5_3b replay env with rich info for logging/metrics.
    Observation (per step):
      - per-runner L1: bbp, bbs, blp, bls (padded/truncated to n_runners)
      - aggregates: spread_ticks_mean, back_over, lay_over, over_diff, tto_norm

    Action: Box(3,) -> [runner_cont, template_cont, size]
      runner_cont ∈ [0, n_runners-1], template_cont ∈ [0,3], size ∈ [0,1].
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_root: str,
        file_glob: str = "**/*.parquet",
        n_runners: int = 12,
        tick: float = 0.01,
        commission: float = 0.05,
        per_market_loss_cap_bps: float = 150.0,
        reject_unaffordable: bool = True,
        forward_fill: bool = True,
        tto_max_seconds: int = 900,
        rng_seed: int = 42,
    ):
        super().__init__()
        self.ds = V53bWideReplay(data_root, file_glob=file_glob, forward_fill=forward_fill)
        self.rng = np.random.default_rng(rng_seed)

        self.n_runners = int(n_runners)
        self.tick = float(tick)
        self.commission = float(commission)
        self.loss_cap_frac = float(per_market_loss_cap_bps) / 1e4
        self.reject_unaffordable = bool(reject_unaffordable)
        self.tto_max_seconds = float(tto_max_seconds)

        # obs = per-runner (4) + aggregates(5)
        self._obs_per_runner = 4  # bbp,bbs,blp,bls
        self._obs_agg = 5
        self._obs_dim = self.n_runners * self._obs_per_runner + self._obs_agg
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([float(self.n_runners - 1), 3.0, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )

        # episode state
        self.df: Optional[pd.DataFrame] = None
        self.ts_col: Optional[str] = None
        self.sched_col: Optional[str] = None
        self.l1_map: Optional[Dict[Tuple[str, int], str]] = None
        self.market_id: Optional[str] = None
        self.t = 0

        # positions & bookkeeping
        self.positions = None               # (n_runners,)
        self.entry_prices = None            # (n_runners,)
        self.cash = 0.0

    # -------------------- Helpers for obs/features --------------------

    def _slice_row_L1(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.df is not None and self.l1_map is not None
        row = self.df.iloc[idx]
        max_ridx = max(r for (_, r) in self.l1_map.keys())
        R = max(self.n_runners, max_ridx + 1)

        bbp = np.zeros((R,), dtype=np.float32)
        bbs = np.zeros((R,), dtype=np.float32)
        blp = np.zeros((R,), dtype=np.float32)
        bls = np.zeros((R,), dtype=np.float32)

        for (field, r0), col in self.l1_map.items():
            if 0 <= r0 < R:
                v = _safe_float(row[col])
                if field == "bbp":
                    bbp[r0] = v
                elif field == "bbs":
                    bbs[r0] = v
                elif field == "blp":
                    blp[r0] = v
                elif field == "bls":
                    bls[r0] = v

        def pad_trunc(x: np.ndarray) -> np.ndarray:
            if len(x) >= self.n_runners:
                return x[: self.n_runners]
            if len(x) == 0:
                return np.zeros((self.n_runners,), dtype=np.float32)
            pad = np.zeros((self.n_runners - len(x),), dtype=np.float32)
            return np.concatenate([x, pad], axis=0)

        return pad_trunc(bbp), pad_trunc(bbs), pad_trunc(blp), pad_trunc(bls)

    def _aggregates(self, bbp: np.ndarray, bbs: np.ndarray, blp: np.ndarray, bls: np.ndarray) -> Tuple[float, float, float, float]:
        spread_ticks = (blp - bbp) / max(self.tick, 1e-6)
        spread_ticks_mean = float(np.mean(_nan_to_num(spread_ticks)))
        inv_back = 1.0 / np.maximum(1.01, _nan_to_num(bbp))
        inv_lay = 1.0 / np.maximum(1.01, _nan_to_num(blp))
        back_over = float(np.sum(inv_back))
        lay_over = float(np.sum(inv_lay))
        over_diff = lay_over - back_over
        return spread_ticks_mean, back_over, lay_over, over_diff

    def _tto_seconds(self, idx: int) -> float:
        if self.sched_col is not None and self.sched_col in self.df.columns:
            scheduled = self.df[self.sched_col].iloc[idx]
            ts = self.df[self.ts_col].iloc[idx]
            if pd.notnull(scheduled) and pd.notnull(ts):
                return max(0.0, float(scheduled) - float(ts))
        remain = max(0, (len(self.df) - 1 - idx))
        return float(remain)

    def _obs_from_row(self, idx: int) -> np.ndarray:
        bbp, bbs, blp, bls = self._slice_row_L1(idx)
        spread_ticks_mean, back_over, lay_over, over_diff = self._aggregates(bbp, bbs, blp, bls)
        tto_s = self._tto_seconds(idx)
        tto_norm = float(np.clip(tto_s / max(1.0, self.tto_max_seconds), 0.0, 1.0))

        obs = np.concatenate(
            [bbp, bbs, blp, bls, np.asarray([spread_ticks_mean, back_over, lay_over, over_diff, tto_norm], dtype=np.float32)],
            axis=0,
        ).astype(np.float32)

        return _nan_to_num(obs)

    # -------------------- Finance / PnL --------------------

    def _mtm_pnl(self, bbp: np.ndarray, blp: np.ndarray) -> float:
        pnl = 0.0
        for i in range(self.n_runners):
            pos = self.positions[i]
            if pos > 0:
                pnl += pos * (blp[i] - self.entry_prices[i])
            elif pos < 0:
                pnl += (-pos) * (self.entry_prices[i] - bbp[i])
        pnl -= abs(pnl) * self.commission
        return float(pnl)

    def _worst_roi(self, bbp: np.ndarray, blp: np.ndarray) -> float:
        bankroll_unit = 100.0
        per_runner = []
        for i in range(self.n_runners):
            pos = self.positions[i]
            if pos > 0:
                pnl = pos * (blp[i] - self.entry_prices[i])
            elif pos < 0:
                pnl = (-pos) * (self.entry_prices[i] - bbp[i])
            else:
                pnl = 0.0
            per_runner.append(pnl / bankroll_unit)
        return float(min(per_runner)) if per_runner else 0.0

    def _all_green(self, bbp: np.ndarray, blp: np.ndarray) -> bool:
        for i in range(self.n_runners):
            pos = self.positions[i]
            if pos > 0 and blp[i] < self.entry_prices[i]:
                return False
            if pos < 0 and bbp[i] > self.entry_prices[i]:
                return False
        return True

    # -------------------- Gym API --------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        ep = self.ds.sample_episode()
        self.df = ep.df
        self.market_id = ep.market_id
        self.ts_col = self.df.attrs["ts_col"]
        self.sched_col = self.df.attrs["sched_col"]
        self.l1_map = self.df.attrs["l1_map"]

        self.t = 0
        self.positions = np.zeros((self.n_runners,), dtype=np.float32)
        self.entry_prices = np.zeros((self.n_runners,), dtype=np.float32) + np.nan
        self.cash = 0.0

        obs = self._obs_from_row(self.t)
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        assert self.df is not None
        ridx_cont = float(action[0])
        template_cont = float(action[1])
        size = float(np.clip(action[2], 0.0, 1.0))

        ridx = int(np.clip(np.floor(ridx_cont + 1e-6), 0, self.n_runners - 1))
        template = int(np.clip(np.floor(template_cont + 1e-6), 0, 3))

        bbp, bbs, blp, bls = self._slice_row_L1(self.t)

        orders_submitted = 0
        fills_step = 0
        penalty = 0.0

        best_back = bbp[ridx]
        best_lay = blp[ridx]
        new_pos = self.positions[ridx]

        # 0: enter long(back), 1: enter short(lay), 2: exit long (lay), 3: exit short (back)
        if template == 0:
            new_pos += size
            entry_price = best_back
        elif template == 1:
            new_pos -= size
            entry_price = best_lay
        elif template == 2:
            new_pos = max(0.0, new_pos - size)
            entry_price = best_lay
        else:
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

        # advance to next row
        self.t += 1
        terminated = self.t >= (len(self.df) - 1)

        # next snapshot for reward/obs
        next_idx = min(self.t, len(self.df) - 1)
        bbp2, bbs2, blp2, bls2 = self._slice_row_L1(next_idx)

        mtm = self._mtm_pnl(bbp2, blp2)
        worst_roi = self._worst_roi(bbp2, blp2)
        spread_ticks_mean, back_over, lay_over, over_diff = self._aggregates(bbp2, bbs2, blp2, bls2)
        tto_seconds = self._tto_seconds(next_idx)

        reward = 0.3 * (mtm / 100.0) - 0.8 * max(0.0, -worst_roi) + penalty

        bankrupt = False
        if worst_roi < -self.loss_cap_frac:
            terminated = True
            bankrupt = True

        obs = np.zeros(self.observation_space.shape, dtype=np.float32) if terminated else self._obs_from_row(self.t)
        obs = _nan_to_num(obs)

        liq_back = float(np.sum(bbs2))
        liq_lay = float(np.sum(bls2))

        # exposures
        exposure_by_runner = self.positions.copy().astype(np.float32)
        exposure_total = float(np.sum(np.abs(exposure_by_runner)))

        # mark “green at off” only at terminal step
        green_at_off_step = float(self._all_green(bbp2, blp2)) if terminated else np.nan

        info: Dict[str, float | int | bool | np.ndarray] = {
            # core microstructure
            "avg_spread_ticks_step": spread_ticks_mean,
            "back_over_step": back_over,
            "lay_over_step": lay_over,
            "over_diff_step": over_diff,
            "tto_seconds_step": tto_seconds,
            "liq_back_step": liq_back,
            "liq_lay_step": liq_lay,

            # finance
            "mtm_pnl_step": mtm,

            # actions & risk
            "orders_submitted_step": orders_submitted,
            "fills_step": fills_step,
            "worst_roi_step": worst_roi,
            "green_all_step": bool(self._all_green(bbp2, blp2)),
            "bankrupt_step": bankrupt,
            "loss_cap_hit_step": bankrupt,
            "runner_index": ridx,
            "template_id": template,
            "action_size": size,
            "action_clipped": bool(size <= 0.0 or size >= 1.0),

            # terminal-only metrics
            "green_at_off_step": green_at_off_step,

            # exposures
            "exposure_total": exposure_total,
            "exposure_by_runner": exposure_by_runner,  # array; callback reduces it
        }

        truncated = False
        return obs, float(reward), bool(terminated), bool(truncated), info
