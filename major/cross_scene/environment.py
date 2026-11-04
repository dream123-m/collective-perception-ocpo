"""Vehicle perception and broadcast simulation."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
import numpy as np
from scipy.io import loadmat
DEFAULT_CONFIG: dict[str, Any] = {
    "target_coverage": 0.95,
    "coverage_reward_sigma": 0.04,
    "episode_steps": 8,
    "roi_distance": 150.0,
    "perception_distance": 150.0,
    "mean_power": 100.0,
    "noise_power": 1.0233,
    "m": 1.0,
    "dis": 30.0,
    "interference_distance": 300.0,
    "path_loss_exp": 2.0,
    "sinr_threshold": 0.1,
    "w1": 0.2,
    "w2": 0.6,
    "w3": 0.2,
    "penalty_exponent": 1.4,
    "w5": 6.0,
    "road_length": 1000.0,
    "road_width": 15.0,
    "required_vehicle_count": 200,
    "required_cav_count": 100,
}
def _seed(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("A seed must be a nonnegative integer, not a Boolean.")
    result = int(value)
    if result != value or result < 0 or result >= 2**63:
        raise ValueError("Seeds must be integers in [0, 2**63).")
    return result
def _configuration(overrides: dict[str, Any] | None) -> dict[str, Any]:
    supplied = dict(overrides or {})
    if "comm_distance" in supplied:
        legacy_dis = supplied.pop("comm_distance")
        if "dis" in supplied and supplied["dis"] != legacy_dis:
            raise ValueError("dis and comm_distance disagree.")
        supplied["dis"] = legacy_dis
    if "horizon" in supplied:
        if "episode_steps" in supplied and supplied["episode_steps"] != supplied["horizon"]:
            raise ValueError("horizon and episode_steps disagree.")
        supplied["episode_steps"] = supplied.pop("horizon")
    weights = supplied.pop("reward_weights", {})
    if not isinstance(weights, dict):
        raise ValueError("reward_weights must be a dictionary.")
    for name, value in weights.items():
        destination = "penalty_exponent" if name == "w4" else name
        if destination not in ("w1", "w2", "w3", "penalty_exponent", "w5"):
            raise ValueError(f"Unknown reward weight: {name}")
        if destination in supplied and supplied[destination] != value:
            raise ValueError(f"Conflicting values for {destination}.")
        supplied[destination] = value
    unknown = sorted(set(supplied) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(f"Unknown environment configuration keys: {unknown}")
    result = dict(DEFAULT_CONFIG)
    result.update(supplied)
    integers = {"episode_steps", "required_vehicle_count", "required_cav_count"}
    for name in result:
        value = result[name]
        if not np.isscalar(value) or not np.isfinite(value):
            raise ValueError(f"Configuration {name} must be finite and numeric.")
        if name in integers:
            if isinstance(value, (bool, np.bool_)) or int(value) != value or value <= 0:
                raise ValueError(f"Configuration {name} must be a positive integer.")
            result[name] = int(value)
        else:
            result[name] = float(value)
    if not 0.0 <= result["target_coverage"] <= 1.0:
        raise ValueError("target_coverage must be in [0, 1].")
    for name in (
        "coverage_reward_sigma", "roi_distance", "perception_distance", "mean_power",
        "noise_power", "m", "dis", "interference_distance", "path_loss_exp",
        "sinr_threshold", "penalty_exponent", "road_length", "road_width",
    ):
        if result[name] <= 0:
            raise ValueError(f"Configuration {name} must be positive.")
    if result["m"] < 0.5:
        raise ValueError("The Nakagami parameter m must be at least 0.5.")
    for name in ("w1", "w2", "w3", "w5"):
        if result[name] < 0:
            raise ValueError(f"Configuration {name} must be nonnegative.")
    if result["dis"] > result["interference_distance"]:
        raise ValueError("Communication range cannot exceed interference range.")
    if result["required_cav_count"] > result["required_vehicle_count"]:
        raise ValueError("CAV count cannot exceed total vehicle count.")
    return result
class CrossSceneEnv:
    def __init__(self, mat_path: str | Path, seed: int, config: dict[str, Any] | None = None):
        self.config = _configuration(config)
        self.mat_path = Path(mat_path).resolve()
        self.initial_seed = _seed(seed)
        self._episode_rng = np.random.default_rng(self.initial_seed)
        self._load_scene()
        self._build_local_slots()
        self.reset(seed=self.initial_seed)
    def _load_scene(self) -> None:
        data = loadmat(self.mat_path)
        required = {"positions", "is_cav", "vehicle_ids", "los_matrix", "comm_los_matrix"}
        missing = sorted(required - data.keys())
        if missing:
            raise ValueError(f"Scene is missing required MAT fields: {missing}")
        positions = np.asarray(data["positions"], dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 2 or not np.isfinite(positions).all():
            raise ValueError("positions must be a finite [vehicles, 2] array.")
        count = len(positions)
        if count != self.config["required_vehicle_count"]:
            raise ValueError(f"Expected {self.config['required_vehicle_count']} vehicles; found {count}.")
        if count < 2:
            raise ValueError("A scene must contain at least two vehicles.")
        ids = np.asarray(data["vehicle_ids"]).reshape(-1)
        labels = np.asarray(data["is_cav"]).reshape(-1)
        if ids.shape != (count,) or not np.issubdtype(ids.dtype, np.number):
            raise ValueError("vehicle_ids must contain one numeric ID per vehicle.")
        if not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all() or len(np.unique(ids)) != count:
            raise ValueError("vehicle_ids must be finite, unique integers.")
        if labels.shape != (count,) or not np.isin(labels, [0, 1]).all():
            raise ValueError("is_cav must contain one binary label per vehicle.")
        if int(labels.sum()) != self.config["required_cav_count"]:
            raise ValueError(f"Expected {self.config['required_cav_count']} CAVs; found {int(labels.sum())}.")
        order = np.argsort(ids, kind="stable")
        self.vehicle_ids = ids[order].astype(np.int64)
        self.positions = positions[order].copy()
        self.is_cav = labels[order].astype(bool)
        self.total_vehicles = count
        self.cav_indices = np.flatnonzero(self.is_cav)
        self.num_agents = len(self.cav_indices)
        self.cav_to_agent_idx = {int(c): i for i, c in enumerate(self.cav_indices)}
        self.distance_matrix = np.linalg.norm(
            self.positions[:, None, :] - self.positions[None, :, :], axis=-1,
        )
        if np.any(self.distance_matrix[np.triu_indices(count, k=1)] <= 0):
            raise ValueError("Distinct vehicle centers must not coincide.")
        for name in ("los_matrix", "comm_los_matrix"):
            matrix = np.asarray(data[name])
            if matrix.shape != (count, count) or not np.isin(matrix, [0, 1]).all():
                raise ValueError(f"{name} must be a binary [vehicles, vehicles] matrix.")
            matrix = matrix[np.ix_(order, order)].astype(bool)
            if not np.array_equal(matrix, matrix.T):
                raise ValueError(f"{name} must be symmetric for static geometric LOS.")
            setattr(self, name, matrix)
        self.perception_distance = self.config["perception_distance"]
        self.roi_distance = self.config["roi_distance"]
        expected_sensing = self.comm_los_matrix & (self.distance_matrix <= self.perception_distance)
        if not np.array_equal(self.los_matrix, expected_sensing):
            raise ValueError("los_matrix must equal full geometric comm_los_matrix restricted to sensing range.")
        self.sensing_matrix = expected_sensing.copy()
        np.fill_diagonal(self.sensing_matrix, False)
        self.max_targets_per_cav = count - 1
        self.state_dim = 3 + 5 * self.max_targets_per_cav
    def _build_local_slots(self) -> None:
        n, capacity = self.num_agents, self.max_targets_per_cav
        self.slot_targets = np.full((n, capacity), -1, dtype=np.int64)
        self.mask = np.zeros((n, capacity), dtype=bool)
        self.target_bool = self.sensing_matrix[self.cav_indices].copy()
        self.cav_targets: dict[int, list[int]] = {}
        self.cav_distances: dict[int, list[float]] = {}
        self.roi_neighbors: dict[int, list[int]] = {}
        self.co_perception = np.zeros((n, capacity), dtype=np.float64)
        self._has_roi_neighbors = np.zeros(n, dtype=bool)
        self._template = np.zeros((n, self.state_dim), dtype=np.float32)
        self._template[:, 0] = self.positions[self.cav_indices, 0] / self.config["road_length"]
        self._template[:, 1] = self.positions[self.cav_indices, 1] / self.config["road_width"]
        features = self._template[:, 3:].reshape(n, capacity, 5)
        for agent, cav in enumerate(self.cav_indices):
            candidates = np.flatnonzero(self.target_bool[agent])
            delta = self.positions[candidates] - self.positions[cav]
            ordering = np.lexsort((candidates, delta[:, 1], delta[:, 0], self.distance_matrix[cav, candidates]))
            targets = candidates[ordering]
            length = len(targets)
            self.slot_targets[agent, :length] = targets
            self.mask[agent, :length] = True
            self.cav_targets[int(cav)] = targets.tolist()
            distances = self.distance_matrix[cav, targets]
            self.cav_distances[int(cav)] = distances.tolist()
            neighbors = self.cav_indices[
                (self.cav_indices != cav)
                & (self.distance_matrix[cav, self.cav_indices] <= self.roi_distance)
            ]
            self.roi_neighbors[int(cav)] = neighbors.tolist()
            if len(neighbors):
                rows = [self.cav_to_agent_idx[int(neighbor)] for neighbor in neighbors]
                self.co_perception[agent, :length] = self.target_bool[rows][:, targets].mean(axis=0)
                self._has_roi_neighbors[agent] = True
            features[agent, :length, 0] = 1.0
            features[agent, :length, 1:3] = (
                self.positions[targets] - self.positions[cav]
            ) / self.perception_distance
            features[agent, :length, 3] = self.co_perception[agent, :length]
        self.target_universe = np.flatnonzero(self.target_bool.any(axis=0))
        self.total_targets = len(self.target_universe)
        if not self.total_targets:
            raise ValueError("The scene has no other-vehicle objects perceptible by any CAV.")
        self._distance_value = np.zeros((n, capacity), dtype=np.float64)
        for i, cav in enumerate(self.cav_indices):
            self._distance_value[i, self.mask[i]] = 1.0 - (
                self.distance_matrix[cav, self.slot_targets[i, self.mask[i]]] / self.perception_distance
            )
    def _sample_channel(self, seed: int) -> None:
        self.channel_seed = seed
        rng = np.random.default_rng(seed)
        c = self.cav_indices
        distance = self.distance_matrix[np.ix_(c, c)]
        geometric_los = self.comm_los_matrix[np.ix_(c, c)]
        attenuation = np.where(geometric_los, 1.0, 0.1)
        shape = self.config["m"]
        fading_power = rng.gamma(shape, self.config["mean_power"] * attenuation / shape)
        self.received_power = np.zeros((self.num_agents, self.num_agents), dtype=np.float64)
        off_diagonal = ~np.eye(self.num_agents, dtype=bool)
        self.received_power[off_diagonal] = fading_power[off_diagonal] / (
            distance[off_diagonal] ** self.config["path_loss_exp"]
        )
        interferer_power = np.where(
            off_diagonal & (distance <= self.config["interference_distance"]),
            self.received_power, 0.0,
        )
        self.interference_power = interferer_power.sum(axis=0, keepdims=True) - interferer_power
        denominator = self.config["noise_power"] + np.maximum(self.interference_power, 0.0)
        self.sinr = self.received_power / denominator
        self.decoding_links = (
            off_diagonal & (distance <= self.config["dis"])
            & (self.sinr >= self.config["sinr_threshold"])
        )
        self._success = self.decoding_links.any(axis=1)
        self.communication_success = {int(cav): bool(self._success[i]) for i, cav in enumerate(c)}
        self.best_sinr: dict[int, float] = {}
        self.best_link: dict[int, int | None] = {}
        for i, cav in enumerate(c):
            eligible = np.flatnonzero(off_diagonal[i] & (distance[i] <= self.config["dis"]))
            if len(eligible):
                receiver = eligible[np.argmax(self.sinr[i, eligible])]
                self.best_sinr[int(cav)] = float(self.sinr[i, receiver])
                self.best_link[int(cav)] = int(c[receiver])
            else:
                self.best_sinr[int(cav)] = 0.0
                self.best_link[int(cav)] = None
        if not np.isfinite(self.received_power).all() or not np.isfinite(self.sinr).all():
            raise FloatingPointError("The sampled channel contains non-finite power or SINR.")
        self.channel_hash = hashlib.sha256(
            np.ascontiguousarray(self.received_power, dtype="<f8").tobytes()
        ).hexdigest()
    def reset(self, seed: int | None = None) -> np.ndarray:
        channel_seed = (
            int(self._episode_rng.integers(0, 2**63, dtype=np.int64))
            if seed is None else _seed(seed)
        )
        self._sample_channel(channel_seed)
        self.current_step = 0
        self.previous_actions = np.zeros_like(self.mask)
        self.target_counts = np.zeros(self.total_vehicles, dtype=np.int64)
        self.target_sources: dict[int, list[int]] = {}
        self.covered_targets: set[int] = set()
        self.current_coverage_rate = 0.0
        self.episode_actions: dict[int, np.ndarray] = {}
        return self._observations()
    def _observations(self) -> np.ndarray:
        observations = self._template.copy()
        observations[:, 2] = self.previous_actions.sum(axis=1) / self.max_targets_per_cav
        observations[:, 3:].reshape(self.num_agents, self.max_targets_per_cav, 5)[:, :, 4] = self.previous_actions
        if not np.isfinite(observations).all():
            raise FloatingPointError("Observations must be finite.")
        return observations
    def get_action_mask(self, agent_idx: int) -> np.ndarray:
        if not 0 <= agent_idx < self.num_agents:
            raise IndexError("agent_idx is outside the scene's CAV roles.")
        return self.mask[agent_idx].copy()
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        if self.current_step >= self.config["episode_steps"]:
            raise RuntimeError("Episode has ended; call reset before stepping again.")
        values = np.asarray(actions)
        if values.shape != self.mask.shape:
            raise ValueError(f"actions must have shape {self.mask.shape}; found {values.shape}.")
        if values.dtype.kind not in "biuf" or not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
            raise ValueError("Actions must be finite binary values.")
        selected = values.astype(bool)
        if np.any(selected & ~self.mask):
            raise ValueError("Actions select padded or invalid target slots.")
        self.previous_actions = selected.copy()
        self.episode_actions = {
            int(cav): selected[i, :len(self.cav_targets[int(cav)])].copy()
            for i, cav in enumerate(self.cav_indices)
        }
        self.target_counts.fill(0)
        self.target_sources = {}
        for i in np.flatnonzero(self._success):
            targets = self.slot_targets[i, selected[i]]
            self.target_counts[targets] += 1
            for target in targets:
                self.target_sources.setdefault(int(target), []).append(int(self.cav_indices[i]))
        self.covered_targets = set(np.flatnonzero(self.target_counts).tolist())
        self.current_coverage_rate = len(self.covered_targets) / self.total_targets
        score = float(np.exp(
            -((self.current_coverage_rate - self.config["target_coverage"]) ** 2)
            / (2 * self.config["coverage_reward_sigma"] ** 2)
        ))
        counts = selected.sum(axis=1)
        distance_reward = (selected * self._distance_value).sum(axis=1)
        uniqueness_reward = (selected * (1.0 - self.co_perception)).sum(axis=1)
        uniqueness_reward[~self._has_roi_neighbors] = 0.0
        local = (
            self.config["w1"] * distance_reward
            + self.config["w2"] * uniqueness_reward
            - self.config["w3"] * counts ** self.config["penalty_exponent"]
        )
        rewards = np.where(self._success, local + self.config["w5"] * score, 0.0).astype(np.float32)
        if not np.isfinite(rewards).all():
            raise FloatingPointError("Rewards must be finite.")
        self.current_step += 1
        info = {
            "coverage_rate": float(self.current_coverage_rate),
            "coverage_reward": score,
            "target_coverage": self.config["target_coverage"],
            "total_targets": self.total_targets,
            "covered_targets": len(self.covered_targets),
            "payload": float(counts.mean()),
            "successful_senders": int(self._success.sum()),
            "channel_seed": self.channel_seed,
            "channel_hash": self.channel_hash,
        }
        return self._observations(), rewards, self.current_step >= self.config["episode_steps"], info
    def global_state(self) -> tuple[np.ndarray, float]:
        broadcasts = np.concatenate((
            (self.target_counts > 0).astype(np.float32),
            self.target_counts.astype(np.float32) / self.num_agents,
        ))
        return broadcasts, float(self.current_coverage_rate)
    def all_on_coverage(self) -> float:
        covered = self.target_bool[self._success].any(axis=0)
        return float(covered.sum() / self.total_targets)
