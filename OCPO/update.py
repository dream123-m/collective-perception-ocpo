"""OCPO overlap factors and team-advantage policy update.

The training runner in major/cross_scene/run.py provides environment rollout,
value estimation, optimization, and checkpoint management. The operations here
are the parts that distinguish OCPO from the MAPPO comparison method.
"""

import numpy as np
import torch


def groups_for(env, rng):
    """Partition valid decisions into same-target pairs and singletons."""
    members = {}
    slots = env.max_targets_per_cav
    for agent in range(env.num_agents):
        for slot in np.flatnonzero(env.mask[agent]):
            target = int(env.slot_targets[agent, slot])
            members.setdefault(target, []).append(agent * slots + int(slot))
    groups = []
    for target in sorted(members):
        order = rng.permutation(members[target])
        groups.extend((int(order[k]), int(order[k + 1]) if k + 1 < len(order) else -1)
                      for k in range(0, len(order), 2))
    result = torch.tensor(groups, dtype=torch.long)
    expected = np.flatnonzero(env.mask.ravel())
    if not np.array_equal(np.sort(result[result >= 0].numpy()), expected):
        raise AssertionError('Overlap factors do not partition all valid decisions.')
    return result


def factor_logp(bits, groups):
    """Sum Bernoulli log probabilities within each overlap factor."""
    flat = bits.flatten(-2)
    first = flat.gather(-1, groups[..., 0])
    second = flat.gather(-1, groups[..., 1].clamp_min(0))
    return first + torch.where(groups[..., 1] >= 0, second, 0.)


def clipped_actor_loss(ratio, team_advantage, clip):
    """Apply one team advantage to each OCPO factor's clipped ratio."""
    advantage = team_advantage[:, None]
    return -torch.minimum(ratio * advantage,
                          ratio.clamp(1 - clip, 1 + clip) * advantage).mean(0).sum()
