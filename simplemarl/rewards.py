"""Project-local reward functions for PyQuaticus training."""

from typing import Optional

import numpy as np

from pyquaticus.utils.rewards import caps_and_grabs

# ---------------------------------------------------------------------------
# Balancing offense vs defense (same reward for every agent; roles are emergent).
#
# Home-defense bonuses only apply when `agent_on_sides` and not carrying the
# enemy flag, so the same policy still gets offensive signal when pushing or
# returning a flag from caps_and_grabs.
#
# Sparse outcomes there (typical): own grab +0.25, own capture +1.0, mirrored
# negatives for enemy scores; plus distance-to-flag shaping when not tagged /
# not carrying.
#
# Keep defensive shaping smaller in expectation than scoring over an episode,
# or everyone camps. Rule of thumb:
#   - Lower DEFENSE_SHAPING_SCALE (e.g. 0.5–0.7) if agents rarely press the flag.
#   - Raise CHASE_AWAY_PER_INVADER slightly if clears are rare but offense is fine.
#   - Enemy grab / cap penalties should still dominate “farming” chase-away.
#
# DEFENSE_SHAPING_SCALE scales dense home + chase-away + flag-taken home nudge.
# ---------------------------------------------------------------------------
DEFENSE_SHAPING_SCALE = 1.0

# Per-step dense shaping: home agents get a small reward for each opponent currently on our half.
DENSE_HOME_DEFENSE_PER_INVADER = 0.0025
DENSE_HOME_DEFENSE_MAX_INVADERS = 3

# Bonus when an opponent was on our side last step and is no longer there (pushed back / retreated /
# tagged off), but not if they still carry a flag (crossing midline with the flag is a successful raid).
CHASE_AWAY_PER_INVADER = 0.10

# While our flag is gone, nudge non-carriers toward home (scaled with defense shaping).
FLAG_TAKEN_HOME_NUDGE = 0.02


def _point_on_team_half_plane(pos, side_team_int: int, scrimmage_coords, flag_home) -> bool:
    """
    True if pos lies on the same side of the scrimmage line as `side_team_int`'s flag
    (aligned with PyQuaticus `_check_on_sides`, including points on the line).
    """
    scrimmage_vec = np.asarray(scrimmage_coords[1], dtype=np.float64) - np.asarray(
        scrimmage_coords[0], dtype=np.float64
    )
    scrim0 = np.asarray(scrimmage_coords[0], dtype=np.float64)
    p = np.asarray(pos, dtype=np.float64).reshape(-1)[:2]
    scrim2pos = p - scrim0
    cross = scrimmage_vec[0] * scrim2pos[1] - scrimmage_vec[1] * scrim2pos[0]
    cp_sign = np.sign(cross)
    fh = np.asarray(flag_home[side_team_int], dtype=np.float64).reshape(-1)[:2]
    scrim2flag = fh - scrim0
    team_cross = scrimmage_vec[0] * scrim2flag[1] - scrimmage_vec[1] * scrim2flag[0]
    team_sign = np.sign(team_cross)
    return bool((cp_sign == team_sign) | (cp_sign == 0))


def _opponent_agent_indices(agent_inds_of_team: dict, my_team_int: int):
    for t, inds in agent_inds_of_team.items():
        if int(t) != my_team_int:
            return {int(i) for i in np.asarray(inds).ravel()}
    return set()


def _num_invaders_on_my_side(state, my_team: int, scrimmage_coords, flag_home, opp_inds: set) -> int:
    n = 0
    for j in opp_inds:
        if _point_on_team_half_plane(state["agent_position"][j], my_team, scrimmage_coords, flag_home):
            n += 1
    return n


def _count_invaders_left_our_half(
    prev_state: dict, state: dict, my_team: int, scrimmage_coords, flag_home, opp_inds: set
) -> int:
    """Opponents who were on our half-plane last step but are not now, excluding flag carriers."""
    n = 0
    for j in opp_inds:
        prev_on = _point_on_team_half_plane(
            prev_state["agent_position"][j], my_team, scrimmage_coords, flag_home
        )
        now_on = _point_on_team_half_plane(
            state["agent_position"][j], my_team, scrimmage_coords, flag_home
        )
        if not (prev_on and not now_on):
            continue
        if bool(state["agent_has_flag"][j]):
            continue
        n += 1
    return n


def _eligible_home_defender(state: dict, agent_idx: int) -> bool:
    return bool(state["agent_on_sides"][agent_idx] and not state["agent_has_flag"][agent_idx])


def _dense_home_and_chase_away_bonus(
    state: dict,
    prev_state: Optional[dict],
    my_team: int,
    scrimmage_coords,
    fh,
    opp_inds: set,
    agent_idx: int,
) -> float:
    bonus = 0.0
    n_inv = _num_invaders_on_my_side(state, my_team, scrimmage_coords, fh, opp_inds)
    if n_inv > 0 and _eligible_home_defender(state, agent_idx):
        capped = min(n_inv, DENSE_HOME_DEFENSE_MAX_INVADERS)
        bonus += DENSE_HOME_DEFENSE_PER_INVADER * capped

    if prev_state is None:
        return DEFENSE_SHAPING_SCALE * bonus
    cap_delta = np.asarray(state["captures"], dtype=np.int64) - np.asarray(
        prev_state["captures"], dtype=np.int64
    )
    if np.any(cap_delta > 0):
        return bonus
    left = _count_invaders_left_our_half(
        prev_state, state, my_team, scrimmage_coords, fh, opp_inds
    )
    if left > 0 and _eligible_home_defender(state, agent_idx):
        bonus += CHASE_AWAY_PER_INVADER * left
    return DEFENSE_SHAPING_SCALE * bonus


def balanced_caps_and_grabs(
    agent_id: str,
    team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size,
    agent_radius,
    catch_radius: float,
    scrimmage_coords,
    max_speeds: list,
    tagging_cooldown: float,
):
    """
    Start with caps_and_grabs and add defensive pressure:
    - Stronger team-level penalty when the opponent grabs/captures.
    - Small positive signal when your team tags opponents.
    - Extra credit for personally tagging an opponent carrying your flag.
    - Mild positioning signal: stay home while your flag is threatened.
    - Dense shaping: small per-step reward while on home turf and opponents are on our half.
    - Chase-away: bonus when an opponent leaves our half without carrying a flag (defensive pressure).
    """
    reward = caps_and_grabs(
        agent_id,
        team,
        agents,
        agent_inds_of_team,
        state,
        prev_state,
        env_size,
        agent_radius,
        catch_radius,
        scrimmage_coords,
        max_speeds,
        tagging_cooldown,
    )

    my_team = int(team)
    other_team = 1 - my_team
    agent_idx = agents.index(agent_id)

    # Team-level defensive pressure: do not allow free enemy progress.
    enemy_grab_delta = state["grabs"][other_team] - prev_state["grabs"][other_team]
    if enemy_grab_delta > 0:
        reward -= 0.50 * enemy_grab_delta

    enemy_cap_delta = state["captures"][other_team] - prev_state["captures"][other_team]
    if enemy_cap_delta > 0:
        reward -= 1.50 * enemy_cap_delta

    # Reward team tags slightly to make disruption behavior more attractive.
    my_tag_delta = state["tags"][my_team] - prev_state["tags"][my_team]
    if my_tag_delta > 0:
        reward += 0.15 * my_tag_delta

    # Per-agent bonus if this agent tags an enemy flag carrier.
    tagged_idx = state["agent_made_tag"][agent_idx]
    if tagged_idx is not None and prev_state["agent_has_flag"][tagged_idx]:
        reward += 0.35

    fh = state["flag_home"]
    opp_inds = _opponent_agent_indices(agent_inds_of_team, my_team)
    reward += _dense_home_and_chase_away_bonus(
        state, prev_state, my_team, scrimmage_coords, fh, opp_inds, agent_idx
    )

    # If your flag is taken, nudge non-carriers home to defend. Skip agents who are
    # carrying the opponent's flag — they must stay off-side until they can score.
    if state["flag_taken"][my_team] and not state["agent_has_flag"][agent_idx]:
        nudge = FLAG_TAKEN_HOME_NUDGE * DEFENSE_SHAPING_SCALE
        if state["agent_on_sides"][agent_idx]:
            reward += nudge
        else:
            reward -= nudge

    return reward
