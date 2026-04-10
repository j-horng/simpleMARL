"""Project-local reward functions for PyQuaticus training."""

import numpy as np

from pyquaticus.utils.rewards import caps_and_grabs

# Per-step dense shaping: home agents get a small reward for each opponent currently on our half.
DENSE_HOME_DEFENSE_PER_INVADER = 0.0025
DENSE_HOME_DEFENSE_MAX_INVADERS = 3


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
    n_inv = _num_invaders_on_my_side(state, my_team, scrimmage_coords, fh, opp_inds)
    if (
        n_inv > 0
        and state["agent_on_sides"][agent_idx]
        and not state["agent_has_flag"][agent_idx]
    ):
        capped = min(n_inv, DENSE_HOME_DEFENSE_MAX_INVADERS)
        reward += DENSE_HOME_DEFENSE_PER_INVADER * capped

    # If your flag is taken, nudge non-carriers home to defend. Skip agents who are
    # carrying the opponent's flag — they must stay off-side until they can score.
    if state["flag_taken"][my_team] and not state["agent_has_flag"][agent_idx]:
        if state["agent_on_sides"][agent_idx]:
            reward += 0.02
        else:
            reward -= 0.02

    return reward
