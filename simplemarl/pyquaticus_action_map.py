"""
NRL-recommended discrete relative headings for PyQuaticus training/eval.

PyQuaticus expects ACTION_MAP entries as [speed_fraction, relative_heading_deg]
(relative heading = PID turn command, not compass bearing).
"""

import pyquaticus.config as _pq_config


def nrl_recommended_headings():
    """[-100, -60, -10..10, 60, 100] in degrees."""
    return [-100, -60, *range(-10, 11), 60, 100]


def nrl_action_map_entries():
    """Full-speed steering at each heading plus no-op."""
    out = [[1.0, float(h)] for h in nrl_recommended_headings()]
    out.append([0.0, 0.0])
    return out


def apply_action_map_entries(entries):
    """Patch installed pyquaticus.config.ACTION_MAP in place (same list object importers keep)."""
    _pq_config.ACTION_MAP.clear()
    _pq_config.ACTION_MAP.extend(entries)


def apply_nrl_action_map():
    apply_action_map_entries(nrl_action_map_entries())
