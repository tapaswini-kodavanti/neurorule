#!/usr/bin/env python3
"""
Simple pretty-printer for RuleBased `.rules` JSON files produced by evolution runs.

Usage:
    python3 scripts/print_rules.py /path/to/file.rules

Optional:
    --states STATES_JSON   : JSON file mapping feature keys to readable names
    --actions ACTIONS_JSON : JSON file mapping action ids to names
    --decimals N           : number of decimals for coefficients (default 2)
"""
import argparse
import json
from pathlib import Path
from typing import Dict


def _format_side(coeff: float, key: str, lookback: int, exponent: int, states: Dict[str, str], decimals: int) -> str:
    name = states.get(key, f"feat_{key}")
    coeff_part = f"{coeff:.{decimals}f}*"
    part = coeff_part + name
    if lookback and lookback > 0:
        part = f"{part}[{lookback}]"
    if exponent and exponent > 1:
        part = f"{part}^{exponent}"
    return part


def _format_condition(cond: Dict, min_maxes: Dict[str, Dict], states: Dict[str, str], decimals: int) -> str:
    # First side
    first_coeff = cond.get("first_state_coefficient", 0.0)
    first_key = cond.get("first_state_key")
    first_lookback = cond.get("first_state_lookback", 0)
    first_exponent = cond.get("first_state_exponent", 1)
    first_part = _format_side(first_coeff, first_key, first_lookback, first_exponent, states, decimals)

    # Second side: try to render as absolute using min_maxes when possible
    operator = cond.get("operator", "?")
    second_key = cond.get("second_state_key")
    second_val = cond.get("second_state_value")

    # If second key maps to a named state, prefer that
    if second_key in states:
        second_part = states[second_key]
    else:
        # Try to compute an absolute value from min_maxes (use first key's min/max per original code)
        mm = None
        if first_key in min_maxes:
            mm = min_maxes.get(first_key)
        elif second_key in min_maxes:
            mm = min_maxes.get(second_key)

        if mm and second_val is not None:
            try:
                minv = float(mm.get("min", 0.0))
                maxv = float(mm.get("max", 1.0))
                absval = minv + float(second_val) * (maxv - minv)
                second_part = f"{absval:.{decimals}f} {{{minv} to {maxv}}}"
            except Exception:
                second_part = f"{second_val:.{decimals}f}"
        else:
            # Fall back to showing the normalized value
            if second_val is None:
                second_part = "<none>"
            else:
                second_part = f"{second_val:.{decimals}f}"

    return f"{first_part} {operator} {second_part}"


def pretty_print_rules(rules_json: Dict, states: Dict[str, str], actions: Dict[str, str], decimals: int) -> str:
    min_maxes = rules_json.get("min_maxes", {}) or {}
    rules = rules_json.get("rules", [])
    out_lines = []
    for r in rules:
        times = r.get("times_applied", 0)
        times_part = f"<{times}> " if times and times > 0 else " <> "
        conds = r.get("conditions", [])
        cond_strs = []
        for c in conds:
            cond_strs.append("(" + _format_condition(c, min_maxes, states, decimals) + ")")
        # Action name
        action = r.get("action")
        action_name = actions.get(action, f"action_{action}")
        coeff = r.get("action_coefficient", 0.0)
        if r.get("action_lookback", 0) > 0:
            action_part = f"Action[{r.get('action_lookback')}]"
        else:
            action_part = f"{coeff:.{decimals}f}*{action_name}"

        out_lines.append(times_part + " " + " ".join(cond_strs) + "\n  --> " + action_part)

    # Default action
    default_coeff = rules_json.get("default_action_coefficient", 0.0)
    default_act = rules_json.get("default_action")
    default_name = actions.get(default_act, f"action_{default_act}")
    out_lines.append(f"Default Action: {default_coeff:.{decimals}f}*{default_name}")

    return "\n".join(out_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rules_file", help="Path to a .rules JSON file")
    parser.add_argument("--states", help="Optional JSON file mapping feature keys to names", default=None)
    parser.add_argument("--actions", help="Optional JSON file mapping action ids to names", default=None)
    parser.add_argument("--decimals", help="Number of decimals for numeric output", default=2, type=int)
    args = parser.parse_args()

    p = Path(args.rules_file)
    if not p.exists():
        print(f"File not found: {p}")
        return

    with p.open() as fh:
        data = json.load(fh)

    states = {}
    actions = {}
    if args.states:
        try:
            with open(args.states) as fh:
                states = json.load(fh)
        except Exception as e:
            print(f"Failed to load states file '{args.states}': {e}")

    if args.actions:
        try:
            with open(args.actions) as fh:
                actions = json.load(fh)
        except Exception as e:
            print(f"Failed to load actions file '{args.actions}': {e}")

    pretty = pretty_print_rules(data, states, actions, args.decimals)
    print(pretty)


if __name__ == '__main__':
    main()
