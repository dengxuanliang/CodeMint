from __future__ import annotations

from codemint.config import SynthesizeConfig
from codemint.models.weakness import WeaknessEntry, WeaknessReport


def allocate_specs(report: WeaknessReport, config: SynthesizeConfig) -> dict[str, int]:
    root_tags = {chain.root for chain in report.causal_chains}
    allocations: dict[str, int] = {}

    for weakness in select_top_weaknesses(report.weaknesses, config.top_n):
        key = weakness_key(weakness)
        base = config.specs_per_weakness
        allocated = min(
            base
            + (2 if key in root_tags else 0)
            + (1 if weakness.trainability >= 0.8 else 0),
            config.max_per_weakness,
        )
        allocations[key] = allocated

    return allocations


def weakness_key(weakness: WeaknessEntry) -> str:
    if weakness.sub_tags:
        return weakness.sub_tags[0]
    return f"{weakness.fault_type}_{weakness.rank}"


def select_top_weaknesses(weaknesses: list[WeaknessEntry], top_n: int) -> list[WeaknessEntry]:
    if top_n <= 0:
        return []

    selected: list[WeaknessEntry] = []
    seen_keys: set[str] = set()
    for weakness in sorted(weaknesses, key=lambda item: item.rank):
        key = weakness_key(weakness)
        if key in seen_keys:
            continue
        selected.append(weakness)
        seen_keys.add(key)
        if len(selected) >= top_n:
            break
    return selected
