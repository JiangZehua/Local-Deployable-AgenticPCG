"""Render inference prompts and expose the fixed domain tool inventories."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from toolusepcg.config import Config, load_config
from toolusepcg.level import Level
from toolusepcg.optimizer import GreedyOptimizer
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOMAIN_CONFIG: dict[str, str] = {
    "binary": "configs/binary.yaml" if (_PROJECT_ROOT / "configs/binary.yaml").exists()
              else "configs/default.yaml",  # binary uses default.yaml in this repo
    "binarydoor": "configs/binarydoor.yaml",
    "zelda": "configs/zelda.yaml",
    "sokoban": "configs/sokoban.yaml",
    "loderunner": "configs/loderunner.yaml",
    "smb": "configs/smb.yaml",
}


FULL_TOOL_INVENTORY: dict[str, list[str]] = {
    "binary": [
        "place_tile",
        "place_wall_segment",
        "generate_random",
        "generate_maze",
        "generate_bsp",
        "generate_ca",
        "generate_connect",
        "generate_digger",
        "generate_wfc",
    ],
    "binarydoor": [
        "place_tile",
        "place_wall_segment",
        "generate_random",
        "generate_maze",
        "generate_bsp",
        "generate_ca",
        "generate_connect",
        "generate_digger",
        "generate_wfc",
    ],
    "zelda": [
        "place_tile",
        "generate_zelda_tile_placing",
        "generate_zelda_random",
        "generate_zelda_maze",
        "generate_zelda_bsp",
        "generate_zelda_ca",
        "generate_zelda_connect",
        "generate_zelda_digger",
        "generate_zelda_wfc",
    ],
    "sokoban": ["place_tile"],
    "loderunner": ["place_tile"],
    "smb": ["place_tile"],
}


class HarnessPromptBuilder:
    """Thin wrapper that drives ``GreedyOptimizer.build_prompt`` as a pure
    prompt renderer.

    One instance holds one initialized optimizer per domain. Creating an
    instance is mildly expensive (loads config, instantiates evaluator +
    scorer + registry, creates a throwaway initial level). ``render_prompt``
    is cheap: set level → evaluate → build prompt.
    """

    def __init__(self, config: Config, problem_type: str | None = None):
        self.config = config
        self._opt = GreedyOptimizer(config, verbose=False)
        self._opt.initialize(seed=getattr(config, "seed", None) or 42)
        self.problem_type = self._opt.problem_type

    @classmethod
    @lru_cache(maxsize=len(DOMAIN_CONFIG))
    def for_domain(cls, domain: str) -> "HarnessPromptBuilder":
        """Cached per-domain builder. Safe to call many times per process.

        Loads the base domain YAML and overrides ``optimizer.edit_tools``
        with :data:`FULL_TOOL_INVENTORY` so the registry matches the tools
        the teacher traces actually used.
        """
        if domain not in DOMAIN_CONFIG:
            raise KeyError(f"unknown domain: {domain!r}; known: {sorted(DOMAIN_CONFIG)}")
        cfg_path = _PROJECT_ROOT / DOMAIN_CONFIG[domain]
        overrides: dict[str, Any] = {}
        if domain in FULL_TOOL_INVENTORY:
            overrides = {"optimizer": {"edit_tools": FULL_TOOL_INVENTORY[domain]}}
        cfg = load_config(str(cfg_path), overrides=overrides)
        return cls(cfg)

    def render_prompt(self, level_str: str) -> str:
        """Render the harness prompt for a given level string.

        The level is loaded via ``Level.from_string`` and installed into the
        optimizer's edit manager. The evaluator + scorer are run once to
        populate the metrics / score blocks of the prompt.
        """
        if not level_str:
            raise ValueError("level_str is empty")
        level = Level.from_string(level_str, problem_type=self.problem_type)
        self._opt.edit_manager.set_level(level)
        self._opt.current_result = self._opt.evaluator.evaluate(self._opt.current_level)
        self._opt.current_score = self._opt.scorer.score(self._opt.current_result)
        return self._opt.build_prompt()


