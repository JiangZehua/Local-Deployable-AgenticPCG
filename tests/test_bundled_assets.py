"""Runtime assets must remain usable after relocation and packaging."""
from pathlib import Path

from PIL import Image
import pytest

from experiments.smoke import run as run_smoke
from toolusepcg.config import load_config
from toolusepcg.evaluator import StatsEvaluator
from toolusepcg.factory import LevelFactory
from toolusepcg.level import Level
from toolusepcg.logging.trace import TraceLogger
from toolusepcg.pattern_metrics import smb_reference_array, smb_reference_distribution


ROOT = Path(__file__).resolve().parents[1]


def test_smb_reference_distribution_outside_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    smb_reference_array.cache_clear()
    smb_reference_distribution.cache_clear()
    reference = smb_reference_array()
    assert reference.ndim == 2 and reference.size > 0
    distribution = smb_reference_distribution()
    assert distribution and sum(distribution.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("domain", ["zelda", "smb"])
def test_simulation_animation_outside_checkout(domain, tmp_path, monkeypatch):
    output = tmp_path / "levels"
    monkeypatch.chdir(ROOT)
    run_smoke(output)
    cfg = load_config(ROOT / "configs" / f"{domain}.yaml")
    factory = LevelFactory(cfg)
    level = Level.from_string((output / f"{domain}.txt").read_text(), domain)
    evaluated = StatsEvaluator(cfg, factory.env).evaluate(level)
    logger = TraceLogger(cfg, run_dir=tmp_path / "trace", render_sim=True)
    logger.set_env(factory.env)

    monkeypatch.chdir(tmp_path)
    render = (logger.render_zelda_simulation_gif if domain == "zelda"
              else logger.render_simulation_gif)
    animation = render(level.data, evaluated.info, max_frames=4)
    assert animation is not None and animation.is_file()
    with Image.open(animation) as image:
        assert image.format == "GIF"
        assert image.n_frames > 1
