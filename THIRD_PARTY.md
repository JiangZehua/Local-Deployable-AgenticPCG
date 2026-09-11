# Third-party components

This project builds on the Agentic PCG framework and bundles the environment and generator code needed to run it independently.

| Included component | Source |
| --- | --- |
| `vendor/pcg_benchmark/` | [JiangZehua/pcg_benchmark](https://github.com/JiangZehua/pcg_benchmark) |
| `vendor/generators/` | [JiangZehua/ConstructiveGenerators](https://github.com/JiangZehua/ConstructiveGenerators) |

The benchmark subset includes the six base domains, tile-based SMB and Lode Runner, shared spaces, simulation engines and required rendering assets. The constructive generator package includes its reusable algorithms and WFC pattern images. Existing inline copyright and attribution notices remain in the source.

One unused top-level `turtle.width` import is removed from the binary maze generator so headless installations do not require Tk. The generation algorithm is unchanged. Local vendor-path registration replaces the original Git-submodule layout.

Game sprites and selected README illustrations retain their original provenance. Game names identify the benchmark environments; this project does not claim ownership of the underlying game assets. This attribution file does not grant a new license for third-party material.

Python dependencies are declared in `pyproject.toml`, with the tested client versions in `requirements/client-lock.txt`. The inference server dependencies are in `requirements/serve-qwen.txt`.
