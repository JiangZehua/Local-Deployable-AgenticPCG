"""One formal episode, with the original producer isolated by process."""
import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


def verify_loaded_studio_model(models, expected):
    """Validate the selected loaded GGUF, not LM Studio's available-model catalog."""
    selected = [m for m in models if m.get('identifier') == expected]
    indexed = 'qwen/qwen3.8-27b@lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf'
    if len(selected) != 1 or any(selected[0].get(k) != value for k, value in {
        'indexedModelIdentifier': indexed, 'format': 'gguf',
        'contextLength': 32768, 'parallel': 1,
    }.items()):
        raise RuntimeError('Loaded LM Studio model does not match the frozen GGUF/context/slot identity')
    return selected[0]['identifier']


def save_step_records(path, records):
    """Serialize evaluated trace values without changing tool requests or scores."""
    import numpy as np
    def numeric_default(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f'Unsupported trace value: {type(value).__name__}')
    payload = json.dumps([asdict(record) for record in records], ensure_ascii=False,
                         indent=2, default=numeric_default)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(payload+'\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cpu-only', action='store_true')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    job = json.loads(args.job.read_text())
    os.environ['RPE_PCG_ENABLE_THINKING'] = 'on' if job['suite'] == 'z64' else 'off'
    source = root
    os.chdir(source)
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('mac_transport', root/'experiments/lmstudio_transport.py')
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    import lmstudio as lms
    model = job.get('model_id', 'qwen38-mac-preflight')

    if job['suite'] == 'm6':
        from experiments import paper_m1_smb_language as producer
        if args.cpu_only:
            producer.structured_ceiling(args.job, args.output/'structured_ceiling.json',
                builder=job.get('smb_builder', 'legacy_r108'))
            return
        with lms.Client('127.0.0.1:1234') as client:
            def chat(_unused, *, model_id, messages, schema, schema_name, seed):
                response = transport.predict(client, model=model_id, messages=messages,
                    schema=schema, thinking=False, max_tokens=1024, seed=seed,
                    neutral_sampling=True, output=args.output/'call_01', timeout=900)
                stats = response['stats']
                if response['reasoning'] or stats['stopReason']=='maxPredictedTokensReached':
                    raise RuntimeError('Unexpected reasoning or truncated SMB parse')
                return json.loads(response['final']), {
                    'input_tokens': stats.get('promptTokensCount', 0),
                    'output_tokens': stats.get('predictedTokensCount', 0),
                    'latency_ms': round(response['elapsed_seconds']*1000)}
            producer._chat = chat
            producer.run_language(args.job, 'http://127.0.0.1:1234/v1', args.output/'result.json',
                model_id=model, revision=job['model_sha256'], seed=job['seed'],
                builder=job.get('smb_builder', 'legacy_r108'))
        return

    if job['suite'] == 'm7':
        # Match the original seeded entrypoint before importing the PCG code.
        import random
        import numpy as np
        random.seed(job['seed']); np.random.seed(job['seed'])
        random.random(); np.random.random()
    from toolusepcg.optimizer.greedy import GreedyOptimizer
    assert Path(sys.modules[GreedyOptimizer.__module__].__file__).resolve().is_relative_to(source)
    if args.cpu_only:
        if job['suite'] == 'z64':
            from experiments.r061_qwen38_m1 import _exact_config
            c = job['case']
            opt = GreedyOptimizer(_exact_config(c['player_key'], c['key_door']), verbose=False)
            opt.initialize(seed=c['seed'])
            import hashlib
            assert hashlib.sha256(opt.current_level.to_string().encode()).hexdigest()==c['initialization_sha256']
        else:
            runpy.run_path(str(source/'main.py'), run_name='mac_import_check')
            from toolusepcg.config import load_config
            from toolusepcg.level import Level
            cfg = load_config(source/job['config'])
            cfg.optimizer.prompt_profile = 'freeform'
            opt = GreedyOptimizer(cfg, verbose=False)
            initial = Level.from_string((source/job['initial']).read_text(), 'zelda')
            opt.initialize(seed=job['seed'], initial_level=initial)
        return

    with lms.Client('127.0.0.1:1234') as client:
        calls = 0
        def call_llm(self, prompt, max_retries=1, timeout=900.0):
            nonlocal calls
            calls += 1
            extra = self._build_openai_extra_body() or {}
            thinking = extra.get('chat_template_kwargs', {}).get('enable_thinking', False)
            response = transport.predict(client, model=model, messages=self._build_messages(prompt),
                schema=extra['structured_outputs']['json'], thinking=thinking,
                max_tokens=self.config.llm.max_tokens, seed=self.config.seed,
                temperature=self.config.llm.temperature, neutral_sampling=True,
                output=args.output/f'call_{calls:02}', timeout=900)
            self._last_reasoning_content = response['reasoning']
            stats = response['stats']
            return (response['final'], stats['stopReason']=='maxPredictedTokensReached' or not response['final'],
                    stats.get('promptTokensCount', 0), stats.get('predictedTokensCount', 0))
        GreedyOptimizer.call_llm = call_llm
        if job['suite'] == 'z64':
            from experiments import r061_qwen38_m1 as producer
            def studio_server_model_id(base_url, *, expected):
                if base_url != 'http://127.0.0.1:1234/v1':
                    raise RuntimeError('Unexpected model endpoint')
                loaded = json.loads(subprocess.check_output(['lms', 'ps', '--json'], text=True))
                identifier = verify_loaded_studio_model(loaded, expected)
                transport.save(args.output/'served_model_identity.json', {
                    'validation': 'selected_loaded_GGUF_context_and_slot', 'loaded_models': loaded,
                    'expected': expected, 'selected_identifier': identifier})
                return identifier
            producer._server_model_id = studio_server_model_id
            def enrich(optimizer, case):
                save_step_records(args.output/'steps.json', optimizer.step_records)
                (args.output/'final_level.txt').write_text(optimizer.current_level.to_string())
                return {'transport_calls': calls}
            producer.run_matched_arm(root/'benchmarks/zelda/test721.json',
                'http://127.0.0.1:1234/v1', args.output/'result.json',
                arm='mac_q4_verified', model_id=model, revision=job['model_sha256'],
                target_ids=[job['id']], max_steps=4, conversation_length=2,
                extra_instruction=(source/'prompts/zelda_geometry_serialization_rules_v3.txt').read_text(),
                truncation_recovery='thinking_off_handoff', max_tokens=8192,
                thinking_max_tokens=6144, temperature=0, request_seed_offset=0,
                record_enricher=enrich)
        else:
            sys.argv = [str(source/'main.py'), '--config', str(source/job['config']),
                '--init', str(source/job['initial']), '--seed', str(job['seed']),
                '--max-steps', '8', '--run-dir', str(args.output/'pcg'),
                '--provider', 'openai', '--model', model, '--base-url', 'http://127.0.0.1:1234/v1',
                '--max-tokens', '4096', '--temperature', '0.0',
                '--extra-instruction', job['instruction'], '--no-render-stats', '--no-render-sim', '--quiet', '--prompt-profile', 'freeform']
            runpy.run_path(str(source/'main.py'), run_name='__main__')


if __name__ == '__main__':
    main()
