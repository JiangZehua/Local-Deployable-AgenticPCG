"""PCG-independent LM Studio prediction adapter for isolated producer checkouts.

Only lmstudio 1.5.0 is supported. Requests, fragments and failures are retained.
"""
from __future__ import annotations
import importlib.metadata
import json
import threading
import time
import lmstudio as lms
from lmstudio.json_api import PredictionEndpoint
from lmstudio.sync_api import PredictionStream

CUSTOM_PREFIX = "ext.virtualModel.customField.qwen.qwen3.827b."


class RecordedPredictionEndpoint(PredictionEndpoint):
    """Keep new server fields that the older SDK's typed config drops."""
    def __init__(self, *args, evidence_dir, **kwargs):
        self.evidence_dir = evidence_dir
        super().__init__(*args, **kwargs)

    def iter_message_events(self, contents):
        if contents and contents.get('type') == 'success':
            save(self.evidence_dir/'wire_success.json', contents)
        yield from super().iter_message_events(contents)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def split_fragments(fragments):
    final, reasoning = [], []
    for f in fragments:
        kind = f['reasoningType']
        if kind == 'none':
            final.append(f['content'])
        elif kind == 'reasoning':
            reasoning.append(f['content'])
        elif kind not in {'reasoningStartTag', 'reasoningEndTag'}:
            raise ValueError(f"Unknown reasoning fragment type: {kind}")
    return ''.join(final), ''.join(reasoning)



def predict(client, *, model, messages, schema, thinking, max_tokens, output,
            temperature=0.0, seed=None, timeout=600, neutral_sampling=False):
    if importlib.metadata.version('lmstudio') != '1.5.0':
        raise RuntimeError('Revalidate wire fields before changing SDK version')
    output.mkdir(parents=True, exist_ok=False)
    chat = lms.Chat()
    for message in messages:
        chat.add_entry(message['role'], message['content'])
    fragments = []
    config = {'temperature': temperature, 'maxTokens': max_tokens,
              'contextOverflowPolicy': 'stopAtLimit'}
    if neutral_sampling:
        config.update(topKSampling=-1, topPSampling=1.0, minPSampling=0.0, repeatPenalty=1.0)
    endpoint = RecordedPredictionEndpoint(model, chat, response_format=schema, config=config,
        evidence_dir=output,
        on_prediction_fragment=lambda f: fragments.append(f.to_dict()))
    fields = endpoint.creation_params['predictionConfigStack']['layers'][-1]['config']['fields']
    fields.extend([
        {'key': CUSTOM_PREFIX+'enableThinking', 'value': thinking},
        {'key': CUSTOM_PREFIX+'reasoningEffort', 'value': 'xhigh'},
    ])
    if seed is not None:
        fields.append({'key': 'llm.prediction.seed', 'value': {'checked': True, 'value': seed}})
    save(output/'request.json', endpoint.creation_params)
    started = time.perf_counter()
    stream = PredictionStream(client.llm._create_channel(endpoint), endpoint)
    timer = threading.Timer(timeout, stream.cancel)
    timer.daemon = True
    timer.start()
    try:
        result = stream.wait_for_result()
        stats = result.stats.to_dict()
        final, reasoning = split_fragments(fragments)
        response = {'content': result.content, 'final': final, 'reasoning': reasoning,
                    'stats': stats, 'prediction_config': result.prediction_config.to_dict(),
                    'elapsed_seconds': time.perf_counter()-started}
        save(output/'response.json', response)
        if neutral_sampling:
            raw = json.loads((output/'wire_success.json').read_text())
            effective = {f['key']: f['value'] for f in raw['predictionConfig']['fields']}
            expected = {
                'llm.prediction.topKSampling': -1,
                'llm.prediction.topPSampling': {'checked': True, 'value': 1},
                'llm.prediction.minPSampling': {'checked': True, 'value': 0},
                'llm.prediction.repeatPenalty': {'checked': True, 'value': 1},
                CUSTOM_PREFIX+'enableThinking': thinking,
            }
            if seed is not None:
                expected['llm.prediction.seed'] = {'checked': True, 'value': seed}
            for key, value in expected.items():
                if effective.get(key) != value:
                    raise RuntimeError(f'Effective server configuration differs: {key}')
        if stats.get('totalDraftTokensCount', 0):
            raise RuntimeError('Speculative decoding unexpectedly active')
        if stats['stopReason'] not in {'eosFound', 'stopStringFound', 'maxPredictedTokensReached'}:
            raise RuntimeError('Incomplete prediction: '+stats['stopReason'])
        return response
    except Exception as error:
        save(output/'error.json', {'error': repr(error), 'elapsed_seconds': time.perf_counter()-started})
        raise
    finally:
        timer.cancel()
        save(output/'fragments.json', fragments)
