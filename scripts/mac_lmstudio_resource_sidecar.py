"""Supplemental OS telemetry for an already-running frozen experiment.

Observes the existing run lock; never starts/stops inference or other apps.
The late start is recorded explicitly and does not repair earlier missing data.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=5)
        return {'stdout': result.stdout, 'stderr': result.stderr, 'returncode': result.returncode}
    except subprocess.TimeoutExpired:
        return {'error': 'timeout'}


def active(lock):
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    fcntl.flock(lock, fcntl.LOCK_UN)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    out = run/'supplemental_resources'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True, exist_ok=False)
    with (run/'run.lock').open('a') as lock:
        if not active(lock):
            raise RuntimeError('No active runner; supplementary collection not started')
        metadata = {'started_utc': utc(), 'planned_interval_seconds': .8,
                    'scope': 'post-start supplemental observations; original resource log retained',
                    'power': command(['pmset', '-g', 'batt']),
                    'power_settings': command(['pmset', '-g', 'custom']),
                    'thermal_start': command(['pmset', '-g', 'therm']),
                    'hardware': command(['sysctl', 'hw.memsize', 'hw.model']),
                    'os': command(['sw_vers'])}
        (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        began = time.monotonic()
        deadline = began
        with (out/'samples.jsonl').open('a') as stream:
            while active(lock) and time.monotonic()-began < 48*3600:
                sample = {'utc': utc(), 'monotonic_seconds': time.monotonic(),
                          'vm_stat': command(['vm_stat']),
                          'swap': command(['sysctl', 'vm.swapusage']),
                          'pressure': command(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'])}
                stream.write(json.dumps(sample)+'\n'); stream.flush()
                deadline += .8
                time.sleep(max(0, deadline-time.monotonic()))
                if deadline < time.monotonic()-1:
                    deadline = time.monotonic()
        (out/'finished.json').write_text(json.dumps({'utc': utc(),
            'elapsed_seconds': time.monotonic()-began, 'runner_active': active(lock)}, indent=2)+'\n')


if __name__ == '__main__':
    main()
