"""Record background process load while the existing experiment lock is held.

This observer never starts or stops inference or other applications. It saves
executable paths from ps comm, never process arguments or environment values.
The recorded start time delimits its coverage; earlier gaps remain missing.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def active(lock):
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    fcntl.flock(lock, fcntl.LOCK_UN)
    return False


def sample_processes():
    try:
        result = subprocess.run(
            ['ps', '-axo', 'pid=,ppid=,rss=,%cpu=,comm='],
            text=True, capture_output=True, timeout=5)
    except subprocess.TimeoutExpired:
        return {'error': 'ps_timeout', 'processes': None}
    if result.returncode:
        return {'error': 'ps_failed', 'returncode': result.returncode,
                'stderr': result.stderr, 'processes': None}
    kept = []
    malformed = 0
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 4)
        try:
            pid, ppid, rss = map(int, parts[:3])
            cpu = float(parts[3])
            executable = parts[4]
        except (ValueError, IndexError):
            malformed += 1
            continue
        if rss >= 256 * 1024 or cpu >= 20:
            kept.append({'pid': pid, 'ppid': ppid, 'rss_kib': rss,
                         'cpu_percent': cpu, 'executable': executable})
    return {'processes': kept, 'malformed_rows': malformed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    with (run / 'run.lock').open('a') as lock:
        if not active(lock):
            raise RuntimeError('No active runner; background observation not started')
        out = run / 'diagnostics/background_resources' / datetime.now(
            timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        out.mkdir(parents=True, exist_ok=False)
        source = Path(__file__).read_bytes()
        (out / 'observer_source.py').write_bytes(source)
        metadata = {'started_utc': utc(), 'planned_interval_seconds': 1,
                    'inclusion_rule': 'RSS >=256 MiB or CPU >=20 percent',
                    'source_sha256': hashlib.sha256(source).hexdigest(),
                    'source_path': str(Path(__file__).resolve()),
                    'scope': 'Late supplemental observation for this active run only',
                    'caveats': ['RSS is not total unified memory',
                                'Executable path is not proof of project ownership',
                                'Below-threshold processes are not retained',
                                'A command timeout creates a measured sampling gap']}
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(json.dumps({'event': 'observer_started', 'output': str(out)}), flush=True)
        began = deadline = time.monotonic()
        with (out / 'samples.jsonl').open('x') as stream:
            while active(lock) and time.monotonic() - began < 48 * 3600:
                sample = {'utc': utc(), 'monotonic_seconds': time.monotonic(),
                          **sample_processes()}
                stream.write(json.dumps(sample) + '\n')
                stream.flush()
                deadline += 1
                time.sleep(max(0, deadline - time.monotonic()))
                if deadline < time.monotonic() - 1:
                    deadline = time.monotonic()
        (out / 'finished.json').write_text(json.dumps({
            'utc': utc(), 'elapsed_seconds': time.monotonic() - began,
            'runner_active': active(lock)}, indent=2) + '\n')


if __name__ == '__main__':
    main()
