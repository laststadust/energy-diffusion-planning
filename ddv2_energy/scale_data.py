"""Prepare all eligible official training tokens with log-disjoint validation.

The fixed protocol uses every eligible window, frame_gap=1 and actor_cap=128.
A complete preparation report is published only after all selected records have
feature, context and official PDM caches. Completed log artifacts are resumable.
"""
import argparse
import collections
import concurrent.futures as cf
import hashlib
import importlib
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import shutil
import time

_CENSUS_ALLOWED = None


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def publish_immutable(path, value):
    """Interrupted uncommitted .partial files may be replaced; committed files not."""
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Conflicting immutable artifact: ' + str(path))
    else:
        atomic_json(path, value)


def disk_guard(path, minimum_gib=300):
    if shutil.disk_usage(path).free < minimum_gib * 1024**3:
        raise RuntimeError(f'DISK_ABORT: output filesystem has less than {minimum_gib:g} GiB free')


def census_frames(frames, log, split, allowed_tokens, seed, actor_cap, selector=None):
    """Account for every official token present in this raw log, including edges.

    Selector is injectable only for pure structural unit tests. Production uses
    the unchanged selector, evaluating every window without quota or spacing loss.
    """
    if selector is None:
        selector = importlib.import_module('ddv2_energy.prepare_experiment').select_windows
    if not isinstance(frames, list):
        raise ValueError('Raw log must be a list')
    tokens = [f.get('token') for f in frames]
    if any(not isinstance(t, str) or not t for t in tokens) or len(tokens) != len(set(tokens)):
        raise ValueError('Raw log tokens missing or duplicated')
    present = set(tokens) & set(allowed_tokens)
    # len(frames) is a strict upper bound; the selector cannot hit this quota.
    rows, rejected = selector(frames, log, split, max(1, len(frames)), seed,
                              1, actor_cap, set(allowed_tokens))
    window_tokens = set(tokens[3:max(3, len(frames) - 10)])
    excluded = {k: v for k, v in rejected.items() if k != 'not_official_navtrain_token'}
    excluded['incomplete_scene_window'] = len(present - window_tokens)
    if excluded.get('temporal_spacing', 0):
        raise ValueError('Full census must not discard temporally adjacent frames')
    if len(rows) + sum(excluded.values()) != len(present):
        raise ValueError('Census official-token accounting mismatch: ' + log)
    selected = set()
    for row in rows:
        index = row['frame_index']
        if (row['token'] not in present or row['token'] in selected
                or (row['log'], row['split']) != (log, split)
                or not 3 <= index < len(frames) - 10 or tokens[index] != row['token']):
            raise ValueError('Invalid census-selected row: ' + log)
        selected.add(row['token'])
    # Keep the existing seeded order; no sort on labels, predictions or risk.
    return dict(log=log, split=split, rows=rows, exclusions=excluded,
                official_tokens_present=sorted(present), raw_frame_count=len(frames),
                nonofficial_window_count=rejected.get('not_official_navtrain_token', 0))


def census_log(job):
    """One owned marker per log; resume only exact raw/config/source identities."""
    raw, marker = Path(job['raw']), Path(job['marker'])
    identity = {k: job[k] for k in ('log', 'split', 'seed', 'actor_cap', 'spec_sha256')}
    identity['raw_sha256'] = sha256(raw)
    if marker.exists():
        previous = read_json(marker)
        if previous.get('identity') != identity:
            raise ValueError('Census resume raw/spec mismatch: ' + job['log'])
        verify_census_digest(previous)
        return previous
    with raw.open('rb') as stream:
        frames = pickle.load(stream)
    allowed = job.get('allowed_tokens', _CENSUS_ALLOWED)
    if allowed is None:
        raise ValueError('Census worker has no frozen official-token allowlist')
    result = census_frames(frames, job['log'], job['split'], allowed,
                           job['seed'], job['actor_cap'])
    if sha256(raw) != identity['raw_sha256']:
        raise ValueError('Raw source changed during census: ' + job['log'])
    result['identity'] = identity
    result['payload_sha256'] = census_digest(result)
    publish_immutable(marker, result)
    return result


def census_digest(value):
    payload = {k:v for k,v in value.items() if k != 'payload_sha256'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify_census_digest(value):
    if value.get('payload_sha256') != census_digest(value):
        raise ValueError('Committed census log payload checksum mismatch')


def initialize_census(allowed):
    global _CENSUS_ALLOWED
    # Transfer the token allowlist once per worker rather than once per log.
    _CENSUS_ALLOWED = set(allowed)
    import torch
    torch.set_num_threads(1)


def build_scale_log(job):
    disk_guard(job['out'], job.get('minimum_free_gib', 300))
    result = importlib.import_module('ddv2_energy.prepare_experiment').build_log(job)
    disk_guard(job['out'], job.get('minimum_free_gib', 300))
    return result


def combine_census(results, ordered_logs, allowed_tokens, spec):
    """Freeze an exhaustive selection; missing raw tokens are reported, not safe."""
    expected = dict(ordered_logs)
    if len(expected) != len(ordered_logs) or set(results) != set(expected):
        raise ValueError('Census has missing, extra or duplicate official logs')
    allowed = set(allowed_tokens)
    present, selected, rows, logs = set(), set(), [], []
    exclusions = collections.Counter()
    for log, split in ordered_logs:
        r = results[log]
        if r['log'] != log or r['split'] != split:
            raise ValueError('Census split mismatch: ' + log)
        found = set(r['official_tokens_present'])
        if not found <= allowed or present & found:
            raise ValueError('Duplicate or nonofficial token across raw logs')
        present.update(found)
        exclusions.update(r['exclusions'])
        for row in r['rows']:
            if (row['token'] not in found or row['token'] in selected
                    or (row['log'], row['split']) != (log, split)):
                raise ValueError('Invalid selected token/log/split')
            selected.add(row['token'])
            rows.append(row)
        logs.append(dict(log=log, split=split, selected=len(r['rows']),
                         excluded=r['exclusions'], raw_sha256=r['identity']['raw_sha256'],
                         official_token_count=len(found)))
    missing = sorted(allowed - present)
    exclusions['official_token_missing_from_raw_logs'] = len(missing)
    if len(rows) + sum(exclusions.values()) != len(allowed):
        raise ValueError('Full official-token coverage accounting failed')
    counts = collections.Counter(r['split'] for r in rows)
    if counts['train'] < 32 or counts['val'] < 16:
        raise ValueError('Insufficient eligible rows for 32-train/16-val engineering gate')
    return dict(spec=spec, rows=rows, logs=logs, exclusions=dict(exclusions),
                counts=dict(counts), official_token_count=len(allowed),
                official_log_count=len(ordered_logs), eligible_count=len(rows),
                eligible_fraction=len(rows) / len(allowed),
                missing_official_tokens=missing, coverage_complete=True)


def assemble_prepared(selection, artifacts):
    """Refuse partial/duplicate preparation before publishing consumer manifests."""
    expected = {r['token']: r for r in selection['rows']}
    if len(expected) != len(selection['rows']):
        raise ValueError('Duplicate frozen selection token')
    rows, contexts = {}, {}
    for artifact in artifacts:
        for field, dest in (('rows', rows), ('contexts', contexts)):
            for row in artifact[field]:
                token = row['token']
                if (token in dest or token not in expected
                        or (row['log'], row['split']) != (expected[token]['log'], expected[token]['split'])):
                    raise ValueError('Prepared rows conflict with selection')
                dest[token] = row
    if set(rows) != set(expected) or set(contexts) != set(expected):
        raise ValueError('Incomplete preparation cannot publish manifest')
    ordered = [rows[r['token']] for r in selection['rows']]
    context_records = [contexts[r['token']] for r in selection['rows']]
    engineering = ([r for r in ordered if r['split'] == 'train'][:32]
                   + [r for r in ordered if r['split'] == 'val'][:16])
    if len(engineering) != 48:
        raise ValueError('Engineering manifest must contain 32 train and 16 val')
    return ordered, context_records, engineering


def _source_hashes(preparation):
    files = [Path(__file__), Path(preparation.__file__), Path(inspect.getfile(preparation.context_from_window)),
             Path(inspect.getfile(preparation.scene_labels)), Path(inspect.getfile(preparation.validate_context)),
             Path(inspect.getfile(preparation.check_context_alignment))]
    return {str(p.resolve()): sha256(p) for p in sorted(set(files))}


def prepare(args):
    import fcntl
    import yaml
    preparation = importlib.import_module('ddv2_energy.prepare_experiment')
    preparation.validate_preparation_paths(args)
    out = Path(args.out)
    if not 0 <= args.min_free_gib < float('inf'):
        raise ValueError('min-free-gib must be finite and nonnegative')
    if args.actor_cap != 128 or args.seed != 0 or not 1 <= args.workers <= 4:
        raise ValueError('Frozen scale protocol requires actor-cap128, split-seed0, workers1..4')
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.scale_preparation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    try:
        disk_guard(out, args.min_free_gib)
        config_root = Path(args.repo) / 'navsim/planning/script/config/common/train_test_split/scene_filter'
        train_path, test_path = config_root / 'navtrain.yaml', config_root / 'navtest.yaml'
        train_cfg, test_cfg = [yaml.safe_load(p.read_text()) for p in (train_path, test_path)]
        if tuple(train_cfg[k] for k in ('num_history_frames', 'num_future_frames', 'frame_interval')) != (4, 10, 1):
            raise ValueError('Official planning window definition changed')
        tokens = train_cfg.get('tokens')
        if (not isinstance(tokens, list) or not tokens or len(tokens) != len(set(tokens))
                or any(not isinstance(t, str) or not t for t in tokens)):
            raise ValueError('An explicit unique official token allowlist is required')
        logs_root = Path(args.data) / 'navsim_logs/trainval'
        available = {p.stem for p in logs_root.glob('*.pkl')}
        if not set(train_cfg['log_names']) <= available:
            raise ValueError('Missing official navtrain raw logs: ' + repr(sorted(set(train_cfg['log_names']) - available)))
        ordered = preparation.split_logs(train_cfg['log_names'], test_cfg['log_names'], available, args.seed)
        if len(ordered) != 1192 or len(tokens) != 103288:
            raise ValueError('Pinned official counts changed; use the documented upstream revision')
        spec = dict(format='ddv2_full_eligible_preparation_v1', data=str(Path(args.data).resolve()),
                    repo=str(Path(args.repo).resolve()), feature_cache=str(Path(args.feature_cache).resolve()) if args.feature_cache else None,
                    seed=args.seed, actor_cap=args.actor_cap, frame_gap=1, max_per_log=None,
                    train_yaml_sha256=sha256(train_path), test_yaml_sha256=sha256(test_path),
                    source_sha256=_source_hashes(preparation), official_logs=len(ordered), official_tokens=len(tokens),
                    split_rule='seeded official logs; every fifth shuffled log is validation')
        spec_path = out / 'preparation_spec.json'
        if not spec_path.exists() and any(p.name != '.scale_preparation.lock' for p in out.iterdir()):
            raise ValueError('Output directory is not new and has no matching immutable scale spec')
        publish_immutable(spec_path, spec)
        spec_hash = sha256(spec_path)
        census_results = {}
        jobs = [dict(log=log, split=split, seed=args.seed, actor_cap=args.actor_cap,
                     raw=str(logs_root / (log + '.pkl')), marker=str(out / 'census_logs' / (log + '.json')),
                     spec_sha256=spec_hash) for log, split in ordered]
        with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
                                    initializer=initialize_census, initargs=(tokens,)) as pool:
            futures = {pool.submit(census_log, job): job['log'] for job in jobs}
            for future in cf.as_completed(futures):
                disk_guard(out, args.min_free_gib)
                r = future.result()
                census_results[r['log']] = r
                progress = dict(state='CENSUS', completed_logs=len(census_results), total_logs=len(ordered),
                                eligible_so_far=sum(len(x['rows']) for x in census_results.values()),
                                elapsed_s=time.monotonic() - started)
                atomic_json(out / 'preparation_progress.json', progress)
                print('CENSUS_LOG_COMPLETE', json.dumps(progress), r['log'], flush=True)
        selection = combine_census(census_results, ordered, tokens, spec)
        selection_path = out / 'selection.json'
        publish_immutable(selection_path, selection)
        atomic_json(out / 'preparation_progress.json', dict(state='CENSUS_FROZEN', counts=selection['counts'],
                    eligible_count=selection['eligible_count'], official_token_count=len(tokens),
                    exclusions=selection['exclusions'], selection_sha256=sha256(selection_path)))
        print('CENSUS_FROZEN', json.dumps({k:selection[k] for k in ('counts', 'eligible_count', 'eligible_fraction', 'exclusions')}), flush=True)
        # Only now can official PDM construction start. Geometry remains unchanged.
        grouped = collections.defaultdict(list)
        for row in selection['rows']:
            grouped[row['log']].append(row)
        raw_hashes = {r['log']: r['raw_sha256'] for r in selection['logs']}
        artifacts = {}
        build_jobs = [dict(log=log, rows=rows, raw_sha256=raw_hashes[log], out=str(out),
                           data=spec['data'], feature_cache=spec['feature_cache'], actor_cap=args.actor_cap,
                           minimum_free_gib=args.min_free_gib)
                      for log, rows in grouped.items()]
        with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
            futures = {pool.submit(build_scale_log, job): job['log'] for job in build_jobs}
            for future in cf.as_completed(futures):
                disk_guard(out, args.min_free_gib)
                r = future.result()
                artifacts[r['log']] = r
                atomic_json(out / 'preparation_progress.json', dict(state='BUILDING',
                    completed=sum(len(a['rows']) for a in artifacts.values()), total=len(selection['rows']),
                    completed_logs=len(artifacts), total_logs=len(build_jobs), elapsed_s=time.monotonic() - started))
        rows, ctx_records, engineering = assemble_prepared(selection, artifacts.values())
        ctx_root = out / 'contexts'
        if (ctx_root / 'manifest.json').exists():
            if preparation.ContextStore(ctx_root).records != {r['token']:r for r in ctx_records}:
                raise ValueError('Existing context manifest differs from complete build')
        else:
            preparation.write_manifest(ctx_root, ctx_records, dict(spec=spec,
                selection_sha256=sha256(selection_path), split_provenance='full eligible official navtrain; log-isolated',
                original_geometry='unchanged scene_labels/scene_sdf',
                future_use='supervision only; no future graph or policy input'))
        publish_immutable(out / 'manifest.json', rows)
        publish_immutable(out / 'engineering_manifest.json', engineering)
        statuses = collections.Counter()
        for r in artifacts.values():
            statuses.update(r['statuses'])
        paths = ['manifest.json', 'contexts/manifest.json', 'selection.json', 'engineering_manifest.json']
        report = dict(state='COMPLETE', format='ddv2_full_eligible_preparation_v1', counts=selection['counts'],
                      total=len(rows), official_token_count=len(tokens), official_log_count=len(ordered),
                      eligible_fraction=selection['eligible_fraction'], coverage_complete=True,
                      exclusions=selection['exclusions'], feature_sources=dict(statuses),
                      manifest_sha256=sha256(out / 'manifest.json'), selection_sha256=sha256(selection_path),
                      context_manifest_sha256=sha256(ctx_root / 'manifest.json'),
                      engineering_counts={'train':32,'val':16},
                      file_sha256={p:sha256(out / p) for p in paths},
                      prepared_log_artifact_sha256={log:sha256(out / 'prepared_logs' / (log + '.json')) for log in grouped},
                      workers=args.workers, elapsed_this_invocation_s=time.monotonic() - started,
                      prepared_log_seconds=sum(a['elapsed_s'] for a in artifacts.values()), spec=spec,
                      pdm_source='unchanged upstream MetricCacheProcessor; history4 future10')
        # Completion marker is last; a consumer must never trust manifest alone.
        previous = out / 'preparation_report.json'
        if previous.exists():
            old = read_json(previous)
            for key in ('state', 'spec', 'file_sha256', 'prepared_log_artifact_sha256', 'counts', 'total'):
                if old.get(key) != report.get(key):
                    raise ValueError('Completed preparation changed: ' + key)
            report = old
        else:
            atomic_json(previous, report)
        atomic_json(out / 'preparation_progress.json', report)
        print('SCALE_PREPARATION_COMPLETE', json.dumps({k:report[k] for k in ('state','counts','total','eligible_fraction')}), flush=True)
        return report
    except Exception as exc:
        atomic_json(out / 'preparation_progress.json', dict(state='FAILED', error=repr(exc), elapsed_s=time.monotonic() - started))
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--actor-cap', type=int, default=128)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--min-free-gib', type=float, default=300, help='Required free space on output filesystem')
    p.add_argument('--repo', required=True, help='Pinned upstream repository')
    p.add_argument('--data-root', '--data', dest='data', required=True, help='Dataset root containing maps and trainval logs/sensors')
    p.add_argument('--feature-cache', default=None, help='Optional existing feature cache; verified before reuse')
    prepare(p.parse_args())


if __name__ == '__main__':
    main()
