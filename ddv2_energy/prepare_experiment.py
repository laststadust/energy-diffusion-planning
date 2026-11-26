"""Deterministic training-subset selection and feature/PDM preparation.

Selection uses official training membership and annotation completeness. It is
frozen before PDM generation and does not depend on model predictions or scores.
Complete per-log artifacts are checksum-verified when resuming.
"""
import argparse
import collections
import concurrent.futures as cf
import contextlib
import gzip
import hashlib
import json
import lzma
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch

from .contexts import (ContextStore, save_context, write_manifest, validate_context,
                       sha256_file, safe_name)
from .prepare_contexts import context_from_window, _check_annotations
from .runtime import check_context_alignment
from .preprocessing import pose, scene_labels, scene_sdf



def write_json(path, payload):
    """Atomic update only for this invocation's progress/owned artifact files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def log_seed(seed, log):
    return int(hashlib.sha256(f'{seed}:{log}'.encode()).hexdigest()[:16], 16)


def split_logs(train_logs, test_logs, available, seed):
    """Assign each available official training log to exactly one training or validation split."""
    train, test = set(train_logs), set(test_logs)
    if not train or train & test:
        raise ValueError('Official train/test log lists empty or overlapping')
    names = sorted(train & set(available))
    if len(names) < 2:
        raise ValueError('Not enough available official navtrain logs')
    random.Random(seed).shuffle(names)
    return [(log, 'val' if index % 5 == 0 else 'train') for index, log in enumerate(names)]


def select_windows(frames, log, split, limit, seed, frame_gap, actor_cap, allowed_tokens=None):
    """Choose complete supervised windows before computing map/PDM/model scores."""
    if split not in ('train', 'val') or limit < 1 or frame_gap < 1 or actor_cap < 1:
        raise ValueError('Invalid selection configuration')
    if not isinstance(frames, list):
        raise ValueError('Raw log must be a list')
    tokens = [f.get('token') for f in frames]
    if any(not isinstance(t, str) for t in tokens) or len(set(tokens)) != len(tokens):
        raise ValueError('Raw log tokens missing or duplicated')
    indices = list(range(3, len(frames) - 10))  # 4 history including current + 10 future
    random.Random(log_seed(seed, log)).shuffle(indices)
    selected, excluded = [], collections.Counter()
    for index in indices:
        if len(selected) == limit:
            break
        frame = frames[index]
        if allowed_tokens is not None and frame['token'] not in allowed_tokens:
            excluded['not_official_navtrain_token'] += 1; continue
        if any(abs(index - row['frame_index']) < frame_gap for row in selected):
            excluded['temporal_spacing'] += 1; continue
        if not frame.get('roadblock_ids'):
            excluded['missing_route'] += 1; continue
        window = frames[index:index + 9]
        locations = [f.get('map_location') for f in window]
        if not isinstance(locations[0], str) or not locations[0] or len(set(locations)) != 1:
            excluded['inconsistent_map_location'] += 1; continue
        if any(f.get('log_name', log) != log for f in frames[index-3:index+11]):
            raise ValueError('Raw log_name does not match its filename')
        try:
            for f in window:
                _check_annotations(f)
            labels = scene_labels(window, cap=actor_cap)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            excluded['invalid_annotation_' + type(exc).__name__] += 1; continue
        if labels is None:
            excluded['invalid_time_ego_or_reverse'] += 1; continue
        if labels['truncated']:
            excluded['current_actor_cap_exceeded'] += 1; continue
        if not labels['label_ok']:
            excluded['incomplete_future_actor_labels'] += 1; continue
        count = labels['roi_count']
        if not labels['mask'][:count].all():
            excluded['invalid_current_actor'] += 1; continue
        safe_name(frame['token'])
        selected.append(dict(token=frame['token'], log=log, split=split,
                             frame_index=index, actor_count=int(count)))
    return selected, dict(excluded)


def check_selection(rows, train_count, val_count, max_per_log, frame_gap):
    counts = collections.Counter(r['split'] for r in rows)
    if counts != {'train': train_count, 'val': val_count}:
        raise ValueError('Insufficient eligible scenes: ' + repr(dict(counts)))
    if len({r['token'] for r in rows}) != len(rows):
        raise ValueError('Duplicate selected token')
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['log']].append(row)
    for log, group in groups.items():
        if len({r['split'] for r in group}) != 1 or len(group) > max_per_log:
            raise ValueError('Log overlap or per-log count exceeded: ' + log)
        indices = sorted(r['frame_index'] for r in group)
        if any(b-a < frame_gap for a, b in zip(indices, indices[1:])):
            raise ValueError('Temporal spacing violated: ' + log)


def verify_log_artifact(artifact):
    for path, checksum in artifact['file_sha256'].items():
        if not Path(path).is_file() or sha256_file(path) != checksum:
            raise ValueError('Completed preparation artifact changed/missing: ' + path)
    return artifact


def _publish_context(root, context):
    path = Path(root) / 'contexts' / (context['token'] + '.pt')
    if not path.exists():
        return save_context(root, context)
    existing = validate_context(torch.load(path, map_location='cpu', weights_only=True))
    for key, value in context.items():
        same = torch.equal(value, existing[key]) if torch.is_tensor(value) else value == existing[key]
        if not same:
            raise ValueError('Existing partial context differs: ' + str(path))
    return dict(token=context['token'], log=context['log'], split=context['split'],
                path='contexts/' + path.name, sha256=sha256_file(path))


def _read_cache(feature_path, target_path, context):
    with gzip.open(feature_path, 'rb') as f:
        features = pickle.load(f)
    with gzip.open(target_path, 'rb') as f:
        targets = pickle.load(f)
    for key, shape in dict(camera_feature=(3, 256, 1024), lidar_feature=(1, 256, 256), status_feature=(8,)).items():
        if tuple(features[key].shape) != shape or not bool(torch.isfinite(features[key]).all()):
            raise ValueError('Invalid cached feature: ' + key)
    check_context_alignment(context, {k:v[None] for k,v in features.items()},
                            {k:v[None] for k,v in targets.items()})
    return features, targets


def _write_cache(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    with gzip.open(temporary, 'wb', compresslevel=1) as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def build_log(job):
    """One process owns all artifacts for one selected raw log."""
    from navsim.common.dataclasses import Scene, SensorConfig
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.agents.diffusiondrivev2.diffusiondrivev2_rl_config import TransfuserConfig
    from navsim.agents.diffusiondrivev2.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    torch.set_num_threads(1)
    out, data = (Path(job[k]) for k in ('out', 'data'))
    feature_cache = Path(job['feature_cache']) if job.get('feature_cache') else None
    log, rows = job['log'], job['rows']
    marker = out / 'prepared_logs' / (log + '.json')
    raw_path = data / 'navsim_logs/trainval' / (log + '.pkl')
    if sha256_file(raw_path) != job['raw_sha256']:
        raise ValueError('Raw log changed since selection: ' + log)
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['selection'] != rows or saved['raw_sha256'] != job['raw_sha256']:
            raise ValueError('Prepared log does not match frozen selection')
        return verify_log_artifact(saved)
    with raw_path.open('rb') as f:
        frames = pickle.load(f)
    processor = MetricCacheProcessor(str(out / 'metric_cache'), False)
    fb, tb = TransfuserFeatureBuilder(TransfuserConfig()), TransfuserTargetBuilder(TransfuserConfig())
    maps, contexts, manifest, hashes = {}, [], [], {}
    statuses = collections.Counter()
    began = time.monotonic()
    for row in rows:
        index, token = row['frame_index'], row['token']
        window = frames[index:index+9]
        if window[0]['token'] != token:
            raise ValueError('Frozen raw frame index/token mismatch')
        location = window[0]['map_location']
        if location not in maps:
            maps[location] = get_maps_api(str(data / 'maps'), 'nuplan-maps-v1.0', location)
        sdf = scene_sdf(maps[location], *pose(window[0]))
        context = context_from_window(row, window, sdf, job['actor_cap'])
        if not context['label_ok'] or context['truncated']:
            raise ValueError('Frozen selection became incomplete')
        record = _publish_context(out / 'contexts', context)
        contexts.append(record)
        hashes[str(out / 'contexts' / record['path'])] = record['sha256']
        source_feature = feature_cache / log / token / 'transfuser_feature.gz' if feature_cache else None
        source_target = feature_cache / log / token / 'transfuser_target.gz' if feature_cache else None
        # Prefer verified user-owned existing features. A stale cache is rebuilt
        # with the official feature builder, never edited in place.
        try:
            if feature_cache is None:
                raise FileNotFoundError('No existing feature cache was requested')
            _read_cache(source_feature, source_target, context)
            feature_path, target_path = source_feature, source_target
            statuses['verified_existing_features'] += 1
        except (OSError, EOFError, KeyError, ValueError, RuntimeError, pickle.UnpicklingError):
            feature_path = out / 'feature_cache' / log / token / 'transfuser_feature.gz'
            target_path = feature_path.with_name('transfuser_target.gz')
            if feature_path.exists() and target_path.exists():
                _read_cache(feature_path, target_path, context)
                statuses['resumed_generated_features'] += 1
            else:
                sensors = SensorConfig.build_no_sensors()
                sensors.cam_f0 = sensors.cam_l0 = sensors.cam_r0 = sensors.lidar_pc = [3]
                scene = Scene.from_scene_dict_list(frames[index-3:index+11], data / 'sensor_blobs/trainval',
                                                   num_history_frames=4, num_future_frames=10, sensor_config=sensors)
                features, targets = fb.compute_features(scene.get_agent_input()), tb.compute_targets(scene)
                check_context_alignment(context, {k:v[None] for k,v in features.items()},
                                        {k:v[None] for k,v in targets.items()})
                _write_cache(feature_path, features); _write_cache(target_path, targets)
                _read_cache(feature_path, target_path, context)
                statuses['generated_official_features'] += 1
        # Use upstream no-sensor scenario, planner, observations and map cache.
        scene = Scene.from_scene_dict_list(frames[index-3:index+11], None,
                                           num_history_frames=4, num_future_frames=10,
                                           sensor_config=SensorConfig.build_no_sensors())
        scenario = NavSimScenario(scene, str(data / 'maps'), 'nuplan-maps-v1.0')
        expected = out / 'metric_cache' / log / scenario.scenario_type / token / 'metric_cache.pkl'
        processor.compute_metric_cache(scenario)
        with lzma.open(expected, 'rb') as f:
            metric = pickle.load(f)
        actual, wanted = metric.ego_state, scenario.initial_ego_state
        if actual.time_point.time_us != wanted.time_point.time_us or not np.allclose(
                [actual.rear_axle.x, actual.rear_axle.y, actual.rear_axle.heading],
                [wanted.rear_axle.x, wanted.rear_axle.y, wanted.rear_axle.heading], atol=1e-6, rtol=0):
            raise ValueError('Metric cache/current scenario mismatch: ' + token)
        item = dict(token=token, log=log, split=row['split'], feature_path=str(feature_path),
                    target_path=str(target_path), metric_cache_path=str(expected))
        for key in ('feature_path', 'target_path', 'metric_cache_path'):
            hashes[item[key]] = sha256_file(item[key])
        manifest.append(item)
        print(f'PREPARED {log}/{token} split={row["split"]} actors={len(context["actors"])}', flush=True)
    result = dict(log=log, selection=rows, raw_sha256=job['raw_sha256'], rows=manifest,
                  contexts=contexts, file_sha256=hashes, statuses=dict(statuses), elapsed_s=time.monotonic()-began)
    write_json(marker, result)
    return result



def validate_preparation_paths(args):
    """Check explicit input roots and prevent output/input directory overlap."""
    out, repo, data = (Path(value).expanduser().resolve() for value in
                       (args.out, args.repo, args.data))
    if not repo.is_dir():
        raise FileNotFoundError("Missing upstream repository: " + str(repo))
    for source in (data / "navsim_logs" / "trainval", data / "maps"):
        if not source.is_dir():
            raise FileNotFoundError("Missing dataset directory: " + str(source))
    inputs = [data]
    if args.feature_cache:
        cache = Path(args.feature_cache).expanduser().resolve()
        if not cache.is_dir():
            raise FileNotFoundError("Missing requested feature cache: " + str(cache))
        inputs.append(cache)
    for source in inputs:
        if out == source or source in out.parents or out in source.parents:
            raise ValueError("Output must not overlap the dataset or reused feature cache")
    args.out, args.repo, args.data = str(out), str(repo), str(data)
    if args.feature_cache:
        args.feature_cache = str(cache)


def prepare(args):
    import yaml
    validate_preparation_paths(args)
    out = Path(args.out)
    if min(args.train_count, args.val_count, args.max_per_log, args.frame_gap, args.actor_cap, args.workers) < 1 or args.workers > 4:
        raise ValueError('Positive settings required; CPU worker cap is 4')
    out.mkdir(parents=True, exist_ok=True)
    # A POSIX lock prevents concurrent writers to the same preparation directory.
    import fcntl
    lock = (out / '.preparation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        directory = Path(args.repo) / 'navsim/planning/script/config/common/train_test_split/scene_filter'
        train_path, test_path = directory / 'navtrain.yaml', directory / 'navtest.yaml'
        train_cfg, test_cfg = [yaml.safe_load(p.read_text()) for p in (train_path, test_path)]
        if (train_cfg['num_history_frames'], train_cfg['num_future_frames'], train_cfg['frame_interval']) != (4, 10, 1):
            raise ValueError('Unexpected official scene filter horizon')
        tokens = train_cfg.get('tokens')
        allowed_tokens = set(tokens) if tokens is not None else None
        spec = dict(format='ddv2_pilot_preparation_v1', data=str(Path(args.data).resolve()),
                    repo=str(Path(args.repo).resolve()), feature_cache=str(Path(args.feature_cache).resolve()) if args.feature_cache else None,
                    train_count=args.train_count, val_count=args.val_count, seed=args.seed,
                    max_per_log=args.max_per_log, frame_gap=args.frame_gap, actor_cap=args.actor_cap,
                    train_yaml_sha256=sha256_file(train_path), test_yaml_sha256=sha256_file(test_path))
        spec_path = out / 'preparation_spec.json'
        if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
            raise ValueError('Preparation settings changed; use a NEW output directory')
        if not spec_path.exists():
            if any(p.name not in {'.preparation.lock'} for p in out.iterdir()):
                raise ValueError('Output is not a new experiment directory')
            write_json(spec_path, spec)
        started = time.monotonic()
        selection_path, work_path = out / 'selection.json', out / 'selection_work.json'
        logs_root = Path(args.data) / 'navsim_logs/trainval'
        available = [p.stem for p in logs_root.glob('*.pkl')]
        ordered = split_logs(train_cfg['log_names'], test_cfg['log_names'], available, args.seed)
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
        else:
            work = json.loads(work_path.read_text()) if work_path.exists() else dict(cursor=0, rows=[], logs=[], exclusions={})
            counts = collections.Counter(r['split'] for r in work['rows'])
            exclusions = collections.Counter(work['exclusions'])
            limits = {'train': args.train_count, 'val': args.val_count}
            for cursor in range(work['cursor'], len(ordered)):
                log, split = ordered[cursor]
                needed = limits[split] - counts[split]
                if needed <= 0:
                    continue
                raw = logs_root / (log + '.pkl')
                with raw.open('rb') as stream:
                    frames = pickle.load(stream)
                rows, rejected = select_windows(frames, log, split, min(args.max_per_log, needed),
                                                 args.seed, args.frame_gap, args.actor_cap, allowed_tokens)
                exclusions.update(rejected)
                counts[split] += len(rows); work['rows'].extend(rows)
                work['logs'].append(dict(log=log, split=split, selected=len(rows), excluded=rejected,
                                          raw_sha256=sha256_file(raw)))
                work.update(cursor=cursor+1, exclusions=dict(exclusions))
                write_json(work_path, work)
                write_json(out / 'preparation_progress.json', dict(state='SELECTING', counts=dict(counts),
                           target=limits, logs_inspected=len(work['logs']), exclusions=dict(exclusions)))
                print('SELECTED', dict(counts), log, flush=True)
                if counts == limits:
                    break
            check_selection(work['rows'], args.train_count, args.val_count, args.max_per_log, args.frame_gap)
            selection = dict(spec=spec, rows=work['rows'], logs=work['logs'], exclusions=dict(exclusions))
            write_json(selection_path, selection)
        if selection['spec'] != spec:
            raise ValueError('Frozen selection does not match experiment spec')
        check_selection(selection['rows'], args.train_count, args.val_count, args.max_per_log, args.frame_gap)
        grouped = collections.defaultdict(list)
        for row in selection['rows']:
            grouped[row['log']].append(row)
        sources = {r['log']: r['raw_sha256'] for r in selection['logs']}
        jobs = [dict(log=log, rows=rows, raw_sha256=sources[log], out=str(out), data=spec['data'],
                     feature_cache=spec['feature_cache'], actor_cap=args.actor_cap) for log, rows in grouped.items()]
        results = {}
        with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
            futures = {pool.submit(build_log, job): job['log'] for job in jobs}
            for future in cf.as_completed(futures):
                result = future.result(); results[result['log']] = result
                completed = sum(len(r['rows']) for r in results.values())
                write_json(out / 'preparation_progress.json', dict(state='BUILDING', completed=completed,
                           total=len(selection['rows']), completed_logs=len(results), total_logs=len(jobs),
                           elapsed_s=time.monotonic()-started))
        lookup = {r['token']:r for result in results.values() for r in result['rows']}
        rows = [lookup[r['token']] for r in selection['rows']]
        ctx_lookup = {r['token']:r for result in results.values() for r in result['contexts']}
        context_records = [ctx_lookup[r['token']] for r in rows]
        context_root = out / 'contexts'
        if (context_root / 'manifest.json').exists():
            store = ContextStore(context_root)
            if store.records != {r['token']:r for r in context_records}:
                raise ValueError('Existing context manifest changed')
        else:
            write_manifest(context_root, context_records, dict(spec=spec, selection_sha256=sha256_file(selection_path),
                           split_provenance='official navtrain log AND token allowlists; seeded log-disjoint subset',
                           original_geometry='ddv2_energy.preprocessing.scene_labels/scene_sdf',
                           future_use='supervision labels only; no future policy or graph inputs'))
        manifest_path = out / 'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != rows:
            raise ValueError('Existing experiment manifest changed')
        if not manifest_path.exists():
            write_json(manifest_path, rows)
        statuses = collections.Counter()
        for result in results.values():
            statuses.update(result['statuses'])
        report = dict(state='COMPLETE', counts=dict(collections.Counter(r['split'] for r in rows)),
                      logs=dict(collections.Counter(r[0]['split'] for r in grouped.values())),
                      total=len(rows), exclusions=selection['exclusions'], feature_sources=dict(statuses),
                      selection_sha256=sha256_file(selection_path), manifest_sha256=sha256_file(manifest_path),
                      context_manifest_sha256=sha256_file(context_root / 'manifest.json'),
                      elapsed_this_invocation_s=time.monotonic()-started, workers=args.workers,
                      prepared_log_seconds=sum(r['elapsed_s'] for r in results.values()), spec=spec,
                      pdm_source='unchanged upstream MetricCacheProcessor with 4 history+10 future frames')
        write_json(out / 'preparation_report.json', report)
        write_json(out / 'preparation_progress.json', report)
        print('PREPARATION_COMPLETE', json.dumps(report), flush=True)
        return report
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    for name, value in [('train-count',512), ('val-count',128), ('seed',0), ('max-per-log',8),
                        ('frame-gap',8), ('actor-cap',32), ('workers',4)]:
        p.add_argument('--'+name, type=int, default=value)
    p.add_argument('--repo', required=True, help='Pinned upstream repository')
    p.add_argument('--data-root', '--data', dest='data', required=True, help='Dataset root containing maps and trainval logs/sensors')
    p.add_argument('--feature-cache', default=None, help='Optional existing feature cache; verified before reuse')
    args = p.parse_args()
    try:
        prepare(args)
    except Exception as exc:
        print('PREPARATION_FAILED', type(exc).__name__, str(exc), flush=True)
        raise


if __name__ == '__main__':
    main()
