#!/usr/bin/env python3
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Convert a Perfetto/Chrome-Trace-Event JSON produced by Intel Trace Tools (ut-tool-ext)
into the analysis.json consumed by trace_viz.html.

The Perfetto JSON is expected to contain ov.genai ITT events captured from the
OpenVINO GenAI Continuous Batching pipeline:
  genai.cb.step           — one complete pipeline step
  genai.cb.scheduling     — scheduler phase (child of step)
  genai.cb.copy_blocks    — KV block copy (child of scheduling)
  genai.cb.forward        — model forward pass (child of step)
  genai.cb.sample         — token sampling (child of step)
  genai.cb.fork_free      — sequence fork/free bookkeeping (child of step)
  genai.cb.notify_dropped — dropped request notification (child of step)
  genai.cb.cleanup        — request cleanup (child of step)

  Block state transition events (child of scheduling/cleanup):
  genai.cb.blk.alloc      — physical block allocated to a sequence
  genai.cb.blk.hit        — prefix-cache block reused by a sequence
  genai.cb.blk.cow        — copy-on-write: old block forked to new block
  genai.cb.blk.hash       — block hash updated after new tokens appended
  genai.cb.blk.free       — block freed from a sequence

    Current block events use stable names (for example `genai.cb.blk.alloc`) and
    numeric task metadata (`seq_id`, `physical_index`, `hash`, `extra`). The
    converter also accepts the earlier format that encoded these values in the
    event name.

Hardware performance counters (cpi, cpu_load, cpu_op_freq) are correlated by
timestamp and averaged per step.

When the corresponding custom ITT metadata events are present, the converter
reconstructs active/waiting counts, cache usage, per-sequence scheduled tokens,
prefix-cache hits, request summaries, and memory snapshots. Cache byte counts
and total-token counts are not included in this Perfetto analysis and remain 0.

Usage:
  python perfetto_to_analysis.py cb.json [--out analysis.json] [--model NAME] [--device DEV] [--notes TEXT]
"""

import argparse
import bisect
import json
import math
import os
import re
import sys
import datetime
from collections import defaultdict


# ── Sub-task names that appear as direct or indirect children of genai.cb.step ─
STEP_SUBTASKS = [
    "genai.cb.scheduling",
    "genai.cb.copy_blocks",  # child of scheduling
    "genai.cb.forward",
    "genai.cb.sample",
    "genai.cb.fork_free",
    "genai.cb.notify_dropped",
    "genai.cb.cleanup",
]

# Short aliases for the subtask keys stored in the output
SUBTASK_KEY = {
    "genai.cb.scheduling": "scheduling_us",
    "genai.cb.copy_blocks": "copy_blocks_us",
    "genai.cb.forward": "forward_us",
    "genai.cb.sample": "sample_us",
    "genai.cb.fork_free": "fork_free_us",
    "genai.cb.notify_dropped": "notify_dropped_us",
    "genai.cb.cleanup": "cleanup_us",
}

HW_COUNTERS = ["cpi", "cpu_load", "cpu_op_freq"]

# Matches: genai.cb.blk.<kind>:s=<seq>,pi=<pi>[,h=<hex>][,x=<int>]
_BLK_RE = re.compile(
    r"genai\.cb\.blk\.(?P<kind>\w+):s=(?P<s>\d+),pi=(?P<pi>\d+)"
    r"(?:,h=(?P<h>[0-9a-fA-F]+))?"
    r"(?:,x=(?P<x>\d+))?"
)

# Matches: genai.cb.config:blk_sz=<N>,blks=<M>
_CFG_RE = re.compile(r"genai\.cb\.config:blk_sz=(?P<blk_sz>\d+),blks=(?P<blks>\d+)")

# Matches: genai.cb.step.meta:a=<active>,w=<waiting>,s=<sched_tokens>,p=<0|1>
_META_RE = re.compile(r"genai\.cb\.step\.meta:a=(?P<a>\d+),w=(?P<w>\d+),s=(?P<s>\d+),p=(?P<p>\d+)")

# Matches: genai.cb.req.done:req=<req_id>,seq=<seq_id>,prompt=<N>,gen=<M>
_DONE_RE = re.compile(r"genai\.cb\.req\.done:req=(?P<req>\d+),seq=(?P<seq>\d+),prompt=(?P<prompt>\d+),gen=(?P<gen>\d+)")

# Matches: genai.cb.seq.step:s=<seq_id>,t=<tokens>,p=<0|1>
_SEQ_STEP_RE = re.compile(r"genai\.cb\.seq\.step:s=(?P<s>\d+),t=(?P<t>\d+),p=(?P<p>\d+)")


def _parse_blk_name(name: str):
    m = _BLK_RE.match(name)
    if not m:
        return None
    return {
        "kind": m.group("kind"),
        "seq_id": int(m.group("s")),
        "pi": int(m.group("pi")),
        "hash": int(m.group("h"), 16) if m.group("h") else 0,
        "extra": int(m.group("x")) if m.group("x") else 0,
    }


def _metadata_int(event: dict, key: str, default: int = 0) -> int:
    value = event.get("args", {}).get(key)
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip()
        if value.endswith(";"):
            value = value[:-1].rstrip()
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected integer ITT metadata '{key}' on event '{event.get('name', '')}', got {value!r}"
        ) from exc


def _parse_blk_event(event: dict):
    name = event.get("name", "")
    legacy = _parse_blk_name(name)
    if legacy is not None:
        return legacy

    prefix = "genai.cb.blk."
    args = event.get("args", {})
    if not name.startswith(prefix) or "seq_id" not in args or "physical_index" not in args:
        return None
    return {
        "kind": name[len(prefix) :],
        "seq_id": _metadata_int(event, "seq_id"),
        "pi": _metadata_int(event, "physical_index"),
        "hash": _metadata_int(event, "hash"),
        "extra": _metadata_int(event, "extra"),
    }


def _parse_config_event(event: dict):
    args = event.get("args", {})
    if "block_size_tokens" in args and "total_kv_blocks" in args:
        return _metadata_int(event, "block_size_tokens"), _metadata_int(event, "total_kv_blocks")
    match = _CFG_RE.match(event.get("name", ""))
    if match:
        return int(match.group("blk_sz")), int(match.group("blks"))
    return None


def _parse_step_metadata(event: dict):
    args = event.get("args", {})
    required = ("active", "waiting", "scheduled_tokens", "is_prefill")
    if all(key in args for key in required):
        return tuple(_metadata_int(event, key) for key in required)
    match = _META_RE.match(event.get("name", ""))
    if match:
        return tuple(int(match.group(key)) for key in ("a", "w", "s", "p"))
    return None


def _parse_done_event(event: dict):
    args = event.get("args", {})
    required = ("request_id", "seq_id", "prompt_len", "generated_len")
    if all(key in args for key in required):
        return {key: _metadata_int(event, key) for key in required}
    match = _DONE_RE.match(event.get("name", ""))
    if match:
        return {
            "request_id": int(match.group("req")),
            "seq_id": int(match.group("seq")),
            "prompt_len": int(match.group("prompt")),
            "generated_len": int(match.group("gen")),
        }
    return None


def _parse_sequence_step(event: dict):
    args = event.get("args", {})
    required = ("seq_id", "scheduled_tokens", "is_prefill")
    if all(key in args for key in required):
        return tuple(_metadata_int(event, key) for key in required)
    match = _SEQ_STEP_RE.match(event.get("name", ""))
    if match:
        return tuple(int(match.group(key)) for key in ("s", "t", "p"))
    return None


def _is_genai_category(category: str) -> bool:
    return category == "ov.genai" or category == "ov::genai" or category.startswith("ov::genai::")


def _replay_block_events(steps_out: list, blk_sorted: list, first_ts: float) -> list:
    """
    Replay block events chronologically and emit one memory snapshot per step.

    blk_sorted: [(ts_abs, parsed_event), ...] sorted by ts_abs
    Returns list of snapshot dicts compatible with trace_viz.html renderMemGrid().
    """
    phys_to_seqs: dict = defaultdict(set)  # pi → set of seq_ids
    seq_to_pis: dict = defaultdict(list)  # seq_id → ordered list of pi
    phys_hash: dict = {}  # pi → hash

    blk_idx = 0
    snapshots = []

    for step in steps_out:
        t0 = first_ts + step["ts_us"]
        t1 = t0 + step["duration_us"]

        # Consume all block events that fall inside this step's time window
        while blk_idx < len(blk_sorted):
            ts_abs, ev = blk_sorted[blk_idx]
            if ts_abs >= t1:
                break
            blk_idx += 1

            kind = ev["kind"]
            seq_id = ev["seq_id"]
            pi = ev["pi"]
            h = ev["hash"]
            extra = ev["extra"]

            if kind in ("alloc", "hit"):
                seq_to_pis[seq_id].append(pi)
                phys_to_seqs[pi].add(seq_id)
                if h:
                    phys_hash[pi] = h
            elif kind == "cow":
                # pi = old physical block, extra = new physical block
                new_pi = extra
                pis = seq_to_pis[seq_id]
                try:
                    pis[pis.index(pi)] = new_pi
                except ValueError:
                    pis.append(new_pi)
                phys_to_seqs[pi].discard(seq_id)
                if not phys_to_seqs[pi]:
                    del phys_to_seqs[pi]
                phys_to_seqs[new_pi].add(seq_id)
                phys_hash[new_pi] = phys_hash.get(pi, 0)
            elif kind == "hash":
                phys_hash[pi] = h
            elif kind == "free":
                # One free event per block (free_sequence emits per-block events)
                phys_to_seqs[pi].discard(seq_id)
                if not phys_to_seqs[pi]:
                    phys_to_seqs.pop(pi, None)
                pis = seq_to_pis.get(seq_id)
                if pis:
                    try:
                        pis.remove(pi)
                    except ValueError:
                        pass
                    if not pis:
                        seq_to_pis.pop(seq_id, None)

        # Emit snapshot after processing this step's events
        used = len(phys_to_seqs)
        shared = sum(1 for s in phys_to_seqs.values() if len(s) > 1)
        frag = round(shared / used, 4) if used > 0 else 0.0

        seqs_snap = {}
        for sid, pis in seq_to_pis.items():
            seqs_snap[str(sid)] = [
                {"pi": p, "r": len(phys_to_seqs.get(p, set())), "h": phys_hash.get(p, 0)} for p in pis
            ]

        snapshots.append(
            {
                "step_id": step["step_id"],
                "used_blocks": used,
                "shared_blocks": shared,
                "fragmentation": frag,
                "sequences": seqs_snap,
            }
        )

    return snapshots


# ── Loading ──────────────────────────────────────────────────────────────────


def load_perfetto(path: str) -> list:
    print(f"Loading {path} …", file=sys.stderr)
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    # Perfetto can wrap the array in {"traceEvents": [...]}
    if isinstance(raw, dict):
        return raw.get("traceEvents", [])
    return raw


# ── Core extraction ──────────────────────────────────────────────────────────


def extract(events: list) -> dict:
    # ── Partition events ─────────────────────────────────────────────────────
    genai_complete = [e for e in events if _is_genai_category(e.get("cat", "")) and e.get("ph") == "X"]
    hw_events = [e for e in events if e.get("ph") == "C" and e.get("name") in HW_COUNTERS]

    # Block state transition events (ph:X, name starts with "genai.cb.blk.")
    blk_raw = [e for e in genai_complete if e.get("name", "").startswith("genai.cb.blk.")]
    blk_sorted: list = []
    for e in blk_raw:
        parsed = _parse_blk_event(e)
        if parsed is not None:
            blk_sorted.append((e["ts"], parsed))
    blk_sorted.sort(key=lambda x: x[0])

    step_events = sorted([e for e in genai_complete if e["name"] == "genai.cb.step"], key=lambda e: e["ts"])
    if not step_events:
        cb_event_counts = defaultdict(int)
        for event in genai_complete:
            name = event.get("name", "")
            if name.startswith("genai.cb."):
                cb_event_counts[name.split(":", 1)[0]] += 1

        observed = ", ".join(f"{name} ({count})" for name, count in sorted(cb_event_counts.items())[:12])
        if cb_event_counts:
            raise ValueError(
                "Found GenAI continuous-batching events, but no complete "
                "'genai.cb.step' span. Observed: " + observed + ". "
                "The capture is missing the step parent event; it may have started "
                "mid-step or ended before the benchmark completed. Record from "
                "before the first request through the benchmark's 'Benchmark finished' "
                "message, and export complete events from the GenAI ITT domains."
            )
        raise ValueError(
            "No 'genai.cb.step' spans or other 'genai.cb.*' events found in "
            "complete GenAI-domain events. Verify the ITT-enabled GenAI library "
            "and continuous-batching benchmark are being captured."
        )

    print(
        f"  Found {len(step_events)} steps, {len(genai_complete)} genai events, "
        f"{len(hw_events)} hw counter samples, {len(blk_sorted)} block events",
        file=sys.stderr,
    )

    # ── Build task_id → event and parent_id → children index ─────────────────
    # task_id is a pair (d1, d2) but d2 is always 0 in practice — use d1.
    by_task_id: dict = {}
    by_parent_id: dict = defaultdict(list)
    for e in genai_complete:
        tid = e["args"].get("task_id.d1")
        pid = e["args"].get("parent_id.d1", 0)
        if tid is not None:
            by_task_id[tid] = e
        by_parent_id[pid].append(e)

    # ── Parse config event (emitted once on first step) ──────────────────────
    cfg_blk_sz = 0  # block_size_tokens from genai.cb.config
    cfg_blk_total = 0  # total_kv_blocks from genai.cb.config
    for e in genai_complete:
        config = _parse_config_event(e)
        if config is not None:
            cfg_blk_sz, cfg_blk_total = config
            break

    # ── Parse req.done events → keyed by seq_id ──────────────────────────────
    # {seq_id → {req_id, prompt_len, gen_len, step_id_when_done}}
    req_done: dict = {}
    for e in genai_complete:
        done = _parse_done_event(e)
        if done is not None:
            seq_id = done["seq_id"]
            req_done[seq_id] = {
                "req_id": done["request_id"],
                "prompt_len": done["prompt_len"],
                "gen_len": done["generated_len"],
                "ts": e["ts"],
            }

    # ── Step timeline ─────────────────────────────────────────────────────────
    first_ts = step_events[0]["ts"]
    steps_out = []

    for step_id, se in enumerate(step_events):
        step_task_id = se["args"]["task_id.d1"]
        ts_us = int(se["ts"] - first_ts)
        duration_us = int(se["dur"])

        # Direct children of this step
        direct_children = by_parent_id.get(step_task_id, [])

        # Parse step.meta from direct children
        active, waiting, sched_tokens, is_prefill = 0, 0, 0, False
        for child in direct_children:
            metadata = _parse_step_metadata(child)
            if metadata is not None:
                active, waiting, sched_tokens, is_prefill_value = metadata
                is_prefill = bool(is_prefill_value)
                break

        # Gather subtask durations — also look one level deeper (copy_blocks
        # is a child of genai.cb.scheduling, not directly of step)
        subtasks = {v: 0 for v in SUBTASK_KEY.values()}
        subtasks["unaccounted_us"] = 0

        accounted = 0
        for child in direct_children:
            key = SUBTASK_KEY.get(child["name"])
            if key:
                dur = int(child["dur"])
                subtasks[key] += dur
                accounted += dur

                # For scheduling: also grab copy_blocks from its children
                if child["name"] == "genai.cb.scheduling":
                    child_tid = child["args"].get("task_id.d1")
                    if child_tid is not None:
                        for gc in by_parent_id.get(child_tid, []):
                            cb_key = SUBTASK_KEY.get(gc["name"])
                            if cb_key:
                                subtasks[cb_key] += int(gc["dur"])
                                # copy_blocks is already inside scheduling_us
                                # so don't double-count in accounted

        subtasks["unaccounted_us"] = max(0, duration_us - accounted)

        tps = int(sched_tokens / duration_us * 1_000_000) if duration_us > 0 and sched_tokens > 0 else 0

        steps_out.append(
            {
                "step_id": step_id,
                "ts_us": ts_us,
                "active": active,
                "waiting": waiting,
                "total": active + waiting,
                "cache_usage": 0.0,
                "cache_bytes": 0,
                "total_tokens": 0,
                "is_prefill": is_prefill,
                "duration_us": duration_us,
                "preemption_count": 0,
                "step_tokens": sched_tokens,
                "throughput_tps": tps,
                "subtasks": subtasks,
            }
        )

    # ── Hardware counter correlation ─────────────────────────────────────────
    # Sort HW samples per counter; for each step window compute the average.
    hw_by_name: dict = defaultdict(list)
    for e in hw_events:
        val = next(iter(e.get("args", {}).values()), None)
        if val is not None:
            hw_by_name[e["name"]].append((e["ts"], float(val)))
    for k in hw_by_name:
        hw_by_name[k].sort()

    for s in steps_out:
        t0 = first_ts + s["ts_us"]
        t1 = t0 + s["duration_us"]
        hw_avg = {}
        for name, samples in hw_by_name.items():
            ts_list = [x[0] for x in samples]
            lo = bisect.bisect_left(ts_list, t0)
            hi = bisect.bisect_right(ts_list, t1)
            vals = [samples[i][1] for i in range(lo, hi)]
            hw_avg[name] = round(sum(vals) / len(vals), 4) if vals else None
        s["hw"] = hw_avg

    # ── Block event replay → memory snapshots ───────────────────────────────
    memory_snapshots = []
    if blk_sorted:
        print("  Replaying block events …", file=sys.stderr)
        memory_snapshots = _replay_block_events(steps_out, blk_sorted, first_ts)

        # Prefer ITT-emitted config; fall back to observed max pi
        total_kv_blocks = (
            cfg_blk_total if cfg_blk_total > 0 else (max((ev["pi"] for _, ev in blk_sorted), default=0) + 1)
        )

        # Prefix hits happen during request admission; later decode allocations
        # are not cache lookup misses and must not dilute this rate.
        hit_count = sum(1 for _, ev in blk_sorted if ev["kind"] == "hit")
        alloc_count = sum(1 for _, ev in blk_sorted if ev["kind"] == "alloc")
        if cfg_blk_sz > 0 and req_done:
            prompt_lengths_by_request = {}
            for info in req_done.values():
                req_id = info["req_id"]
                prompt_lengths_by_request[req_id] = max(
                    info["prompt_len"],
                    prompt_lengths_by_request.get(req_id, 0),
                )
            prompt_block_count = sum(
                (prompt_len + cfg_blk_sz - 1) // cfg_blk_sz for prompt_len in prompt_lengths_by_request.values()
            )
        else:
            prompt_block_count = 0

        if prompt_block_count > 0:
            prefix_cache_hit_rate = round(min(hit_count, prompt_block_count) / prompt_block_count, 4)
        else:
            prefix_cache_hit_rate = (
                round(hit_count / (hit_count + alloc_count), 4) if (hit_count + alloc_count) > 0 else 0.0
            )

        # Hits per sequence (for seq_summaries)
        hit_blocks_per_seq: dict = defaultdict(int)
        for _, ev in blk_sorted:
            if ev["kind"] == "hit":
                hit_blocks_per_seq[ev["seq_id"]] += 1

        # Peak cache usage from snapshots
        peak_used = max((s["used_blocks"] for s in memory_snapshots), default=0)
        peak_cache_usage = round(peak_used / total_kv_blocks, 4) if total_kv_blocks > 0 else 0.0

        # Back-fill per-step cache_usage from snapshots
        for step, snap in zip(steps_out, memory_snapshots):
            step["cache_usage"] = round(snap["used_blocks"] / total_kv_blocks, 4) if total_kv_blocks > 0 else 0.0
    else:
        total_kv_blocks = cfg_blk_total
        prefix_cache_hit_rate = 0.0
        peak_cache_usage = 0.0
        hit_blocks_per_seq = {}

    block_size_tokens = cfg_blk_sz  # 0 if config event not captured yet
    block_meta = {"block_size_tokens": block_size_tokens, "total_kv_blocks": total_kv_blocks}

    # ── seq_summaries from req.done events + memory snapshots ────────────────
    # first_step / last_step per seq_id from snapshot presence
    seq_first_step: dict = {}
    seq_last_step: dict = {}
    for snap in memory_snapshots:
        sid_int_set = {int(k) for k in snap["sequences"]}
        for sid in sid_int_set:
            if sid not in seq_first_step:
                seq_first_step[sid] = snap["step_id"]
            seq_last_step[sid] = snap["step_id"]

    seq_summaries = []
    for seq_id, info in req_done.items():
        prompt_len = info["prompt_len"]
        gen_len = info["gen_len"]
        hit_blks = hit_blocks_per_seq.get(seq_id, 0)
        hit_tokens = min(hit_blks * block_size_tokens, prompt_len) if block_size_tokens > 0 else 0
        hit_pct = round(hit_tokens / prompt_len, 3) if prompt_len > 0 else 0.0
        first_step = seq_first_step.get(seq_id, 0)
        last_step = seq_last_step.get(seq_id, first_step)
        seq_summaries.append(
            {
                "seq_id": seq_id,
                "req_id": info["req_id"],
                "prompt_len": prompt_len,
                "gen_len": gen_len,
                "first_step": first_step,
                "last_step": last_step,
                "step_span": last_step - first_step + 1,
                "prefix_hit_tokens": hit_tokens,
                "prefix_hit_pct": hit_pct,
            }
        )
    seq_summaries.sort(key=lambda x: x["seq_id"])

    request_ids = {summary["req_id"] for summary in seq_summaries}
    request_ids_with_prefix_hits = {
        summary["req_id"] for summary in seq_summaries if hit_blocks_per_seq.get(summary["seq_id"], 0) > 0
    }
    prefix_cache_request_hit_rate = len(request_ids_with_prefix_hits) / len(request_ids) if request_ids else 0.0

    # ── Per-sequence prefill window from block events ─────────────────────────
    # Using step-level is_prefill is wrong: it's True whenever ANY sequence in
    # the batch is prefilling, so a decode sequence looks like it's prefilling
    # again whenever a new request joins.
    #
    # Correct approach: the ceil(prompt_len / block_size_tokens)-th block event
    # (hit or alloc) for a sequence is the last block allocated during its own
    # prefill.  Everything after that event's step is decode.
    seq_block_ts: dict = defaultdict(list)  # seq_id → sorted list of ts for alloc+hit
    for ts_abs, ev in blk_sorted:
        if ev["kind"] in ("alloc", "hit"):
            seq_block_ts[ev["seq_id"]].append(ts_abs)

    seq_prefill_end_step: dict = {}
    for seq_id, info in req_done.items():
        prompt_len = info["prompt_len"]
        if block_size_tokens > 0 and prompt_len > 0:
            prefill_blocks = math.ceil(prompt_len / block_size_tokens)
        else:
            prefill_blocks = 1
        events = seq_block_ts.get(seq_id, [])
        # Timestamp of the last block allocated during this seq's prefill phase
        idx = min(prefill_blocks, len(events)) - 1
        if idx >= 0:
            prefill_last_ts = events[idx]
        else:
            prefill_last_ts = first_ts
        # Map that timestamp back to a step_id
        end_step = seq_first_step.get(seq_id, 0)
        for step in steps_out:
            t0 = first_ts + step["ts_us"]
            t1 = t0 + step["duration_us"]
            if t0 <= prefill_last_ts < t1:
                end_step = step["step_id"]
                break
        seq_prefill_end_step[seq_id] = end_step

    # ── Reconstruct sequences dict from memory_snapshots ─────────────────────
    # Drives swimlane and Gantt charts.
    step_by_id = {s["step_id"]: s for s in steps_out}

    # ── Parse per-sequence step events (seq.step) ─────────────────────────────
    # seq_step_by_step[step_id][seq_id] = {"tokens": N, "is_prefill": bool}
    # These come from genai.cb.seq.step ITT events emitted in scheduler.hpp
    # inside the genai.cb.scheduling scope, so they are grandchildren of
    # genai.cb.step rather than direct children.
    seq_step_by_step: dict = {}
    for step_id, se in enumerate(step_events):
        step_task_id = se["args"]["task_id.d1"]
        direct_children = by_parent_id.get(step_task_id, [])
        seq_step_map = {}
        for child in direct_children:
            # seq.step events live inside genai.cb.scheduling (grandchildren of step)
            if child["name"] == "genai.cb.scheduling":
                sched_tid = child["args"].get("task_id.d1")
                if sched_tid is not None:
                    for gc in by_parent_id.get(sched_tid, []):
                        seq_step = _parse_sequence_step(gc)
                        if seq_step is not None:
                            sid, tokens, is_prefill = seq_step
                            seq_step_map[sid] = {
                                "tokens": tokens,
                                "is_prefill": bool(is_prefill),
                            }
            else:
                # Also accept direct children for forward-compatibility
                seq_step = _parse_sequence_step(child)
                if seq_step is not None:
                    sid, tokens, is_prefill = seq_step
                    seq_step_map[sid] = {
                        "tokens": tokens,
                        "is_prefill": bool(is_prefill),
                    }
        if seq_step_map:
            seq_step_by_step[step_id] = seq_step_map
            # Re-derive step-level is_prefill from authoritative per-seq data
            steps_out[step_id]["is_prefill"] = any(v["is_prefill"] for v in seq_step_map.values())

    sequences_dict: dict = {}
    for snap in memory_snapshots:
        step_id = snap["step_id"]
        for sid_str in snap["sequences"]:
            sid = int(sid_str)
            if sid_str not in sequences_dict:
                done_info = req_done.get(sid, {})
                sequences_dict[sid_str] = {
                    "req_id": done_info.get("req_id", sid),
                    "prompt_len": done_info.get("prompt_len", 0),
                    "max_gen_len": done_info.get("gen_len", 0),
                    "first_step": step_id,
                    "last_step": step_id,
                    "steps": [],
                    "status": [],
                    "is_prefill": [],
                    "sched_tokens": [],
                }
            entry = sequences_dict[sid_str]
            entry["last_step"] = step_id
            entry["steps"].append(step_id)
            entry["status"].append("RUNNING")
            # Use authoritative per-seq data from seq.step events when available;
            # fall back to block-event heuristic for older traces.
            seq_step = seq_step_by_step.get(step_id, {}).get(sid)
            if seq_step is not None:
                entry["is_prefill"].append(seq_step["is_prefill"])
                entry["sched_tokens"].append(seq_step["tokens"])
            else:
                pf_end = seq_prefill_end_step.get(sid, entry["first_step"])
                entry["is_prefill"].append(step_id <= pf_end)
                entry["sched_tokens"].append(step_by_id.get(step_id, {}).get("step_tokens", 0))

    # Mark the last step of each finished sequence as FINISHED
    for sid_str, entry in sequences_dict.items():
        sid = int(sid_str)
        if sid in req_done and entry["status"]:
            entry["status"][-1] = "FINISHED"

    # ── Events list: FINISH events from req.done timestamps ──────────────────
    # Map each finished sequence to the step it completed in.
    events_out = []
    for seq_id, info in req_done.items():
        done_ts = info["ts"]
        finish_step = steps_out[-1]["step_id"] if steps_out else 0
        for step in steps_out:
            t0 = first_ts + step["ts_us"]
            t1 = t0 + step["duration_us"]
            if t0 <= done_ts < t1:
                finish_step = step["step_id"]
                break
        events_out.append(
            {
                "event": "FINISH",
                "step_id": finish_step,
                "req_id": info["req_id"],
                "seq_id": seq_id,
            }
        )

    # ── Histograms derived from per-step data ─────────────────────────────────
    batch_hist: dict = defaultdict(int)
    sched_hist: dict = defaultdict(int)
    for s in steps_out:
        if s["active"] > 0:
            batch_hist[s["active"]] += 1
        if s["step_tokens"] > 0:
            sched_hist[s["step_tokens"]] += 1

    sched_tokens_hist = [{"tokens": t, "count": c} for t, c in sorted(sched_hist.items())]

    # ── Aggregate metrics ────────────────────────────────────────────────────
    durations = sorted([s["duration_us"] for s in steps_out if s["duration_us"] > 0])

    def percentile(vals, pct):
        if not vals:
            return 0
        idx = max(0, int(len(vals) * pct / 100) - 1)
        return int(vals[idx])

    avg_sub = {}
    for key in list(SUBTASK_KEY.values()) + ["unaccounted_us"]:
        vals = [s["subtasks"][key] for s in steps_out]
        avg_sub[key] = int(sum(vals) / len(vals)) if vals else 0

    total_dur = sum(s["duration_us"] for s in steps_out)
    total_fwd = sum(s["subtasks"].get("forward_us", 0) for s in steps_out)
    forward_fraction = round(total_fwd / total_dur, 4) if total_dur > 0 else 0.0

    metrics = {
        "total_steps": len(steps_out),
        "total_preemptions": 0,
        "total_ooms": 0,
        "total_finishes": len(seq_summaries),
        "peak_cache_usage": peak_cache_usage,
        "prefix_cache_hit_rate": prefix_cache_hit_rate,
        "prefix_cache_requests_with_hits": len(request_ids_with_prefix_hits),
        "prefix_cache_request_count": len(request_ids),
        "prefix_cache_request_hit_rate": round(prefix_cache_request_hit_rate, 4),
        "avg_step_duration_us": int(sum(durations) / len(durations)) if durations else 0,
        "p50_step_duration_us": percentile(durations, 50),
        "p99_step_duration_us": percentile(durations, 99),
        "preemption_rate_per_step": [0] * len(steps_out),
        "avg_subtask_us": avg_sub,
        "forward_fraction": forward_fraction,
    }

    return {
        "steps": steps_out,
        "sequences": sequences_dict,
        "seq_summaries": seq_summaries,
        "events": events_out,
        "memory_snapshots": memory_snapshots,
        "batch_hist": dict(sorted(batch_hist.items())),
        "sched_tokens_hist": sched_tokens_hist,
        "metrics": metrics,
        "block_meta": block_meta,
        "source": "perfetto",
        "subtask_legend": SUBTASK_KEY,
        "hw_counter_names": list(hw_by_name.keys()),
    }


# ── Entry point ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Perfetto/Chrome Trace JSON to analysis.json for trace_viz.html."
    )
    parser.add_argument("perfetto", help="Input Perfetto JSON file (Chrome Trace Event format)")
    parser.add_argument("--out", default="analysis.json", help="Output analysis JSON file (default: analysis.json)")
    parser.add_argument("--model", default="", help="Model name to embed in the report")
    parser.add_argument("--device", default="", help="Inference device (e.g. CPU, GPU)")
    parser.add_argument("--notes", default="", help="Free-form notes to embed in the report")
    args = parser.parse_args()

    events = load_perfetto(args.perfetto)
    print(f"  Total events: {len(events)}", file=sys.stderr)

    print("Extracting CB step data …", file=sys.stderr)
    result = extract(events)

    result["run_info"] = {
        "model": args.model or os.path.basename(args.perfetto),
        "device": args.device,
        "notes": args.notes,
        "trace_file": os.path.basename(args.perfetto),
        "analyzed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "block_size_tokens": 0,
        "total_kv_blocks": 0,
        "source": "perfetto",
    }

    print(f"Writing {args.out} …", file=sys.stderr)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, separators=(",", ":"))

    m = result["metrics"]
    sub = m.get("avg_subtask_us", {})
    bm = result.get("block_meta", {})
    snaps = result.get("memory_snapshots", [])
    print(
        f"\nSummary\n"
        f"  Steps:              {m['total_steps']}\n"
        f"  p50 step (µs):      {m['p50_step_duration_us']:,}\n"
        f"  p99 step (µs):      {m['p99_step_duration_us']:,}\n"
        f"  avg forward (µs):   {sub.get('forward_us', 0):,}  "
        f"({m['forward_fraction']:.1%} of step)\n"
        f"  avg schedule (µs):  {sub.get('scheduling_us', 0):,}\n"
        f"  avg sample (µs):    {sub.get('sample_us', 0):,}\n"
        f"  avg overhead (µs):  {sub.get('unaccounted_us', 0):,}\n"
        f"  Block snapshots:    {len(snaps)}\n"
        f"  Block size (tok):   {bm.get('block_size_tokens', 0)}\n"
        f"  Total KV blocks:    {bm.get('total_kv_blocks', 0)}\n"
        f"  Peak cache usage:   {m['peak_cache_usage']:.1%}\n"
        f"  Prompt-block hit rate: {m['prefix_cache_hit_rate']:.1%}\n"
        f"  Requests with hits: {m['prefix_cache_requests_with_hits']}/{m['prefix_cache_request_count']} "
        f"({m['prefix_cache_request_hit_rate']:.1%})\n"
        f"  Requests finished:  {m['total_finishes']}\n"
        f"  Seq summaries:      {len(result.get('seq_summaries', []))}\n",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
