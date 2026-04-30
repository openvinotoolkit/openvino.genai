#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT_DIR/build/tools/continuous_batching/benchmark/continuous_batching_benchmark"

MODEL_PATH="${MODEL_PATH:-$HOME/llm_irs/WW05_llm-optimum_2026.0.0-20947/qwen3-8b/pytorch/ov/OV_FP16-4BIT_DEFAULT}"
# DATASET_PATH="${DATASET_PATH:-/home/ceciliapeng/llm_irs/ShareGPT_V3_unfiltered_cleaned_split1.json}"
DATASET_PATH="${DATASET_PATH:-/home/ceciliapeng/llm_irs/xattn-demo.16k.json}"
DEVICE_CONFIG_FILE="${DEVICE_CONFIG_FILE:-$ROOT_DIR/ov.fp16.config}"
DEVICE="${DEVICE:-GPU}"
CACHE_SIZE="${CACHE_SIZE:-4}"
NUM_PROMPTS="${NUM_PROMPTS:-3}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-32768}"
MAX_OUTPUT_LEN="${MAX_OUTPUT_LEN:-2048}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-0}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4096}"
XATTENTION_BLOCK_SIZE="${XATTENTION_BLOCK_SIZE:-256}"
XATTENTION_BLOCK_THRESHOLD="${XATTENTION_BLOCK_THRESHOLD:-100.0}"

usage() {
    cat <<'EOF'

Usage: benchmark_cb_3cases.sh [--backend cm|ocl|both] [--route multi|split|both] [--xattention_block_size <128|256>] [--xattention_block_threshold <float>] [--profile] [--compare] [--output <dir>] [--help]

Runs continuous batching benchmark cases:
    --backend:   Select backend to run: cm, ocl, or both (default: both)
    --route:     Set route mode for cm backend: multi, split, or both (default: both)
    --xattention_block_size: XATTENTION block size for cm path (default: 256)
    --xattention_block_threshold: XATTENTION threshold for cm path (default: 100.0). Ignored for ocl.
    --profile:   Run via cliloader and dump clintercept traces to <output>/<case>/profiling/
    --compare:   Compare logs/profiling in <output>/ folders and generate a CSV report if any of the following exist:
                   ocl and cm route/xattention variants with OV_GPU_CM_PA_FORCE_LOCKABLE_MAPPING={0,1}
    --output:    Output directory for logs/profiling/report CSV (default: temp). Relative paths are under script root.
    --help:      Show this help message

The benchmark binary is executed as follows, with environment variables controlling the configuration:
    OV_GPU_CM_MIXED_ROUTE_MODE=multi|split (only for cm)
    OV_GPU_CM_PA_FORCE_LOCKABLE_MAPPING=0|1 (only for cm)
    cd ~/openvino.genai/
    ./build/bin/continuous_batching_benchmark  --device GPU -m "$MODEL_PATH" --dataset "$DATASET_PATH" \
    --cache_size "$CACHE_SIZE" -n "$NUM_PROMPTS"  \
    --device_config "$DEVICE_CONFIG_FILE" \
    --max_input_len "$MAX_INPUT_LEN" --max_output_len "$MAX_OUTPUT_LEN" \
    --max_total_tokens "$MAX_TOTAL_TOKENS" --max_batch_size "$MAX_BATCH_SIZE" \
    --use_xattention --xattention_block_size "$XATTENTION_BLOCK_SIZE" --xattention_threshold "$XATTENTION_BLOCK_THRESHOLD"  # cm backend only

Notes:
    1. kvcache_blocksize is internal to backend and not configurable via this script.
    2. ocl backend does not use xattention.
    3. when xattention is enabled, xattention_block_size can be 128 or 256.
        4. cm backend always runs with --use_xattention.
    5. output layout:
            <output>/ocl/
            <output>/cm/use_xattention.on.xbls_<xattention_block_size>.threshold_<xattention_block_threshold>/route_<multi|split>.force_lockable_mapping_<0|1>/

Environment overrides:
    MODEL_PATH          Model path
    DATASET_PATH        Dataset json path
    DEVICE_CONFIG_FILE  Path to device config file (JSON content)
    DEVICE              Device name (default: GPU)
    CACHE_SIZE          Cache size (default: 4)
    NUM_PROMPTS         Number of prompts (default: 3)
    MAX_INPUT_LEN       Max input length (default: 32768)
    MAX_OUTPUT_LEN      Max output length (default: 2048)
    MAX_TOTAL_TOKENS    Max total input+output tokens accepted from dataset entries; 0 disables this filter (default: 0)
    MAX_BATCH_SIZE      Max batch size (default: 4096)
    XATTENTION_BLOCK_SIZE       XATTENTION block size for cm path (default: 256)
    XATTENTION_BLOCK_THRESHOLD  XATTENTION threshold for cm path (default: 100.0). Ignored for ocl.
    OUTPUT_DIR          Output directory (default: temp)

Examples:
    bash benchmark_cb_3cases.sh --backend cm --route split
    bash benchmark_cb_3cases.sh --backend both --route both --profile
    bash benchmark_cb_3cases.sh --compare
    bash benchmark_cb_3cases.sh --output temp.myrun --backend both --route both --profile
    bash benchmark_cb_3cases.sh --backend cm --route both --xattention_block_size 128 --xattention_block_threshold 0.9
    MODEL_PATH=/path/to/model DATASET_PATH=/path/to/data.json bash benchmark_cb_3cases.sh --backend ocl
EOF
}


# (Old for-loop argument parser removed; only new while-loop parser is used)


# Default options
BACKEND="both"
ROUTE="both"
DO_COMPARE=0
PROFILE=0
OUTPUT_DIR="${OUTPUT_DIR:-temp}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            BACKEND="$2"; shift 2;;
        --route)
            ROUTE="$2"; shift 2;;
        --xattention_block_size)
            XATTENTION_BLOCK_SIZE="$2"; shift 2;;
        --xattention_block_threshold)
            XATTENTION_BLOCK_THRESHOLD="$2"; shift 2;;
        --compare)
            DO_COMPARE=1; shift;;
        --profile)
            PROFILE=1; shift;;
        --output)
            OUTPUT_DIR="$2"; shift 2;;
        --help|-h)
            usage; exit 0;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            usage >&2
            exit 2;;
    esac
done

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: Benchmark binary not found or not executable: $BIN"
    exit 1
fi

if [[ ! -f "$DEVICE_CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $DEVICE_CONFIG_FILE"
    exit 1
fi

if [[ "$PROFILE" == "1" ]] && ! command -v cliloader >/dev/null 2>&1; then
    echo "ERROR: --profile requested but cliloader not found in PATH"
    exit 1
fi

if [[ ("$BACKEND" == "cm" || "$BACKEND" == "both") && "$XATTENTION_BLOCK_SIZE" != "128" && "$XATTENTION_BLOCK_SIZE" != "256" ]]; then
    echo "ERROR: --xattention_block_size must be 128 or 256 for cm backend"
    exit 1
fi

DEVICE_CONFIG_JSON="$(cat "$DEVICE_CONFIG_FILE")"



# Output directory for logs/profiling/report
if [[ "$OUTPUT_DIR" = /* ]]; then
    LOG_DIR="$OUTPUT_DIR"
else
    LOG_DIR="$ROOT_DIR/$OUTPUT_DIR"
fi
# Normalize trailing slash (except root) so path-prefix parsing in --compare works.
if [[ "$LOG_DIR" != "/" ]]; then
    LOG_DIR="${LOG_DIR%/}"
fi
mkdir -p "$LOG_DIR"

# If --compare is set, generate CSV report and exit
if [[ "$DO_COMPARE" == "1" ]]; then
    REPORT_CSV="$LOG_DIR/compare_report.csv"
    mkdir -p "$(dirname "$REPORT_CSV")"
    echo "case_id,case_name,xattention_block_size,route,force_lockable_mapping,backend,xattention_threshold,input_throughput,output_throughput,mean_ttft,mean_tpot,benchmark_duration,total_input_tokens,total_output_tokens,log_path,profiling_dir,sum_kernel_ms,kernels" > "$REPORT_CSV"

    parse_kernel_summary() {
        local trace_file="$1"
        local backend="$2"
        python3 - "$trace_file" "$backend" <<'PY'
import json
import math
import re
import sys
from collections import defaultdict

trace_file = sys.argv[1]
backend = sys.argv[2]
prefixes = {
    "cm": ["pa_kv_cache_update", "pa_multi_", "pa_single_"],
    "ocl": ["pa_kv_cache_update", "sdpa_micro", "paged_attention_opt__"],
}

def events_of(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("traceEvents", "events", "Records", "records"):
            v = data.get(k)
            if isinstance(v, list):
                return v
    return []

def get_name(ev):
    for k in ("name", "kernel_name", "KernelName", "kernelName"):
        v = ev.get(k)
        if isinstance(v, str) and v:
            return v
    args = ev.get("args")
    if isinstance(args, dict):
        for k in ("name", "kernel", "kernel_name", "KernelName"):
            v = args.get(k)
            if isinstance(v, str) and v:
                return v
    return None

def get_ms(ev):
    for k, scale in (
        ("duration_ms", 1.0),
        ("dur_ms", 1.0),
        ("dur", 1e-3),
        ("duration_us", 1e-3),
        ("duration_ns", 1e-6),
        ("dur_ns", 1e-6),
        ("execTimeNs", 1e-6),
        ("time_ns", 1e-6),
        ("time_us", 1e-3),
    ):
        v = ev.get(k)
        if isinstance(v, (int, float)):
            return float(v) * scale
    return None

try:
    data = json.load(open(trace_file))
except Exception:
    print("\t")
    sys.exit(0)

stats = defaultdict(lambda: [0, 0.0])
for ev in events_of(data):
    if not isinstance(ev, dict):
        continue
    name = get_name(ev)
    if not name or not any(p in name for p in prefixes.get(backend, [])):
        continue
    ms = get_ms(ev)
    if ms is None or not math.isfinite(ms) or ms < 0:
        continue
    stats[name][0] += 1
    stats[name][1] += ms

sum_kernel_ms = 0.0
kernel_stats = defaultdict(lambda: [0, 0.0])
for name in sorted(stats.keys()):
    calls, total = stats[name]
    sum_kernel_ms += total
    # simplify kernel name for report: remove huge numeric hash and __sa suffix
    simplified = re.sub(r'_[0-9]+__sa$', '', name)
    simplified = re.sub(r'__sa$', '', simplified)
    kernel_stats[simplified][0] += calls
    kernel_stats[simplified][1] += total

if kernel_stats:
    items = []
    for k in sorted(kernel_stats.keys()):
        calls, total = kernel_stats[k]
        items.append(f"{k}(calls={calls},sum_ms={total:.6f})")
    print(f"{sum_kernel_ms:.6f}\t{' | '.join(items)}")
else:
    print("\t")
PY
    }

    case_id=0
    while IFS= read -r log_file; do
        folder="$(dirname "$log_file")"
        rel_dir="${folder#${LOG_DIR}/}"

        key="${rel_dir//\//_}"
        key="${key//./_}"

        xattention_block_size="na"
        xattention_threshold="na"
        route="na"
        backend="${rel_dir%%/*}"
        force_lockable="na"

        if [[ "$backend" == "ocl" ]]; then
            route="multi"
        elif [[ "$rel_dir" =~ ^cm/use_xattention\.off/route_([^./]+)\.force_lockable_mapping_([^/]+)$ ]]; then
            route="${BASH_REMATCH[1]}"
            force_lockable="${BASH_REMATCH[2]}"
        elif [[ "$rel_dir" =~ ^cm/use_xattention\.on\.xbls_([^/]+)\.threshold_([^/]+)/route_([^./]+)\.force_lockable_mapping_([^/]+)$ ]]; then
            xattention_block_size="${BASH_REMATCH[1]}"
            xattention_threshold="${BASH_REMATCH[2]}"
            route="${BASH_REMATCH[3]}"
            force_lockable="${BASH_REMATCH[4]}"
        fi

        profiling_dir="$folder/profiling"

        input_tp=""
        output_tp=""
        mean_ttft=""
        mean_tpot=""
        benchmark_duration=""
        total_input_tokens=""
        total_output_tokens=""
        if [[ -f "$log_file" ]]; then
            input_tp=$(grep -m1 "Input throughput:" "$log_file" | awk -F': ' '{print $2}')
            output_tp=$(grep -m1 "Output throughput:" "$log_file" | awk -F': ' '{print $2}')
            mean_ttft=$(grep -m1 "Mean TTFT:" "$log_file" | awk -F': ' '{print $2}')
            mean_tpot=$(grep -m1 "Mean TPOT:" "$log_file" | awk -F': ' '{print $2}')
            benchmark_duration=$(grep -m1 "Benchmark duration:" "$log_file" | awk -F': ' '{print $2}')
            total_input_tokens=$(grep -m1 "Total number of input tokens:" "$log_file" | awk -F': ' '{print $2}')
            total_output_tokens=$(grep -m1 "Total number of output tokens:" "$log_file" | awk -F': ' '{print $2}')
        fi

        trace_file="$profiling_dir/clintercept_trace.json"
        if [[ ! -f "$trace_file" ]]; then
            trace_file="$folder/clintercept_trace.json"
        fi

        sum_kernel_ms=""
        kernels=""
        if [[ -f "$trace_file" ]]; then
            parsed="$(parse_kernel_summary "$trace_file" "$backend")"
            sum_kernel_ms="${parsed%%$'\t'*}"
            kernels="${parsed#*$'\t'}"
        fi

        if [[ "$log_file" == "$ROOT_DIR/"* ]]; then
            rel_log_path="${log_file#${ROOT_DIR}/}"
        else
            rel_log_path="${log_file#${LOG_DIR}/}"
        fi
        if [[ "$profiling_dir" == "$ROOT_DIR/"* ]]; then
            rel_profiling_dir="${profiling_dir#${ROOT_DIR}/}"
        else
            rel_profiling_dir="${profiling_dir#${LOG_DIR}/}"
        fi

        if [[ -n "$kernels" ]]; then
            kernel_multiline="${kernels// | /$'\n'}"
            # Quote the kernels field so embedded newlines stay in the same CSV cell.
            echo "$case_id,$key,$xattention_block_size,$route,$force_lockable,$backend,$xattention_threshold,$input_tp,$output_tp,$mean_ttft,$mean_tpot,$benchmark_duration,$total_input_tokens,$total_output_tokens,$rel_log_path,$rel_profiling_dir,$sum_kernel_ms,\"$kernel_multiline\"" >> "$REPORT_CSV"
        else
            echo "$case_id,$key,$xattention_block_size,$route,$force_lockable,$backend,$xattention_threshold,$input_tp,$output_tp,$mean_ttft,$mean_tpot,$benchmark_duration,$total_input_tokens,$total_output_tokens,$rel_log_path,$rel_profiling_dir,$sum_kernel_ms," >> "$REPORT_CSV"
        fi
        case_id=$((case_id + 1))
    done < <(
        find "$LOG_DIR" -type f -name "benchmark.log" \
            | awk -v ocl_prefix="$LOG_DIR/ocl/" '{k=(index($0, ocl_prefix)==1)?0:1; print k "\t" $0}' \
            | sort -k1,1n -k2,2 \
            | cut -f2-
    )

    if [[ "$case_id" == "0" ]]; then
        echo "WARNING: No benchmark.log found under output folder: $LOG_DIR"
    fi

    if [[ -s "$REPORT_CSV" ]]; then
        echo "Comparison report generated: $REPORT_CSV"
    fi
    exit 0
fi

COMMON_ARGS=(
    --device "$DEVICE"
    -m "$MODEL_PATH"
    --dataset "$DATASET_PATH"
    --device_config "$DEVICE_CONFIG_JSON"
    --cache_size "$CACHE_SIZE"
    --num_prompts "$NUM_PROMPTS"
    --max_input_len "$MAX_INPUT_LEN"
    --max_output_len "$MAX_OUTPUT_LEN"
    --max_total_tokens "$MAX_TOTAL_TOKENS"
    --max_batch_size "$MAX_BATCH_SIZE"
)

run_case() {
    local case_name="$1"
    local impl="$2"
    local route_mode="$3"
    local force_lockable="$4"
    local xattention_block_size="$5"
    local xattention_block_threshold="$6"
    shift 6

    local case_dir=""
    if [[ "$impl" == "ocl" ]]; then
        case_dir="$LOG_DIR/ocl"
    else
        local cm_mode_dir="use_xattention.on.xbls_${xattention_block_size}.threshold_${xattention_block_threshold}"
        case_dir="$LOG_DIR/cm/${cm_mode_dir}/route_${route_mode}.force_lockable_mapping_${force_lockable}"
    fi
    local log_file="$case_dir/benchmark.log"
    local dump_dir="$case_dir/profiling"

    mkdir -p "$case_dir"

    echo "============================================================"
    echo "Running case: $case_name"
    echo "backend=$impl, OV_GPU_CM_MIXED_ROUTE_MODE=$route_mode, OV_GPU_CM_PA_FORCE_LOCKABLE_MAPPING=$force_lockable, xattention_block_size=${xattention_block_size:-na}, xattention_threshold=${xattention_block_threshold:-na}"
    echo "Log file: $log_file"
    if [[ "$PROFILE" == "1" ]]; then
        echo "Profiling dump dir: $dump_dir"
    fi

    (
        # export OV_VERBOSE=4
        if [[ "$impl" == "cm" ]]; then
            export OV_GPU_CM_MIXED_ROUTE_MODE="$route_mode"
            export OV_GPU_CM_PA_FORCE_LOCKABLE_MAPPING="$force_lockable"
        else
            unset OV_GPU_CM_PA_FORCE_LOCKABLE_MAPPING
        fi

        run_args=("${COMMON_ARGS[@]}")
        if [[ "$impl" == "cm" ]]; then
            run_args+=(--use_xattention --xattention_block_size "$xattention_block_size" --xattention_threshold "$xattention_block_threshold")
        fi
        run_args+=("$@")

        if [[ "$PROFILE" == "1" ]]; then
            cliloader -cdt --dump-dir "$dump_dir" "$BIN" "${run_args[@]}"
        else
            "$BIN" "${run_args[@]}"
        fi
    ) 2>&1 | tee "$log_file"

    echo "---- Summary: $case_name ----"
    grep -E "Benchmark duration:|Input throughput:|Output throughput:|Mean TTFT:|Mean TPOT:" "$log_file" || true
    echo
}

 # Run selected cases based on arguments
if [[ "$BACKEND" == "cm" || "$BACKEND" == "both" ]]; then
    # If route is 'both', run both multi and split
    if [[ "$ROUTE" == "both" ]]; then
        for route_mode in multi split; do
            for force_lockable in 0 1; do
                run_case "cm_${route_mode}_xattn${XATTENTION_BLOCK_SIZE}_thr${XATTENTION_BLOCK_THRESHOLD}_flm${force_lockable}" "cm" "$route_mode" "$force_lockable" "$XATTENTION_BLOCK_SIZE" "$XATTENTION_BLOCK_THRESHOLD"
            done
        done
    else
        for force_lockable in 0 1; do
            run_case "cm_${ROUTE}_xattn${XATTENTION_BLOCK_SIZE}_thr${XATTENTION_BLOCK_THRESHOLD}_flm${force_lockable}" "cm" "$ROUTE" "$force_lockable" "$XATTENTION_BLOCK_SIZE" "$XATTENTION_BLOCK_THRESHOLD"
        done
    fi
fi
if [[ "$BACKEND" == "ocl" || "$BACKEND" == "both" ]]; then
    # ocl backend: route multi, no xattention
    run_case "ocl" "ocl" "multi" "na" "na" "na"
fi

echo "All selected cases finished. Logs are in: $LOG_DIR"
