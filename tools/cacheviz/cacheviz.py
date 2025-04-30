# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
After running a continuous batching workload with debug dumps enabled, observe a "debug" folder in the working directory;
the subdirectories `cache_dump/*N*` correspond to the state of N-th layer cache at each generation step, and can be
visualized by running:
cacheviz.py --dump_folder ./debug/cache_dump/0

Use "A" and "D" (or "left arrow" and "right arrow") keys to move to the previous or next steps correspondingly,
with "Alt" modifier to move 10 steps at a time, and "Shift" modifier to move 100 steps at a time.
"""

import argparse
import hashlib
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib import patches
plt.switch_backend('TkAgg')

BLOCK_SIZE = 32
EVICTION_START_SIZE = 32
EVICTION_EVICTABLE_SIZE = 64
EVICTION_RECENT_SIZE = 32


def is_evictable(logical_block_idx: int, total_occupied_logical_blocks: int):
    assert(logical_block_idx < total_occupied_logical_blocks)
    if total_occupied_logical_blocks <= (EVICTION_START_SIZE + EVICTION_EVICTABLE_SIZE + EVICTION_RECENT_SIZE) / BLOCK_SIZE:
        return False
    logical_block_idx_in_tokens = logical_block_idx * BLOCK_SIZE
    return EVICTION_START_SIZE <= logical_block_idx_in_tokens < EVICTION_START_SIZE + EVICTION_EVICTABLE_SIZE


def get_hashed_rgb_color(idx: int) -> str:
    return '#' + hashlib.sha1(str(idx).encode()).hexdigest()[0:6]  # nosec


@dataclass
class StepDumpData:
    dump_file_name: str = None
    num_blocks: int = None
    occupied_blocks: dict[int, list[tuple[int, int]]] = field(default_factory=lambda: defaultdict(list))
    occupied_blocks_per_sequence: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    sequence_groups: dict[int, list[int]] = field(default_factory=dict)


def load_data(dump_dir: pathlib.Path) -> list[StepDumpData]:
    retval = []
    num_step_files = 0
    step_file_names_dict: dict[int, list[pathlib.Path]] = defaultdict(list)

    for f in dump_dir.iterdir():
        if f.is_file() and f.suffix == '.txt' and 'usage' not in f.name:
            file_name = f.stem
            step_number = int(file_name.split("_")[-1])
            step_file_names_dict[step_number].append(f)
            num_step_files += 1

    if num_step_files == 0:
        print(f"No step files found")
        exit(-1)

    print(f"Step files found: {num_step_files}")
    step_file_names_in_order = [name_lex_sorted for _, names_for_step in sorted(step_file_names_dict.items()) for
                                name_lex_sorted in sorted(names_for_step)]

    for dump_file_name in tqdm.tqdm(step_file_names_in_order):
        collected_data = StepDumpData()
        collected_data.dump_file_name = dump_file_name.name
        with open(dump_file_name, "r") as f:
            num_blocks_line = f.readline()
            collected_data.num_blocks = int(num_blocks_line)
            num_sequence_groups_line = f.readline()
            num_sequence_groups = int(num_sequence_groups_line)
            for i in range(num_sequence_groups):
                sequence_group_line = f.readline()
                sequence_group_tokens = sequence_group_line.split()
                sequence_group_id = int(sequence_group_tokens[0])
                sequence_group_seq_ids = [int(s) for s in sequence_group_tokens[1:]]
                collected_data.sequence_groups[sequence_group_id] = sequence_group_seq_ids

            for (i, line) in enumerate(f):
                tokens = line.split()
                seq_id, block_idx, ref_count = int(tokens[0]), int(tokens[1]), int(tokens[2])
                if block_idx not in collected_data.occupied_blocks:
                    collected_data.occupied_blocks[block_idx] = [(seq_id, ref_count)]
                else:
                    collected_data.occupied_blocks[block_idx].append((seq_id, ref_count))
                collected_data.occupied_blocks_per_sequence[seq_id].append(block_idx)
        retval.append(collected_data)
    return retval


def get_allocated_usage_series(step_data: list[StepDumpData]) -> list[float]:
    return [len(sd.occupied_blocks) / sd.num_blocks * 100 for sd in step_data]


def draw_from_step_data(plot_axes: plt.Axes, step_data: StepDumpData) -> plt.Axes:
    num_blocks = step_data.num_blocks
    occupied_blocks = step_data.occupied_blocks
    occupied_blocks_per_sequence = step_data.occupied_blocks_per_sequence
    sequence_groups = step_data.sequence_groups

    seq_id_to_sequence_group_id: dict[int, int] = { seq_id: seq_group_id for seq_group_id, seq_id_list in sequence_groups.items() for seq_id in seq_id_list }

    nrows = 1
    ncols = num_blocks // nrows

    width = 1
    height = width

    # Positions of the square patches are shifted half-unit to the right so that the ticks on the X axis end up
    # centered at the squares middle points
    patch_x_positions = np.arange(0.0, ncols, width)
    patch_x_positions -= 0.5

    # Shade the areas occupied by at least one sequence for a visual representation of the cache usage
    for occupied_block_idx in occupied_blocks:
        vspan_from = patch_x_positions[occupied_block_idx]
        vspan_to = vspan_from + 1
        plot_axes.axvspan(vspan_from, vspan_to, alpha=0.5, color='gray')

    max_ylim = 1

    # Set up the squares for individual sequences and for the overall block table usage
    for block_idx, patch_xpos in enumerate(patch_x_positions):
        # Block table usage indicator (occupying position -1 on the Y axis)
        base_pos = (patch_xpos, -1.5)
        base_face_color = '1'
        num_occupying_sequences = 0
        base_text_color = 'black'
        if block_idx in occupied_blocks:
            num_occupying_sequences = occupied_blocks[block_idx][0][1]
            base_face_color = str(1 / (2 * num_occupying_sequences))
            base_text_color = 'white'
        sq = patches.Rectangle(base_pos, width, height, fill=True, facecolor=base_face_color, edgecolor='black')
        plot_axes.add_patch(sq)

        # Mark the block with the number of occupying sequences
        text = str(num_occupying_sequences)
        center = (base_pos[0] + 0.5, base_pos[1] + 0.5)
        plot_axes.annotate(text, center, ha='center', va='center', color=base_text_color)

        if block_idx in occupied_blocks:
            for seq_idx, ref_count in occupied_blocks[block_idx]:
                # Draw the blocks representing the occupied sequence - a block at each occupied position on the X axis,
                # with Y position equal to the sequence ID.
                sequence_local_text = str(seq_idx)
                sequence_local_center = (center[0], center[1] + (seq_idx + 1) * height)
                seq_sq_pos = (base_pos[0], base_pos[1] + (seq_idx + 1))
                max_ylim = max(max_ylim, seq_idx + 1)
                seq_color = get_hashed_rgb_color(seq_idx)
                seq_group_color = get_hashed_rgb_color(-seq_id_to_sequence_group_id[seq_idx] - 1)
                linestyle = 'solid'
                logical_idx_in_seq = occupied_blocks_per_sequence[seq_idx].index(block_idx)
                if is_evictable(logical_idx_in_seq, len(occupied_blocks_per_sequence[seq_idx])):
                    linestyle = 'dotted'
                seq_sq = patches.Rectangle(seq_sq_pos, width, height, fill=True, facecolor=seq_color, edgecolor=seq_group_color, lw=3,
                                           linestyle=linestyle)
                plot_axes.add_patch(seq_sq)
                plot_axes.annotate(sequence_local_text, sequence_local_center, ha='center', va='center')

                # Display total blocks used on the right side of the plot
                pos_on_right_of_plot_at_sequence_idx = (num_blocks, sequence_local_center[1])
                plot_axes.annotate(str(len(occupied_blocks_per_sequence[seq_idx])), pos_on_right_of_plot_at_sequence_idx,
                                   ha='center', va='center',
                                   color=seq_color, weight='bold')

    # Set limits and ticks so that only integer ticks are visible and all the range is shown
    plot_axes.set_yticks(np.arange(max_ylim))
    plot_axes.set_ylim(-1.5, max_ylim)
    plot_axes.set_xticks(np.arange(num_blocks))
    plot_axes.set_xlim(-0.5, num_blocks + 0.5)

    # Labels
    plot_axes.set_xlabel('Block index')
    plot_axes.set_ylabel('Sequence index')
    plot_axes.set_title(step_data.dump_file_name)

    # Legend for sequence group colors
    plot_axes.legend(handles=[patches.Patch(facecolor=get_hashed_rgb_color(-seq_group_idx - 1),
                                            label=f'Sequence group {seq_group_idx}') for seq_group_idx in
                              sequence_groups], loc='center left', bbox_to_anchor=(1, 0.5))

    return plot_axes


def load_and_draw_usage(plot_axes: plt.Axes, usage_dump_file: pathlib.Path, current_step: int, allocated_usage_series: list[float], eviction_relation='before') -> tuple[plt.Axes, float, tuple[list, list]]:
    usage_values: dict[int, tuple[float, float]] = {}
    with open(usage_dump_file, "r") as f:
        while True:
            before_eviction_line = f.readline()
            after_eviction_line = f.readline()
            if before_eviction_line is None or after_eviction_line is None or before_eviction_line == '' or after_eviction_line == '':
                break
            before_step_num, before_cache_usage = before_eviction_line.split()
            after_step_num, after_cache_usage = after_eviction_line.split()
            assert before_step_num == after_step_num
            step_num = int(before_step_num)
            usage_values[step_num] = (float(before_cache_usage), float(after_cache_usage))

    step_numbers = [k for k in usage_values.keys()]
    before_series = [v[0] for v in usage_values.values()]
    after_series = [v[1] for v in usage_values.values()]

    # plot "after" first so that it ends up under the "before" plot for better visibility of eviction
    plot_axes.plot(step_numbers, after_series, color='blue')
    plot_axes.plot(step_numbers, before_series, color='green')

    allocated_usage_before_series = [v for v in allocated_usage_series[0::2]]
    allocated_usage_after_series = [v for v in allocated_usage_series[1::2]]

    leaked_before_series = [r - a if (r - a) > 0 else 0 for r, a in zip(before_series, allocated_usage_before_series)]
    leaked_after_series = [r - a if (r - a) > 0 else 0 for r, a in zip(after_series, allocated_usage_after_series)]
    plot_axes.plot(step_numbers, leaked_after_series, color='orange')
    plot_axes.plot(step_numbers, leaked_before_series, color='red')

    plot_axes.set_yticks(np.arange(0, 100, 10))
    plot_axes.set_ylim(0, 100)
    plot_axes.grid(visible=True, which='major', axis='y')

    plot_axes.set_xticks(np.arange(0, step_num, 100))
    plot_axes.set_xlim(0, step_num)

    # Labels
    plot_axes.set_xlabel('Step')
    plot_axes.set_ylabel('Cache usage, %')

    plot_axes.vlines(current_step, ymin=0, ymax=100, colors='red')

    plot_axes.legend(['after eviction', 'before eviction', 'leaked (after eviction)', 'leaked (before eviction)'])

    if eviction_relation == 'before':
        reported_cache_usage = usage_values[current_step][0]
        allocated_usage_series = allocated_usage_before_series[current_step]
    if eviction_relation == 'after':
        reported_cache_usage = usage_values[current_step][1]
        allocated_usage_series = allocated_usage_after_series[current_step]

    plot_axes.annotate(
            f'Block table usage: {allocated_usage_series:.2f}% (occupied), {reported_cache_usage:.2f}% (reported)',
            xy=(0.5, 0), xytext=(0, 10),
            xycoords=('axes fraction', 'figure fraction'),
            textcoords='offset points',
            size=14, ha='center', va='bottom')


def get_eviction_relation(dump_file_name: str) -> str:
    return 'before' if 'before' in str(dump_file_name) else 'after'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_folder", help="Cache info dump folder", required=True)
    parser.add_argument("--step", help="Step ID to show at startup", required=False, default=0, type=int)
    args = parser.parse_args()
    dump_folder = args.dump_folder

    dump_folder_path = pathlib.Path(dump_folder)
    step_data = load_data(dump_folder_path)
    allocated_usage_series = get_allocated_usage_series(step_data)

    fig = plt.figure(figsize=(10, 10))
    fig.tight_layout()
    plot_axes = fig.add_subplot(211, aspect='equal')

    current_file_idx_displayed: int = args.step * 2  # 2 files per step - before and after eviction

    usage_dump_file = dump_folder_path / "cache_usage.txt"

    def on_press(event):
        nonlocal current_file_idx_displayed
        if event.key == 'd' or event.key == 'right':
            current_file_idx_displayed += 1
        elif event.key == 'a' or event.key == 'left':
            current_file_idx_displayed -= 1
        if event.key == 'alt+d' or event.key == 'alt+right':
            current_file_idx_displayed += 10 * 2
        elif event.key == 'alt+a' or event.key == 'alt+left':
            current_file_idx_displayed -= 10 * 2
        if event.key == 'D' or event.key == 'shift+right':
            current_file_idx_displayed += 100 * 2
        elif event.key == 'A' or event.key == 'shift+left':
            current_file_idx_displayed -= 100 * 2
        current_file_idx_displayed %= len(step_data)

        mode = get_eviction_relation(step_data[current_file_idx_displayed].dump_file_name)

        plot_axes.clear()
        draw_from_step_data(plot_axes, step_data[current_file_idx_displayed])

        usage_plot_axes.clear()
        load_and_draw_usage(usage_plot_axes, usage_dump_file, current_file_idx_displayed // 2, allocated_usage_series=allocated_usage_series, eviction_relation=mode)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_press)
    usage_plot_axes = fig.add_subplot(212, aspect='auto')

    curr_step_file_data = step_data[current_file_idx_displayed]
    mode = get_eviction_relation(curr_step_file_data.dump_file_name)

    draw_from_step_data(plot_axes, curr_step_file_data)
    load_and_draw_usage(usage_plot_axes, usage_dump_file, args.step, allocated_usage_series=allocated_usage_series, eviction_relation=mode)

    plt.show()


if __name__ == "__main__":
    main()



