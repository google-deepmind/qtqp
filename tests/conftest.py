import pytest
from collections import defaultdict

def pytest_configure(config):
    # Use a dictionary to group stats by test name
    config.solver_stats = defaultdict(list)
    # Per-solver stats: solver_name -> list of dicts
    config.per_solver_stats = defaultdict(list)

def _problem_type(test_name):
    """Derive problem type from test function name."""
    if 'infeasible' in test_name:
        return 'infeasible'
    if 'unbounded' in test_name:
        return 'unbounded'
    return 'feasible'

@pytest.fixture
def record_iterations(request):
    """
    Fixture that records iterations and solve time, grouped by test name and solver.
    """
    # .originalname handles parametrized tests (e.g. groups 'test_solve[True...]' under 'test_solve')
    test_name = request.node.originalname or request.node.name

    # Extract solver name from parametrized params if available.
    solver_name = None
    if hasattr(request.node, 'callspec'):
        ls = request.node.callspec.params.get('linear_solver')
        if ls is not None:
            solver_name = ls.name

    def _record(iters, time=None):
        request.config.solver_stats[test_name].append(iters)
        if solver_name is not None:
            request.config.per_solver_stats[solver_name].append({
                'test': test_name,
                'type': _problem_type(test_name),
                'iters': iters,
                'time': time,
            })
    return _record


# ---------------------------------------------------------------------------
# Log-scale time histogram bins (seconds)
# ---------------------------------------------------------------------------
_TIME_BINS = [
    (0.001,  '<1ms'),
    (0.003,  '1-3ms'),
    (0.01,   '3-10ms'),
    (0.03,   '10-30ms'),
    (0.1,    '30-100ms'),
    (0.3,    '0.1-0.3s'),
    (1.0,    '0.3-1s'),
    (3.0,    '1-3s'),
    (float('inf'), '>3s'),
]


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    stats = config.solver_stats
    if not stats:
        return

    # --- Per-test summary (existing) ---
    terminalreporter.section("QTQP Solver Iteration Statistics")

    name_width = max(len("Test Name"), max((len(n) for n in stats), default=0))
    fmt = "{:<" + str(name_width) + "} | {:<10} | {:<10} | {:<10}"
    terminalreporter.write_line(fmt.format("Test Name", "Count", "Avg Iter", "Max Iter"))
    terminalreporter.write_line("-" * (name_width + 40))

    all_iterations = []
    for name, values in sorted(stats.items()):
        all_iterations.extend(values)
        avg = sum(values) / len(values)
        maximum = max(values)
        terminalreporter.write_line(fmt.format(name, len(values), f"{avg:.2f}", maximum))

    terminalreporter.write_line("-" * (name_width + 40))
    if all_iterations:
        total_avg = sum(all_iterations) / len(all_iterations)
        total_max = max(all_iterations)
        terminalreporter.write_line(fmt.format("AGGREGATE", len(all_iterations), f"{total_avg:.2f}", total_max))

    # --- Per-solver summary ---
    per_solver = config.per_solver_stats
    if not per_solver:
        return

    # ---------------------------------------------------------------
    # Per-solver + problem-type breakdown
    # ---------------------------------------------------------------
    terminalreporter.section("QTQP Per-Solver Statistics")

    types = ['feasible', 'infeasible', 'unbounded']
    fmt2 = "{:<15} {:<12} | {:>6} | {:>9} | {:>9} | {:>11} | {:>11} | {:>11}"
    terminalreporter.write_line(fmt2.format(
        "Solver", "Type", "Count", "Avg Iter", "Max Iter",
        "Med Time", "Max Time", "Tot Time",
    ))
    terminalreporter.write_line("-" * 95)

    for solver_name in sorted(per_solver.keys()):
        entries = per_solver[solver_name]

        for ptype in types + ['ALL']:
            if ptype == 'ALL':
                subset = entries
                label = 'ALL'
            else:
                subset = [e for e in entries if e['type'] == ptype]
                label = ptype
            if not subset:
                continue

            iters = [e['iters'] for e in subset]
            times = sorted(e['time'] for e in subset if e['time'] is not None)

            avg_iter = sum(iters) / len(iters)
            max_iter = max(iters)

            if times:
                med_time = times[len(times) // 2]
                max_time = times[-1]
                tot_time = sum(times)
                time_cols = f"{med_time:>11.2e} | {max_time:>11.2e} | {tot_time:>11.2e}"
            else:
                time_cols = f"{'n/a':>11} | {'n/a':>11} | {'n/a':>11}"

            terminalreporter.write_line(fmt2.format(
                solver_name if (ptype == types[0]) else '',
                label, len(subset), f"{avg_iter:.2f}", max_iter,
                *time_cols.split(" | "),
            ))

        terminalreporter.write_line("-" * 95)

    # Aggregate row
    all_entries = [e for entries in per_solver.values() for e in entries]
    all_iters = [e['iters'] for e in all_entries]
    all_times = sorted(e['time'] for e in all_entries if e['time'] is not None)
    if all_iters:
        avg_iter = sum(all_iters) / len(all_iters)
        max_iter = max(all_iters)
        if all_times:
            med_time = all_times[len(all_times) // 2]
            max_time = all_times[-1]
            tot_time = sum(all_times)
            time_cols = f"{med_time:>11.2e} | {max_time:>11.2e} | {tot_time:>11.2e}"
        else:
            time_cols = f"{'n/a':>11} | {'n/a':>11} | {'n/a':>11}"
        terminalreporter.write_line(fmt2.format(
            "AGGREGATE", "ALL", len(all_entries), f"{avg_iter:.2f}", max_iter,
            *time_cols.split(" | "),
        ))

    # ---------------------------------------------------------------
    # Solve-time histogram: vertical layout (one row per time bucket,
    # solvers as columns) for easy cross-solver comparison.
    # ---------------------------------------------------------------
    all_times_flat = [e['time'] for entries in per_solver.values()
                      for e in entries if e['time'] is not None]
    if not all_times_flat:
        return

    terminalreporter.section("QTQP Solve Time Distribution")

    solver_names = sorted(per_solver.keys())

    # Build count matrix: counts[bin_label][solver_name] = count
    counts = {}
    for solver_name in solver_names:
        times = [e['time'] for e in per_solver[solver_name] if e['time'] is not None]
        bin_counts = [0] * len(_TIME_BINS)
        for t in times:
            for i, (upper, _) in enumerate(_TIME_BINS):
                if t < upper:
                    bin_counts[i] += 1
                    break
        for (_, label), c in zip(_TIME_BINS, bin_counts):
            counts.setdefault(label, {})[solver_name] = c

    # Determine column width from solver names (min 10 for the bar+count).
    col_w = max(10, max((len(s) for s in solver_names), default=0) + 3)
    global_max = max((c for by_solver in counts.values() for c in by_solver.values()), default=0)
    bar_w = col_w - 5  # leave room for " NNN"

    # Header row with solver names.
    header = f"{'':>10}" + "".join(f"{s:>{col_w}}" for s in solver_names)
    terminalreporter.write_line(header)
    terminalreporter.write_line("-" * len(header))

    # One row per time bucket.
    for _, label in _TIME_BINS:
        by_solver = counts.get(label, {})
        if not any(by_solver.get(s, 0) for s in solver_names):
            continue
        cells = []
        for s in solver_names:
            c = by_solver.get(s, 0)
            if c:
                bw = max(1, round(c / global_max * bar_w))
                bar = "\u2588" * bw
                cell = f"{bar} {c}"
            else:
                cell = ""
            cells.append(f"{cell:>{col_w}}")
        terminalreporter.write_line(f"{label:>10}" + "".join(cells))
