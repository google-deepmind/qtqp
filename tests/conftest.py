import pytest
from collections import defaultdict

def pytest_configure(config):
    # Use a dictionary to group stats by test name
    config.solver_stats = defaultdict(list)

@pytest.fixture
def record_iterations(request):
    """
    Fixture that records iterations, grouped by the test function name.
    """
    # .originalname handles parametrized tests (e.g. groups 'test_solve[True...]' under 'test_solve')
    test_name = request.node.originalname or request.node.name
    
    def _record(n):
        request.config.solver_stats[test_name].append(n)
    return _record

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    stats = config.solver_stats
    if not stats:
        return

    terminalreporter.section("QTQP Solver Iteration Statistics")
    
    # Define column widths for cleaner output
    fmt = "{:<20} | {:<10} | {:<10} | {:<10}"
    terminalreporter.write_line(fmt.format("Test Name", "Count", "Avg Iter", "Max Iter"))
    terminalreporter.write_line("-" * 60)

    all_iterations = []

    # Print rows for each individual test function
    for name, values in sorted(stats.items()):
        all_iterations.extend(values)
        avg = sum(values) / len(values)
        maximum = max(values)
        terminalreporter.write_line(fmt.format(name, len(values), f"{avg:.2f}", maximum))

    # Print Grand Total
    terminalreporter.write_line("-" * 60)
    if all_iterations:
        total_avg = sum(all_iterations) / len(all_iterations)
        total_max = max(all_iterations)
        terminalreporter.write_line(fmt.format("AGGREGATE", len(all_iterations), f"{total_avg:.2f}", total_max))
