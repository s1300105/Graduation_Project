from collections import OrderedDict
from utils.functions.cpg_dir.function import Function


def _as_int(v, default=10**9):
    """
    Line/column can be None or even strings in some CPGs.
    Normalize to int with a large default so missing values sort last.
    """
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def order_nodes(nodes, max_nodes):
    """
    Sort nodes by (line, column, id) safely even if line/column are None/strings.
    Assign a stable sequential `order` and cap by max_nodes.
    """
    nodes_by_pos = sorted(
        nodes.items(),
        key=lambda it: (
            _as_int(it[1].get_line_number()),
            _as_int(it[1].get_column_number()),
            it[0],  # tie-breaker for stability
        ),
    )

    for i, (_, node) in enumerate(nodes_by_pos):
        node.order = i

    if len(nodes_by_pos) > max_nodes:
        print(f"CPG cut - original nodes: {len(nodes_by_pos)} to max: {max_nodes}")
        nodes_by_pos = nodes_by_pos[:max_nodes]

    return OrderedDict(nodes_by_pos)


def filter_nodes(nodes):
    # keep only nodes that are useful for modeling
    return {
        n_id: node
        for n_id, node in nodes.items()
        if node.has_code()
        and node.has_line_number()  # column may be missing; sorter handles it safely
        and node.label not in ["Comment", "Unknown"]
    }


def parse_to_nodes(cpg, max_nodes=500):
    nodes = {}
    for function in cpg.get("functions", []):
        func = Function(function)
        filtered_nodes = filter_nodes(func.get_nodes())
        nodes.update(filtered_nodes)

    return order_nodes(nodes, max_nodes)
