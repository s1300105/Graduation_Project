"""
utils/export_gephi.py — Export CPG PKL data to Gephi GEXF format.

Usage:
    python utils/export_gephi.py --pkl data/cpg/train_0_cpg.pkl --func 0 --out tmp/gephi_out/
    python utils/export_gephi.py --pkl data/cpg/train_0_cpg.pkl --out tmp/gephi_out/  # all functions

Arguments:
    --pkl   Path to a *_cpg.pkl file (DataFrame with 'cpg' and 'target' columns)
    --row   Row index in the DataFrame (default: 0)
    --func  Function index within the row's CPG (default: export all)
    --out   Output directory (created if not exists)
"""

import argparse
import os
import pickle
import sys
import xml.etree.ElementTree as ET

import pandas as pd

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.functions.cpg import parse_function_v4_to_nodes, EDGE_KINDS


def export_function_to_gexf(func_json: dict, target: int, out_dir: str, prefix: str):
    """Parse one function's CPG JSON and write a GEXF file."""
    node_map = parse_function_v4_to_nodes(func_json, max_nodes=10000, strict_filter=False)

    if not node_map:
        print(f"  [skip] {prefix}: no nodes after parsing")
        return

    # Build node id → sequential index mapping
    node_ids = list(node_map.keys())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # Root element
    gexf = ET.Element("gexf", {
        "xmlns": "http://gexf.net/1.3",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": "http://gexf.net/1.3 http://gexf.net/1.3/gexf.xsd",
        "version": "1.3",
    })

    graph = ET.SubElement(gexf, "graph", defaultedgetype="directed")

    # Node attribute declarations
    node_attrs = ET.SubElement(graph, "attributes", {"class": "node"})
    ET.SubElement(node_attrs, "attribute", id="0", title="label", type="string")
    ET.SubElement(node_attrs, "attribute", id="1", title="code", type="string")
    ET.SubElement(node_attrs, "attribute", id="2", title="line_number", type="integer")
    ET.SubElement(node_attrs, "attribute", id="3", title="column_number", type="integer")
    ET.SubElement(node_attrs, "attribute", id="4", title="target", type="integer")

    # Edge attribute declarations
    edge_attrs = ET.SubElement(graph, "attributes", {"class": "edge"})
    ET.SubElement(edge_attrs, "attribute", id="0", title="edgetype", type="string")

    # Nodes
    nodes_el = ET.SubElement(graph, "nodes")
    for nid in node_ids:
        node = node_map[nid]
        idx = id_to_idx[nid]
        n = ET.SubElement(nodes_el, "node", id=str(idx), label=node.label or "")
        attvalues = ET.SubElement(n, "attvalues")
        ET.SubElement(attvalues, "attvalue", **{"for": "0", "value": node.label or ""})
        ET.SubElement(attvalues, "attvalue", **{"for": "1", "value": node.code or ""})
        ET.SubElement(attvalues, "attvalue", **{
            "for": "2",
            "value": str(node.line_number) if node.line_number is not None else "",
        })
        ET.SubElement(attvalues, "attvalue", **{
            "for": "3",
            "value": str(node.column_number) if node.column_number is not None else "",
        })
        ET.SubElement(attvalues, "attvalue", **{"for": "4", "value": str(target)})

    # Edges
    edges_el = ET.SubElement(graph, "edges")
    seen_edges = set()
    edge_count = 0
    for nid in node_ids:
        node = node_map[nid]
        for ekey, edge in node.edges.items():
            src = str(edge.node_in)
            dst = str(edge.node_out)
            canonical = (src, dst, edge.type)
            if canonical in seen_edges:
                continue
            seen_edges.add(canonical)
            if src not in id_to_idx or dst not in id_to_idx:
                continue
            e = ET.SubElement(edges_el, "edge", {
                "id": str(edge_count),
                "source": str(id_to_idx[src]),
                "target": str(id_to_idx[dst]),
                "weight": "1",
            })
            attvalues = ET.SubElement(e, "attvalues")
            ET.SubElement(attvalues, "attvalue", **{"for": "0", "value": edge.type or ""})
            edge_count += 1

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}.gexf")
    tree = ET.ElementTree(gexf)
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="unicode", xml_declaration=True)
    print(f"  Wrote {len(node_ids)} nodes, {edge_count} edges → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Export CPG PKL to Gephi GEXF")
    parser.add_argument("--pkl", required=True, help="Path to *_cpg.pkl file")
    parser.add_argument("--row", type=int, default=0, help="Row index in DataFrame (default: 0)")
    parser.add_argument("--func", type=int, default=None,
                        help="Function index within the CPG (default: all functions)")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    print(f"Loading {args.pkl} ...")
    with open(args.pkl, "rb") as f:
        df = pickle.load(f)

    if not isinstance(df, pd.DataFrame):
        print("ERROR: PKL file does not contain a pandas DataFrame", file=sys.stderr)
        sys.exit(1)

    if args.row >= len(df):
        print(f"ERROR: row {args.row} out of range (DataFrame has {len(df)} rows)", file=sys.stderr)
        sys.exit(1)

    row = df.iloc[args.row]
    cpg = row["cpg"]
    target = int(row["target"]) if "target" in row else 0
    functions = cpg.get("functions", [])
    print(f"Row {args.row}: target={target}, {len(functions)} function(s) in CPG")

    if args.func is not None:
        if args.func >= len(functions):
            print(f"ERROR: --func {args.func} out of range ({len(functions)} functions)", file=sys.stderr)
            sys.exit(1)
        func_json = functions[args.func]
        prefix = f"func_{args.func}"
        print(f"Exporting function {args.func} ...")
        export_function_to_gexf(func_json, target, args.out, prefix)
    else:
        for fi, func_json in enumerate(functions):
            prefix = f"func_{fi}"
            print(f"Exporting function {fi} ...")
            export_function_to_gexf(func_json, target, args.out, prefix)

    print("Done.")


if __name__ == "__main__":
    main()
