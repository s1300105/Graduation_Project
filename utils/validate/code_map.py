# utils/validate/code_map.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import html
import random

def _pick_palette(n: int) -> List[str]:
    # できるだけ見分けやすい固定色 + ランダム補完（重複を避けるだけ）
    base = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    ]
    if n <= len(base):
        return base[:n]
    extra = []
    random.seed(42)
    while len(base) + len(extra) < n:
        c = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        if c not in base and c not in extra:
            extra.append(c)
    return base + extra

def _build_code_html(
    code: str,
    node2lines: Dict[int, Sequence[int]],
    node2color: Dict[int, str],
    title: str = "Function Source",
) -> str:
    """
    左ペイン：コード表示。node2lines に含まれる行を色ハイライト。
    - 行番号は 1 始まり
    """
    esc = html.escape
    lines = code.splitlines()
    # 行ごとの所属ノード（複数ノードが同じ行に乗る場合もある）
    line_tags: List[List[int]] = [[] for _ in range(len(lines))]
    for nid, L in node2lines.items():
        for li in L:
            if 1 <= li <= len(lines):
                line_tags[li-1].append(nid)

    rows = []
    for i, text in enumerate(lines, start=1):
        tags = line_tags[i-1]
        if tags:
            # 複数ノードが同一行に乗るときは縞で重ねる
            grad = ",".join(node2color[t] for t in tags[:4])  # 見切れ防止で最大4つ
            style = f"background: linear-gradient(90deg, {grad});"
        else:
            style = ""
        row = (
            f"<tr>"
            f"  <td class='ln'>{i}</td>"
            f"  <td class='code' style='{style}'>{esc(text if text else ' ')}</td>"
            f"</tr>"
        )
        rows.append(row)

    table = (
        f"<div class='pane'>"
        f"<h3>{esc(title)}</h3>"
        f"<table class='codebox'>"
        f"{''.join(rows)}"
        f"</table>"
        f"</div>"
    )
    return table

def _build_graph_html(elements_html: str, legend_html: str, title: str = "Graph") -> str:
    return (
        f"<div class='pane'>"
        f"<h3>{html.escape(title)}</h3>"
        f"{legend_html}"
        f"{elements_html}"
        f"</div>"
    )

def _nx_from_pyg(data):
    import networkx as nx
    from torch_geometric.utils import to_networkx
    has_etype = hasattr(data, "edge_type") and getattr(data, "edge_type") is not None
    G = to_networkx(
        data, node_attrs=[], edge_attrs=["edge_type"] if has_etype else None, to_undirected=True
    )
    # edge_type を int へ寄せる
    if has_etype:
        for u, v, d in G.edges(data=True):
            try:
                d["edge_type"] = int(d.get("edge_type", -1))
            except Exception:
                d["edge_type"] = -1
    return G


def render_code_and_graph_html(
    data,
    out_html: str = "data/inspect/graph_with_code.html",
    node_lines: Optional[Dict[int, Sequence[int]]] = None,
    edge_type_labels: Optional[Dict[int, str]] = None,
    title: Optional[str] = None,
):
    from pathlib import Path
    import json, html as _html

    # --- 1) コード ---
    code = getattr(data, "func", None) or ""
    if not code:
        raise ValueError("data.func（関数コード）がありません。")

    # --- 2) node_lines 推定 ---
    N = int(getattr(data, "num_nodes", 0) or 0)
    if node_lines is None:
        node_lines = {}
        cand = {}
        for name in ("node_line", "line", "lineno"):
            if hasattr(data, name):
                v = getattr(data, name)
                try: v = v.tolist()
                except Exception: pass
                if isinstance(v, (list, tuple)) and len(v) == N:
                    cand["start"] = [int(x) if (x is not None) else -1 for x in v]
                    break
        for name in ("node_end_line", "end_line", "endlineno"):
            if hasattr(data, name):
                v = getattr(data, name)
                try: v = v.tolist()
                except Exception: pass
                if isinstance(v, (list, tuple)) and len(v) == N:
                    cand["end"] = [int(x) if (x is not None) else -1 for x in v]
                    break
        if "start" in cand:
            starts = cand["start"]
            ends = cand.get("end", [s for s in starts])
            for nid in range(N):
                s, e = starts[nid], ends[nid]
                if s is None or s < 1: continue
                e = e if (e is not None and e >= s) else s
                node_lines[nid] = list(range(s, e + 1))

    # --- 3) 色割り当て（ノード） ---
    from .code_map import _pick_palette, _build_code_html, _nx_from_pyg, _build_graph_html
    node_ids = sorted(node_lines.keys())
    node_palette = _pick_palette(max(1, len(node_lines)))
    node2color = {nid: node_palette[i % len(node_palette)] for i, nid in enumerate(node_ids)}

    # --- 4) エッジタイプの抽出＆色割り当て ---
    import torch
    edge_types_present: List[int] = []
    if hasattr(data, "edge_type") and getattr(data, "edge_type") is not None:
        et = getattr(data, "edge_type")
        try:
            uniq = torch.unique(et).tolist()
            edge_types_present = [int(u) for u in uniq]
        except Exception:
            pass
    edge_types_present.sort()
    if edge_type_labels is None:
        # 名前が無ければ "etype k" にする
        edge_type_labels = {k: f"etype {k}" for k in edge_types_present}

    edge_palette = _pick_palette(max(1, len(edge_types_present))) if edge_types_present else []
    et2color = {k: edge_palette[i % len(edge_palette)] for i, k in enumerate(edge_types_present)}

    # --- 5) PyVis で描画（できなければPNGにフォールバック） ---
    graph_div = None
    pyvis_ok = False
    try:
        from pyvis.network import Network

        G = _nx_from_pyg(data)
        net = Network(height="740px", width="100%", directed=False, notebook=False)
        net.barnes_hut()

        # ノード
        for i in G.nodes:
            color = node2color.get(int(i), "#89a")
            net.add_node(int(i), label=str(i), color=color, title=f"node {i}")

        # エッジ（タイプ別に色分け）
        if G.number_of_edges() > 0:
            for u, v, d in G.edges(data=True):
                et = int(d.get("edge_type", -1))
                et_name = edge_type_labels.get(et, f"etype {et}")
                ecolor = et2color.get(et, "#999")
                net.add_edge(int(u), int(v), title=et_name, color=ecolor, width=2)

        # ファイル書き出し
        out_path = Path(out_html)
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        embed_name = out_path.stem + "_pyvis.html"   # 例: graph_with_code_pyvis.html
        embed_path = out_dir / embed_name

        net.write_html(str(embed_path), notebook=False, open_browser=False)

        # === PyVis 側にブリッジJSを注入（親<->iframe 連携） ===
        _inject_pyvis_bridge(embed_path)

        # iframe で埋め込む
        graph_div = (
            f"<iframe id='graph_iframe' src='{embed_name}' "
            f"style='width:100%; height:740px; border:0;' "
            f"sandbox='allow-scripts allow-same-origin'></iframe>"
        )
        pyvis_ok = True

    except Exception:
        # PNG フォールバック
        from .visualize_graph import plot_pyg_graph
        png_path = Path(out_html).with_suffix(".png")
        plot_pyg_graph(
            data,
            out_path=str(png_path),
            use_khop=True,
            center_node=None,
            num_hops=2,
            max_nodes=150,
            edge_type_labels=edge_type_labels,
            auto_center=True,
            show_node_ids=True,
        )
        graph_div = f"<img src='{png_path.name}' style='max-width:100%;'>"

    # --- 6) 凡例（ノード色 + エッジ色） ---
    legend_node_items = []
    for nid in node_ids[:30]:
        legend_node_items.append(f"<span class='chip' style='background:{node2color[nid]}'></span> {nid}")
    node_legend_html = "<div class='legend'><b>Nodes</b>: " + " ".join(legend_node_items) + "</div>"

    legend_edge_items = []
    for k in edge_types_present[:30]:
        title_txt = _html.escape(edge_type_labels.get(k, f"etype {k}"))
        legend_edge_items.append(f"<span class='chip' style='background:{et2color[k]}'></span> {title_txt}")
    edge_legend_html = "<div class='legend'><b>Edges</b>: " + (" ".join(legend_edge_items) if legend_edge_items else "(no edge types)") + "</div>"

    legend_html = node_legend_html + edge_legend_html

    # --- 7) 左ペイン：コードHTML ---
    code_html = _build_code_html(code, node_lines, node2color, title="Function Source")

    # --- 8) 親ページ（HTML本体 + 連動スクリプト） ---
    css = """
    body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; margin:0; }
    .wrap { display:flex; gap:10px; padding:10px; }
    .pane { flex:1; border:1px solid #ddd; border-radius:8px; padding:8px; overflow:auto; max-height:88vh; }
    table.codebox { border-collapse: collapse; width: 100%; }
    td.ln { width: 3.5em; color:#888; text-align:right; padding-right:8px; vertical-align: top; user-select:none; }
    td.code { white-space: pre; }
    h3 { margin: 6px 0 10px; }
    .legend { margin-bottom: 6px; font-size: 12px; }
    .chip { display:inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 4px; vertical-align: middle; }
    tr.code-row.hl > td.code { outline: 2px solid #ff9800; background: #fff3e0 !important; }
    """
    node_lines_json = json.dumps({int(k): v for k, v in node_lines.items()}, ensure_ascii=False)
    interop_js = f"""
    <script>
    (function() {{
      const iframe = document.getElementById('graph_iframe');
      const node2lines = {node_lines_json};

      function setCodeHighlight(nodeIds) {{
        document.querySelectorAll('tr.code-row.hl').forEach(tr => tr.classList.remove('hl'));
        const lines = new Set();
        (nodeIds || []).forEach(id => (node2lines[id] || []).forEach(li => lines.add(li)));
        lines.forEach(li => {{
          const tr = document.getElementById('L' + li);
          if (tr) tr.classList.add('hl');
        }});
        // 一番小さい行へスクロール
        if (lines.size > 0) {{
          const minLine = Math.min.apply(null, Array.from(lines));
          const tr = document.getElementById('L' + minLine);
          if (tr) tr.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
        }}
      }}

      // iframe -> 親：ノード選択通知
      window.addEventListener('message', (event) => {{
        if (!event.data || !event.data.type) return;
        if (event.data.type === 'highlight_lines') {{
          setCodeHighlight(event.data.ids || []);
        }}
      }});

      // 親 -> iframe：行ホバーでノード選択
      function postToIframe(ids) {{
        if (!iframe || !iframe.contentWindow) return;
        iframe.contentWindow.postMessage({{ type: 'highlight_nodes', ids: ids || [] }}, '*');
      }}

      // 行ホバーで連動
      document.querySelectorAll('tr.code-row').forEach(tr => {{
        tr.addEventListener('mouseenter', () => {{
          const s = tr.getAttribute('data-nodes') || '';
          const ids = s.split(',').filter(Boolean).map(x => parseInt(x));
          postToIframe(ids);
          setCodeHighlight(ids);
        }});
        tr.addEventListener('mouseleave', () => {{
          postToIframe([]);
          setCodeHighlight([]);
        }});
      }});
    }})();
    </script>
    """

    html_all = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>{_html.escape(title or "Graph & Code")}</title>
<style>{css}</style></head>
<body>
  <div class="wrap">
    {code_html}
    {_build_graph_html(graph_div, legend_html, title="Graph")}
  </div>
  {interop_js if pyvis_ok else ""}
</body></html>
"""
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(out_html).write_text(html_all, encoding="utf-8")
    return out_html




def _inject_pyvis_bridge(embed_path: Path):
    """
    PyVis が生成した HTML の末尾に、親<->iframe 連携のJSを差し込む。
    - 親から {type:'highlight_nodes', ids:[...]} を受けると、該当ノードを選択＆フォーカス
    - ノード選択/解除時に {type:'highlight_lines', ids:[...]} を親へ postMessage
    """
    txt = embed_path.read_text(encoding="utf-8")

    # PyVisは vis.Network を 'network' という変数名で作るのが標準。
    # 念のため存在チェックの try を入れて安全に動かす。
    bridge = r"""
<script>
(function(){
  function safe(fn){ try{ fn(); }catch(e){} }
  window.addEventListener('message', function(event){
    if (!event.data || event.data.type !== 'highlight_nodes') return;
    safe(function(){
      if (typeof network === 'undefined') return;
      const ids = event.data.ids || [];
      network.unselectAll();
      if (ids.length) {
        network.selectNodes(ids);
        network.fit({nodes: ids, animation: true});
      }
    });
  });

  // ノード選択 → 親に該当ノードIDを投げる
  safe(function(){
    if (typeof network === 'undefined') return;
    network.on("selectNode", function(params){
      const ids = (params && params.nodes) ? params.nodes : [];
      parent.postMessage({type: 'highlight_lines', ids: ids}, "*");
    });
    network.on("deselectNode", function(){
      parent.postMessage({type: 'highlight_lines', ids: []}, "*");
    });
  });
})();
</script>
"""
    # </body> の直前に差し込む
    if "</body>" in txt:
        txt = txt.replace("</body>", bridge + "\n</body>")
    else:
        txt = txt + bridge
    embed_path.write_text(txt, encoding="utf-8")
