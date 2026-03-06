import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components


st.set_page_config(page_title="KG Viewer", layout="wide")


# ---------- Helpers ----------
def get_entity_uid(entity_type: str, entity: Dict[str, Any]) -> str:
    # 1 优先寻找 *_id
    for k, v in entity.items():
        if k.endswith("_id") and v:
            return str(v)

    # 2 再找 id
    if "id" in entity:
        return str(entity["id"])

    # 3 最后 fallback
    return f"{entity_type}_{abs(hash(json.dumps(entity, sort_keys=True)))}"

import hashlib

def color_from_type(entity_type: str) -> str:
    h = hashlib.md5(entity_type.encode()).hexdigest()
    return f"#{h[:6]}"

def get_entity_label(entity_type: str, entity: Dict[str, Any]) -> str:
    for key in ["name", "title", "symbol", "description"]:
        if key in entity and entity[key]:
            text = str(entity[key])
            break
    else:
        text = get_entity_uid(entity_type, entity)
    if len(text) > 42:
        text = text[:39] + "..."
    return f"{entity_type}\n{text}"


def build_tooltip(entity_type: str, entity: Dict[str, Any]) -> str:
    parts = [f"<b>{entity_type}</b>"]
    for k, v in entity.items():
        if v in ("", None, [], {}):
            continue
        if isinstance(v, list):
            v = ", ".join(map(str, v))
        parts.append(f"<b>{k}</b>: {v}")
    return "<br>".join(parts)


def parse_graph(data: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Supports either:
    1) {"entities": {"Paper":[...], "Model":[...]}, "relationships":[...]}
    2) {"entities": {"paper_1": {"type":"Paper", ...}, ...}, "relationships":[...]}
    """
    raw_entities = data.get("entities", {})
    relationships = data.get("relationships", [])

    nodes: Dict[str, Dict[str, Any]] = {}

    # Style 1: grouped by entity type
    if raw_entities and all(isinstance(v, list) for v in raw_entities.values()):
        for entity_type, items in raw_entities.items():
            for item in items:
                uid = get_entity_uid(entity_type, item)
                nodes[uid] = {
                    "id": uid,
                    "type": entity_type,
                    "data": item,
                }

    # Style 2: flat dict keyed by arbitrary keys, each item has "type"
    elif raw_entities and all(isinstance(v, dict) for v in raw_entities.values()):
        for _, item in raw_entities.items():
            entity_type = item.get("type", "Unknown")
            uid = get_entity_uid(entity_type, item)
            nodes[uid] = {
                "id": uid,
                "type": entity_type,
                "data": item,
            }

    return nodes, relationships


def build_network(nodes: Dict[str, Dict[str, Any]], relationships: List[Dict[str, Any]]) -> Network:
    net = Network(height="780px", width="100%", bgcolor="#ffffff", font_color="#222222", directed=True)
    net.barnes_hut()

    for node_id, node_obj in nodes.items():
        entity_type = node_obj["type"]
        entity = node_obj["data"]
        label = get_entity_label(entity_type, entity)
        title = build_tooltip(entity_type, entity)

        net.add_node(
            node_id,
            label=label,
            title=title,
            color=color_from_type(entity_type),
            shape="dot",
            size=18,
        )

    for rel in relationships:
        src = rel.get("from_entity_id")
        dst = rel.get("to_entity_id")
        if src not in nodes or dst not in nodes:
            continue

        rel_type = rel.get("relationship_id", "RELATED_TO")
        desc = rel.get("description", "")
        evidence = rel.get("evidence", "")

        edge_title_parts = [f"<b>{rel_type}</b>"]
        if desc:
            edge_title_parts.append(f"<b>description</b>: {desc}")
        if evidence:
            edge_title_parts.append(f"<b>evidence</b>: {evidence}")
        edge_title = "<br>".join(edge_title_parts)

        net.add_edge(
            src,
            dst,
            label=rel_type,
            title=edge_title,
            arrows="to",
            smooth=False,
        )

    net.set_options(
        """
        const options = {
          "nodes": {
            "font": {
              "size": 14,
              "multi": "html"
            }
          },
          "edges": {
            "font": {
              "size": 11,
              "align": "middle"
            },
            "color": {
              "color": "#888888",
              "highlight": "#333333"
            }
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          },
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -4500,
              "springLength": 160,
              "springConstant": 0.03
            },
            "minVelocity": 0.75
          }
        }
        """
    )
    return net


def filter_graph(
    nodes: Dict[str, Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    selected_types: List[str],
    keyword: str,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    keyword = keyword.strip().lower()

    kept_nodes = {}
    for node_id, node_obj in nodes.items():
        entity_type = node_obj["type"]
        entity = node_obj["data"]

        if selected_types and entity_type not in selected_types:
            continue

        text_blob = json.dumps(entity, ensure_ascii=False).lower()
        if keyword and keyword not in text_blob:
            continue

        kept_nodes[node_id] = node_obj

    kept_rels = [
        r for r in relationships
        if r.get("from_entity_id") in kept_nodes and r.get("to_entity_id") in kept_nodes
    ]
    return kept_nodes, kept_rels


# ---------- UI ----------

st.title("Scientific Knowledge Graph Viewer")

st.markdown(
    """
Upload a JSON knowledge graph file or paste JSON directly.
Supported formats:

1. `{"entities": {"Paper":[...], "Model":[...]}, "relationships":[...]}`
2. `{"entities": {"paper_1": {"type":"Paper", ...}, ...}, "relationships":[...]}`
"""
)

left, right = st.columns([1, 1])

with left:
    uploaded_file = st.file_uploader("Upload JSON", type=["json"])

with right:
    example_path = st.text_input("Or load local JSON path", value="")

raw_text = st.text_area("Or paste JSON here", height=220)

data = None
load_error = None

try:
    if uploaded_file is not None:
        data = json.load(uploaded_file)
    elif raw_text.strip():
        data = json.loads(raw_text)
    elif example_path.strip():
        data = json.loads(Path(example_path).read_text(encoding="utf-8"))
except Exception as e:
    load_error = str(e)

if load_error:
    st.error(f"Failed to load JSON: {load_error}")

if data:
    nodes, relationships = parse_graph(data)

    if not nodes:
        st.warning("No entities found.")
        st.stop()

    all_types = sorted({n["type"] for n in nodes.values()})

    with st.sidebar:
        st.header("Filters")
        selected_types = st.multiselect("Entity types", all_types, default=all_types)
        keyword = st.text_input("Keyword")
        show_table = st.checkbox("Show entity table", value=True)
        show_rel_table = st.checkbox("Show relationship table", value=True)

    filtered_nodes, filtered_relationships = filter_graph(
        nodes, relationships, selected_types, keyword
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Nodes", len(filtered_nodes))
    c2.metric("Edges", len(filtered_relationships))
    c3.metric("Entity types", len({n['type'] for n in filtered_nodes.values()}))

    net = build_network(filtered_nodes, filtered_relationships)
    html = net.generate_html()

    components.html(html, height=800, scrolling=True)

    if show_table:
        st.subheader("Entities")
        entity_rows = []
        for node_id, node_obj in filtered_nodes.items():
            row = {
                "id": node_id,
                "type": node_obj["type"],
            }
            row.update(node_obj["data"])
            entity_rows.append(row)
        st.dataframe(entity_rows, use_container_width=True, height=320)

    if show_rel_table:
        st.subheader("Relationships")
        st.dataframe(filtered_relationships, use_container_width=True, height=260)