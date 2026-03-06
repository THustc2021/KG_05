# app.py
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components


# =========================
# 基础函数
# =========================

DEFAULT_ONTOLOGY_PATH = Path("ontology.json")
DEFAULT_KG_DIR = Path("PerPaperKG")


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_uploaded_json(uploaded_file) -> Any:
    return json.load(uploaded_file)


def is_nonempty_string(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""


def as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def uniq_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for e in edges:
        key = (e["from"], e["to"], e["label"], e["source"])
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out

# =========================
# Ontology 严格解析
# =========================

def normalize_type_str(type_str: str) -> str:
    return re.sub(r"\s+", " ", str(type_str).strip())


def strip_optional(type_str: str) -> str:
    s = normalize_type_str(type_str)
    if s.startswith("optional "):
        s = s[len("optional "):].strip()
    return s


def split_union_types(type_str: str) -> List[str]:
    return [x.strip() for x in strip_optional(type_str).split("|")]


def is_list_type(type_str: str) -> bool:
    return "[]" in normalize_type_str(type_str)


def strip_list_suffix(type_name: str) -> str:
    return re.sub(r"\s*\[\]\s*$", "", normalize_type_str(type_name))


def get_entity_id_field(entity_name: str, entity_spec: Dict[str, Any]) -> str:
    """
    严格规则：
    每个实体必须且只能有一个属性 whose type == "string" 且属性名以 _id 结尾，并作为主键。
    """
    props = entity_spec["properties"]
    candidates = []
    for prop_name, prop_spec in props.items():
        if prop_name.endswith("_id") and normalize_type_str(prop_spec["type"]) == "string":
            candidates.append(prop_name)

    if len(candidates) != 1:
        raise ValueError(
            f"实体 {entity_name} 无法唯一确定主键字段。候选={candidates}"
        )
    return candidates[0]


def build_ontology_schema(ontology: Dict[str, Any]) -> Dict[str, Any]:
    if "entities" not in ontology or "relationships" not in ontology:
        raise ValueError("ontology 必须包含 entities 和 relationships 字段。")

    entities = ontology["entities"]
    relationships = ontology["relationships"]

    if not isinstance(entities, dict):
        raise ValueError("ontology.entities 必须是对象。")
    if not isinstance(relationships, list):
        raise ValueError("ontology.relationships 必须是数组。")

    entity_id_field: Dict[str, str] = {}
    entity_allowed_props: Dict[str, set] = {}
    entity_ref_props: Dict[str, List[Dict[str, Any]]] = {}
    entity_id_type_token_to_entity: Dict[str, str] = {}
    relationship_defs: Dict[str, Dict[str, str]] = {}

    # 1) 实体 schema
    for entity_name, entity_spec in entities.items():
        if not isinstance(entity_spec, dict) or "properties" not in entity_spec:
            raise ValueError(f"ontology.entities.{entity_name} 格式非法。")

        props = entity_spec["properties"]
        if not isinstance(props, dict):
            raise ValueError(f"ontology.entities.{entity_name}.properties 必须是对象。")

        id_field = get_entity_id_field(entity_name, entity_spec)
        entity_id_field[entity_name] = id_field
        entity_allowed_props[entity_name] = set(props.keys())
        entity_id_type_token_to_entity[id_field] = entity_name

    # 2) 属性引用 schema
    for entity_name, entity_spec in entities.items():
        props = entity_spec["properties"]
        ref_specs = []

        for prop_name, prop_spec in props.items():
            type_str = normalize_type_str(prop_spec["type"])
            union_members = split_union_types(type_str)

            target_entities = []
            for member in union_members:
                member = strip_list_suffix(member)
                if member in entity_id_type_token_to_entity:
                    target_entities.append(entity_id_type_token_to_entity[member])

            if target_entities:
                ref_specs.append({
                    "property_name": prop_name,
                    "target_entities": target_entities,
                    "is_list": is_list_type(type_str),
                    "type_str": type_str,
                })

        entity_ref_props[entity_name] = ref_specs

    # 3) relationship schema
    for rel in relationships:
        if not isinstance(rel, dict):
            raise ValueError("ontology.relationships 中每个元素都必须是对象。")

        required = {"relationship_id", "from_entity", "to_entity"}
        missing = required - set(rel.keys())
        if missing:
            raise ValueError(f"ontology.relationships 中存在缺字段: {missing}")

        rid = rel["relationship_id"]
        if rid in relationship_defs:
            raise ValueError(f"ontology.relationships 中 relationship_id 重复: {rid}")

        relationship_defs[rid] = {
            "from_entity": rel["from_entity"],
            "to_entity": rel["to_entity"],
        }

    return {
        "entity_id_field": entity_id_field,
        "entity_allowed_props": entity_allowed_props,
        "entity_ref_props": entity_ref_props,
        "relationship_defs": relationship_defs,
        "entities": entities,
    }


# =========================
# KG 严格解析
# =========================

def validate_top_level_kg_shape(kg: Dict[str, Any]) -> List[str]:
    errors = []

    required_top_keys = {"title", "year", "authors", "journal", "doi", "entities", "relationships"}
    actual_keys = set(kg.keys())

    missing = required_top_keys - actual_keys
    extra = actual_keys - required_top_keys

    if missing:
        errors.append(f"KG 顶层缺少字段: {sorted(missing)}")
    if extra:
        errors.append(f"KG 顶层存在非法字段: {sorted(extra)}")

    if "entities" in kg and not isinstance(kg["entities"], dict):
        errors.append("KG.entities 必须是对象。")
    if "relationships" in kg and not isinstance(kg["relationships"], list):
        errors.append("KG.relationships 必须是数组。")

    return errors


def parse_entities_strict(
    kg: Dict[str, Any],
    schema: Dict[str, Any]
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    errors = []
    nodes_by_id: Dict[str, Dict[str, Any]] = {}

    ontology_entities = schema["entities"]
    entity_id_field = schema["entity_id_field"]
    entity_allowed_props = schema["entity_allowed_props"]

    kg_entities = kg["entities"]

    # entities 下只能出现 ontology 中定义的实体类型
    extra_entity_types = set(kg_entities.keys()) - set(ontology_entities.keys())
    if extra_entity_types:
        errors.append(f"KG.entities 中存在非法实体类型: {sorted(extra_entity_types)}")

    for entity_name in ontology_entities.keys():
        if entity_name not in kg_entities:
            continue

        items = kg_entities[entity_name]
        if not isinstance(items, list):
            errors.append(f"KG.entities.{entity_name} 必须是数组。")
            continue

        id_field = entity_id_field[entity_name]
        allowed_props = entity_allowed_props[entity_name]

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(f"KG.entities.{entity_name}[{idx}] 必须是对象。")
                continue

            extra_props = set(item.keys()) - allowed_props
            if extra_props:
                errors.append(
                    f"KG.entities.{entity_name}[{idx}] 存在非法属性: {sorted(extra_props)}"
                )

            if id_field not in item:
                errors.append(
                    f"KG.entities.{entity_name}[{idx}] 缺少主键字段 {id_field}"
                )
                continue

            entity_id = item[id_field]
            if not is_nonempty_string(entity_id):
                errors.append(
                    f"KG.entities.{entity_name}[{idx}].{id_field} 必须是非空字符串"
                )
                continue

            if entity_id in nodes_by_id:
                errors.append(f"实体 id 重复: {entity_id}")
                continue

            nodes_by_id[entity_id] = {
                "id": entity_id,
                "entity_type": entity_name,
                "data": item,
            }

    return nodes_by_id, errors


def parse_relationships_strict(
    kg: Dict[str, Any],
    schema: Dict[str, Any],
    nodes_by_id: Dict[str, Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    errors = []
    rels = []

    relationship_defs = schema["relationship_defs"]

    for idx, rel in enumerate(kg["relationships"]):
        if not isinstance(rel, dict):
            errors.append(f"KG.relationships[{idx}] 必须是对象。")
            continue

        required = {"relationship_id", "from_entity", "to_entity"}
        actual = set(rel.keys())

        missing = required - actual
        extra = actual - required

        if missing:
            errors.append(f"KG.relationships[{idx}] 缺少字段: {sorted(missing)}")
            continue
        if extra:
            errors.append(f"KG.relationships[{idx}] 存在非法字段: {sorted(extra)}")
            continue

        rid = rel["relationship_id"]
        from_entity = rel["from_entity"]
        to_entity = rel["to_entity"]

        if not is_nonempty_string(rid):
            errors.append(f"KG.relationships[{idx}].relationship_id 必须是非空字符串")
            continue
        if not is_nonempty_string(from_entity):
            errors.append(f"KG.relationships[{idx}].from_entity 必须是非空字符串")
            continue
        if not is_nonempty_string(to_entity):
            errors.append(f"KG.relationships[{idx}].to_entity 必须是非空字符串")
            continue

        if rid not in relationship_defs:
            errors.append(f"KG.relationships[{idx}] 使用了 ontology 未定义的 relationship_id: {rid}")
            continue

        if from_entity not in nodes_by_id:
            errors.append(f"KG.relationships[{idx}] 的 from_entity 不存在: {from_entity}")
            continue
        if to_entity not in nodes_by_id:
            errors.append(f"KG.relationships[{idx}] 的 to_entity 不存在: {to_entity}")
            continue

        expected_from = relationship_defs[rid]["from_entity"]
        expected_to = relationship_defs[rid]["to_entity"]
        actual_from = nodes_by_id[from_entity]["entity_type"]
        actual_to = nodes_by_id[to_entity]["entity_type"]

        if actual_from != expected_from:
            errors.append(
                f"KG.relationships[{idx}] 类型 {rid} 的 from_entity={from_entity} 实体类型应为 {expected_from}，实际为 {actual_from}"
            )
            continue

        if actual_to != expected_to:
            errors.append(
                f"KG.relationships[{idx}] 类型 {rid} 的 to_entity={to_entity} 实体类型应为 {expected_to}，实际为 {actual_to}"
            )
            continue

        rels.append({
            "relationship_id": rid,
            "from_entity": from_entity,
            "to_entity": to_entity,
            "source": "explicit",
        })

    return rels, errors


def validate_property_references(
    schema: Dict[str, Any],
    nodes_by_id: Dict[str, Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    errors = []
    derived_rels = []

    entity_ref_props = schema["entity_ref_props"]

    for node_id, node in nodes_by_id.items():
        entity_type = node["entity_type"]
        data = node["data"]

        for ref_spec in entity_ref_props[entity_type]:
            prop_name = ref_spec["property_name"]
            target_entities = set(ref_spec["target_entities"])
            values = as_list(data.get(prop_name))

            for v in values:
                if v is None:
                    continue

                if not is_nonempty_string(v):
                    errors.append(
                        f"实体 {node_id} 的属性 {prop_name} 包含非法值，必须是字符串 id"
                    )
                    continue

                if v not in nodes_by_id:
                    errors.append(
                        f"实体 {node_id} 的属性 {prop_name} 引用了不存在的 id: {v}"
                    )
                    continue

                actual_target_type = nodes_by_id[v]["entity_type"]
                if actual_target_type not in target_entities:
                    errors.append(
                        f"实体 {node_id} 的属性 {prop_name} 引用 id={v}，其实体类型为 {actual_target_type}，"
                        f"但 ontology 要求目标类型为 {sorted(target_entities)}"
                    )
                    continue

                derived_rels.append({
                    "relationship_id": f"ATTR::{prop_name}",
                    "from_entity": node_id,
                    "to_entity": v,
                    "source": "property",
                })

    return derived_rels, errors


# =========================
# 可视化
# =========================
import re
import textwrap
from typing import Dict, Any, List
from pyvis.network import Network


def wrap_text(text: Any, width: int = 48) -> str:
    s = "" if text is None else str(text)
    return textwrap.fill(s, width=width, break_long_words=False, break_on_hyphens=False)


def pretty_label(name: str, max_line_len: int = 14) -> str:
    parts = re.findall(r"[A-Z][a-z0-9]*|[A-Z]+(?=[A-Z][a-z]|$)", str(name))
    if not parts:
        return str(name)

    lines = []
    current = []
    current_len = 0

    for p in parts:
        add_len = len(p) if not current else len(p) + 1
        if current and current_len + add_len > max_line_len:
            lines.append(" ".join(current))
            current = [p]
            current_len = len(p)
        else:
            current.append(p)
            current_len += add_len

    if current:
        lines.append(" ".join(current))

    return "\n".join(lines)


def format_tooltip_value(v: Any, width: int = 48) -> str:
    if v is None:
        return "None"

    if isinstance(v, list):
        if not v:
            return "[]"
        out = []
        for item in v:
            wrapped = wrap_text(item, width=width)
            wrapped = wrapped.replace("\n", "\n    ")
            out.append(f"  - {wrapped}")
        return "\n".join(out)

    if isinstance(v, dict):
        if not v:
            return "{}"
        out = []
        for kk, vv in v.items():
            wrapped = wrap_text(vv, width=width)
            wrapped = wrapped.replace("\n", "\n    ")
            out.append(f"  {kk}: {wrapped}")
        return "\n".join(out)

    return wrap_text(v, width=width)


def build_node_tooltip(node_id: str, node: Dict[str, Any], schema: Dict[str, Any]) -> str:
    entity_type = node["entity_type"]
    data = node["data"]

    ref_prop_names = {
        spec["property_name"]
        for spec in schema["entity_ref_props"].get(entity_type, [])
    }
    id_field = schema["entity_id_field"][entity_type]

    lines = [str(node_id), str(entity_type), ""]

    for k, v in data.items():
        if k == id_field:
            continue
        if k in ref_prop_names:
            continue

        formatted = format_tooltip_value(v, width=48)
        if "\n" in formatted:
            lines.append(f"{k}:")
            lines.append(formatted)
        else:
            lines.append(f"{k}: {formatted}")

    return "\n".join(lines)


def build_edge_tooltip(e: Dict[str, Any]) -> str:
    lines = [
        f"type: {e['relationship_id']}",
        f"source: {e['source']}",
    ]
    return "\n".join(lines)


def node_size_by_type(entity_type: str) -> int:
    size_map = {
        "Phenomenon": 42,
        "Effect": 34,
        "Condition": 34,
        "Parameter": 30,
        "Term": 30,
        "SimulationModel": 38,
        "ModelEquation": 36,
        "MaterialSystem": 36,
    }
    return size_map.get(entity_type, 32)


def build_graph(
    nodes_by_id: Dict[str, Dict[str, Any]],
    edges_raw: List[Dict[str, Any]],
    schema: Dict[str, Any],
):
    nodes = []
    edges = []

    for node_id, node in nodes_by_id.items():
        entity_type = node["entity_type"]
        nodes.append({
            "id": node_id,
            "label": pretty_label(node_id),
            "group": entity_type,
            "title": build_node_tooltip(node_id, node, schema),
            "size": node_size_by_type(entity_type),
        })

    for e in edges_raw:
        edges.append({
            "from": e["from_entity"],
            "to": e["to_entity"],
            "label": e["relationship_id"],
            "title": build_edge_tooltip(e),
            "source": e["source"],
            "arrows": "to",
        })

    return nodes, uniq_edges(edges)


def render_graph(nodes, edges, output_html="graph.html"):
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
        directed=True,
    )

    for n in nodes:
        net.add_node(
            n["id"],
            label=n["label"],
            title=n["title"],
            group=n["group"],
            size=n.get("size", 32),
        )

    for e in edges:
        net.add_edge(
            e["from"],
            e["to"],
            label=e["label"],
            title=e["title"],
            arrows=e.get("arrows", "to"),
        )

    net.set_options("""
    const options = {
      "interaction": {
        "hover": true,
        "tooltipDelay": 120,
        "navigationButtons": true,
        "keyboard": true
      },
      "nodes": {
        "shape": "dot",
        "scaling": {
          "min": 20,
          "max": 50
        },
        "font": {
          "size": 22,
          "face": "Arial",
          "multi": "md",
          "vadjust": -2
        },
        "borderWidth": 2
      },
      "edges": {
        "smooth": {
          "type": "dynamic"
        },
        "width": 1.6,
        "selectionWidth": 2.5,
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.7
          }
        },
        "font": {
          "size": 10,
          "align": "middle",
          "strokeWidth": 0
        }
      },
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -5000,
          "centralGravity": 0.18,
          "springLength": 165,
          "springConstant": 0.03,
          "damping": 0.18,
          "avoidOverlap": 0.2
        },
        "minVelocity": 0.75
      }
    }
    """)

    net.write_html(output_html)


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="Strict Per-Paper KG Visualizer", layout="wide")
st.title("Strict Per-Paper Knowledge Graph Visualizer")

st.markdown(
    """
严格模式：

- 完全按 ontology 原样校验
- 不做格式兼容
- 检查非法属性
- 检查所有 id 引用完整性
- 只有检查全部通过才可视化
"""
)

with st.sidebar:
    st.header("输入")
    ontology_upload = st.file_uploader("可选：上传 ontology.json", type=["json"])
    kg_upload = st.file_uploader("上传知识图谱 JSON（可选，不上传则默认从 PerPaperKG 加载）", type=["json"])


# 1) 读取 ontology
try:
    if ontology_upload is not None:
        ontology = load_uploaded_json(ontology_upload)
        ontology_source = ontology_upload.name
    else:
        if not DEFAULT_ONTOLOGY_PATH.exists():
            st.error(f"未找到默认 ontology 文件：{DEFAULT_ONTOLOGY_PATH}")
            st.stop()
        ontology = load_json(DEFAULT_ONTOLOGY_PATH)
        ontology_source = str(DEFAULT_ONTOLOGY_PATH)

    schema = build_ontology_schema(ontology)
except Exception as e:
    st.error(f"ontology 解析失败：{e}")
    st.stop()

st.success(f"Ontology 已加载：{ontology_source}")


# 2) 读取 KG
try:
    if kg_upload is not None:
        kg = load_uploaded_json(kg_upload)
        kg_source = kg_upload.name
    else:
        if not DEFAULT_KG_DIR.exists():
            st.error(f"未找到默认 KG 目录：{DEFAULT_KG_DIR}")
            st.stop()

        kg_files = sorted(DEFAULT_KG_DIR.glob("*.json"))
        if not kg_files:
            st.error("PerPaperKG 目录下没有 json 文件。")
            st.stop()

        selected = st.selectbox(
            "选择 PerPaperKG 中的知识图谱文件",
            options=kg_files,
            format_func=lambda p: p.name
        )
        kg = load_json(selected)
        kg_source = str(selected)
except Exception as e:
    st.error(f"KG 读取失败：{e}")
    st.stop()

st.info(f"当前 KG：{kg_source}")


# 3) 顶层结构检查
top_errors = validate_top_level_kg_shape(kg)
if top_errors:
    st.error("KG 顶层结构不合法。")
    for err in top_errors:
        st.error(err)
    st.stop()


# 4) 实体检查
nodes_by_id, entity_errors = parse_entities_strict(kg, schema)
if entity_errors:
    st.error("实体检查未通过。")
    with st.expander("实体错误详情", expanded=True):
        for err in entity_errors:
            st.error(err)
    st.stop()


# 5) relationship 检查
explicit_rels, rel_errors = parse_relationships_strict(kg, schema, nodes_by_id)
if rel_errors:
    st.error("显式 relationship 检查未通过。")
    with st.expander("relationship 错误详情", expanded=True):
        for err in rel_errors:
            st.error(err)
    st.stop()


# 6) 属性引用检查 + 额外边抽取
property_rels, prop_errors = validate_property_references(schema, nodes_by_id)
if prop_errors:
    st.error("属性引用检查未通过。")
    with st.expander("属性引用错误详情", expanded=True):
        for err in prop_errors:
            st.error(err)
    st.stop()

st.success("完整性检查通过，可以可视化。")


# 7) 控制显示
col_a, col_b, col_c = st.columns(3)
col_a.metric("实体数", len(nodes_by_id))
col_b.metric("显式关系数", len(explicit_rels))
col_c.metric("属性抽取关系数", len(property_rels))

show_explicit = st.checkbox("显示显式 relationship", value=True)
show_property = st.checkbox("显示属性抽取 relationship", value=True)

edges_raw = []
if show_explicit:
    edges_raw.extend(explicit_rels)
if show_property:
    edges_raw.extend(property_rels)

nodes, edges = build_graph(nodes_by_id, edges_raw, schema)

left, right = st.columns([3, 1])

with left:
    st.subheader("Knowledge Graph")
    render_graph(nodes, edges, height=880)

with right:
    st.subheader("统计")

    entity_type_count = {}
    for node in nodes_by_id.values():
        entity_type_count[node["entity_type"]] = entity_type_count.get(node["entity_type"], 0) + 1

    edge_type_count = {}
    for e in edges:
        edge_type_count[e["label"]] = edge_type_count.get(e["label"], 0) + 1

    st.markdown("**实体类型分布**")
    st.json(entity_type_count)

    st.markdown("**边类型分布**")
    st.json(edge_type_count)

with st.expander("Ontology 解析结果", expanded=False):
    st.json({
        "entity_id_field": schema["entity_id_field"],
        "entity_ref_props": schema["entity_ref_props"],
        "relationship_defs": schema["relationship_defs"],
    })

with st.expander("显式 relationship", expanded=False):
    st.json(explicit_rels)

with st.expander("属性抽取 relationship", expanded=False):
    st.json(property_rels)