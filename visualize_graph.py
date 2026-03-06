import json
import re
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components


st.set_page_config(page_title="Ontology-Driven KG Viewer", layout="wide")


# =========================================================
# Generic helpers
# =========================================================

def color_from_type(entity_type: str) -> str:
    h = hashlib.md5(entity_type.encode()).hexdigest()
    return f"#{h[:6]}"


def normalize_text(s: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", s.lower())


def first_id_key_in_dict(d: Dict[str, Any]) -> Optional[str]:
    for k in d.keys():
        if k.endswith("_id"):
            return k
    return "id" if "id" in d else None


def get_raw_entity_id(entity: Dict[str, Any]) -> str:
    id_key = first_id_key_in_dict(entity)
    if id_key and entity.get(id_key):
        return str(entity[id_key])
    return f"anon_{abs(hash(json.dumps(entity, sort_keys=True, ensure_ascii=False)))}"


def get_entity_uid(entity_type: str, entity: Dict[str, Any]) -> str:
    return f"{entity_type}:{get_raw_entity_id(entity)}"


def get_entity_label(entity_type: str, entity: Dict[str, Any]) -> str:
    raw_id = get_raw_entity_id(entity)

    preferred_keys = []
    if entity_type == "Parameter":
        preferred_keys = ["parameter_id", "symbol", "description"]
    elif entity_type == "ModelEquation":
        preferred_keys = ["equation_id", "description"]
    elif entity_type == "SimulationModel":
        preferred_keys = ["model_id", "objective", "description"]
    elif entity_type == "Phenomenon":
        preferred_keys = ["phenomenon_id", "description"]
    elif entity_type == "Condition":
        preferred_keys = ["condition_id", "description"]
    elif entity_type == "Effect":
        preferred_keys = ["effect_id", "description"]
    elif entity_type == "MaterialSystem":
        preferred_keys = ["material_system_id", "composition"]
    elif entity_type == "Term":
        preferred_keys = ["term_id", "description"]
    else:
        preferred_keys = ["name", "title", "symbol", "description"]

    text = None
    for key in preferred_keys:
        if key in entity and entity[key]:
            if entity_type == "Parameter" and key == "parameter_id":
                symbol = entity.get("symbol", "")
                text = f"{entity[key]} ({symbol})" if symbol else str(entity[key])
            else:
                text = str(entity[key])
            break

    if not text:
        text = raw_id

    if len(text) > 52:
        text = text[:49] + "..."
    return f"{entity_type}\n{text}"


def build_tooltip(entity_type: str, entity: Dict[str, Any]) -> str:
    parts = [f"<b>{entity_type}</b>"]
    for k, v in entity.items():
        if v in ("", None, [], {}):
            continue
        if isinstance(v, list):
            v = ", ".join(map(str, v))
        else:
            v = str(v)

        if len(v) > 500:
            v = v[:500] + "..."
        parts.append(f"<b>{k}</b>: {v}")
    return "<br>".join(parts)


# =========================================================
# Ontology parsing
# =========================================================

def extract_entities_schema(ontology: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    entities = ontology.get("entities", {})
    if not isinstance(entities, dict):
        raise ValueError("Ontology must contain an 'entities' object.")
    return entities


def extract_relationship_schema(ontology: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    rels = ontology.get("relationships", [])
    result = {}
    if isinstance(rels, list):
        for rel in rels:
            rid = rel.get("relationship_id") or rel.get("type")
            if rid:
                result[rid] = {
                    "from_entity": rel.get("from_entity") or rel.get("from"),
                    "to_entity": rel.get("to_entity") or rel.get("to"),
                    "description": rel.get("description", "")
                }
    return result


def choose_primary_id_field(entity_type: str, entity_schema: Dict[str, Any]) -> str:
    properties = entity_schema.get("properties", {})
    if not isinstance(properties, dict) or not properties:
        raise ValueError(f"Ontology entity '{entity_type}' has no properties.")

    for prop_name in properties.keys():
        if prop_name.endswith("_id"):
            return prop_name

    raise ValueError(f"Ontology entity '{entity_type}' does not define any *_id property.")


def extract_reference_tokens(type_str: str) -> List[str]:
    """
    从 ontology 属性类型中提取 *_id token，例如：
    - parameter_id
    - effect_id
    - system_id
    - optional condition_id []
    """
    if not isinstance(type_str, str):
        return []

    tokens = re.findall(r"\b[a-zA-Z_]+_id\b", type_str)
    return list(dict.fromkeys(tokens))


def resolve_reference_token_to_entity_type(
    ref_token: str,
    entity_primary_id_fields: Dict[str, str]
) -> str:
    """
    通过 ontology 中各实体主 id 字段，自动把引用 token 映射到目标实体类型。
    不使用预定义字段。
    """
    token_n = normalize_text(ref_token)
    scores: List[Tuple[int, str]] = []

    for entity_type, id_field in entity_primary_id_fields.items():
        field_n = normalize_text(id_field)
        entity_n = normalize_text(entity_type)

        score = -1
        if ref_token == id_field:
            score = 100
        elif token_n == field_n:
            score = 95
        elif field_n.endswith(token_n) or token_n.endswith(field_n):
            score = 80
        elif token_n.replace("_id", "") == entity_n:
            score = 70
        elif token_n.replace("_id", "") in field_n.replace("_id", ""):
            score = 60
        elif field_n.replace("_id", "") in token_n.replace("_id", ""):
            score = 50

        if score >= 0:
            scores.append((score, entity_type))

    if not scores:
        raise ValueError(f"Cannot resolve ontology reference token '{ref_token}' to any entity type.")

    scores.sort(reverse=True, key=lambda x: x[0])
    top_score = scores[0][0]
    top = [et for sc, et in scores if sc == top_score]

    if len(top) != 1:
        raise ValueError(
            f"Ambiguous ontology reference token '{ref_token}'. Candidate entity types: {top}"
        )
    return top[0]


def build_ontology_reference_map(
    ontology: Dict[str, Any]
) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """
    返回：
    1) entity_type -> primary_id_field
    2) entity_type -> property_name -> target_entity_type  （只包含引用属性）
    """
    entities_schema = extract_entities_schema(ontology)

    entity_primary_id_fields: Dict[str, str] = {}
    for entity_type, entity_schema in entities_schema.items():
        entity_primary_id_fields[entity_type] = choose_primary_id_field(entity_type, entity_schema)

    property_ref_targets: Dict[str, Dict[str, str]] = {}

    for entity_type, entity_schema in entities_schema.items():
        properties = entity_schema.get("properties", {})
        property_ref_targets[entity_type] = {}

        for prop_name, prop_spec in properties.items():
            if prop_name == entity_primary_id_fields[entity_type]:
                continue

            if not isinstance(prop_spec, dict):
                continue

            type_str = prop_spec.get("type", "")
            ref_tokens = extract_reference_tokens(type_str)
            if not ref_tokens:
                continue

            if len(ref_tokens) > 1:
                raise ValueError(
                    f"Ontology property '{entity_type}.{prop_name}' contains multiple id reference tokens: {ref_tokens}"
                )

            ref_token = ref_tokens[0]
            target_entity_type = resolve_reference_token_to_entity_type(ref_token, entity_primary_id_fields)
            property_ref_targets[entity_type][prop_name] = target_entity_type

    return entity_primary_id_fields, property_ref_targets


def summarize_ontology_reference_graph(
    ontology: Dict[str, Any]
) -> Dict[str, Any]:
    entity_primary_id_fields, property_ref_targets = build_ontology_reference_map(ontology)

    entity_summary = []
    reference_edges = []

    for entity_type, primary_id in entity_primary_id_fields.items():
        refs = property_ref_targets.get(entity_type, {})
        entity_summary.append(
            {
                "entity_type": entity_type,
                "primary_id_field": primary_id,
                "reference_property_count": len(refs),
            }
        )

        for prop_name, target_entity_type in refs.items():
            reference_edges.append(
                {
                    "from_entity": entity_type,
                    "property": prop_name,
                    "to_entity": target_entity_type,
                }
            )

    return {
        "entity_summary": entity_summary,
        "reference_edges": reference_edges,
    }


# =========================================================
# KG parsing
# =========================================================

def parse_kg_entities(kg: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    返回：
    {
      "EntityType": {
         "raw_id": entity_instance_dict
      }
    }
    仅支持 grouped style:
    {
      "entities": {
        "EntityType": [ ... ]
      }
    }
    """
    raw_entities = kg.get("entities", {})
    if not isinstance(raw_entities, dict):
        raise ValueError("KG must contain an 'entities' object.")

    parsed: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for entity_type, items in raw_entities.items():
        if not isinstance(items, list):
            raise ValueError(f"KG entity bucket '{entity_type}' must be a list of entity instances.")

        parsed[entity_type] = {}
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"Entity instance under '{entity_type}' must be an object.")

            raw_id = get_raw_entity_id(item)
            if raw_id in parsed[entity_type]:
                raise ValueError(f"Duplicate entity id '{raw_id}' found in entity type '{entity_type}'.")
            parsed[entity_type][raw_id] = item

    return parsed


def normalize_explicit_relationships(kg: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_relationships = kg.get("relationship_instances") or kg.get("relationships") or []
    if not isinstance(raw_relationships, list):
        raise ValueError("'relationship_instances' or 'relationships' must be a list.")

    normalized = []
    for rel in raw_relationships:
        if not isinstance(rel, dict):
            raise ValueError("Each relationship instance must be an object.")

        normalized.append(
            {
                "relationship_id": rel.get("relationship_id") or rel.get("type"),
                "from_entity_type": rel.get("from_entity_type"),
                "from_entity_id": rel.get("from_entity_id") or rel.get("from"),
                "to_entity_type": rel.get("to_entity_type"),
                "to_entity_id": rel.get("to_entity_id") or rel.get("to"),
                "description": rel.get("description", ""),
                "evidence": rel.get("evidence", ""),
                "source": "explicit",
            }
        )
    return normalized


# =========================================================
# Completeness check
# =========================================================

def validate_kg_completeness(
    kg: Dict[str, Any],
    ontology: Dict[str, Any],
    raise_on_error: bool = True
) -> Tuple[
    Dict[str, Dict[str, Dict[str, Any]]],
    Dict[str, str],
    Dict[str, Dict[str, str]],
    List[Dict[str, Any]],
    Dict[str, Any]
]:
    entities_schema = extract_entities_schema(ontology)
    rel_schema = extract_relationship_schema(ontology)
    entity_primary_id_fields, property_ref_targets = build_ontology_reference_map(ontology)
    kg_entities = parse_kg_entities(kg)
    explicit_relationships = normalize_explicit_relationships(kg)

    errors: List[str] = []
    missing_references: List[Dict[str, Any]] = []
    invalid_relationships: List[Dict[str, Any]] = []

    # 1) KG entity types must exist in ontology
    for entity_type in kg_entities.keys():
        if entity_type not in entities_schema:
            errors.append(f"KG entity type '{entity_type}' is not defined in ontology.")

    # 2) Each entity instance must contain its ontology primary id field
    for entity_type, instances in kg_entities.items():
        if entity_type not in entity_primary_id_fields:
            continue
        primary_id_field = entity_primary_id_fields[entity_type]

        for raw_id, instance in instances.items():
            if primary_id_field not in instance:
                errors.append(
                    f"Entity '{entity_type}:{raw_id}' is missing primary id field '{primary_id_field}'."
                )

    # 3) Property-based reference check
    for entity_type, instances in kg_entities.items():
        if entity_type not in property_ref_targets:
            continue

        ref_props = property_ref_targets[entity_type]
        for raw_id, instance in instances.items():
            for prop_name, target_entity_type in ref_props.items():
                if prop_name not in instance:
                    continue

                value = instance[prop_name]
                if value in ("", None, []):
                    continue

                target_bucket = kg_entities.get(target_entity_type, {})

                if isinstance(value, str):
                    if value not in target_bucket:
                        msg = (
                            f"Broken reference in '{entity_type}:{raw_id}.{prop_name}' -> "
                            f"'{value}' (target entity type: {target_entity_type})"
                        )
                        errors.append(msg)
                        missing_references.append(
                            {
                                "source_entity_type": entity_type,
                                "source_entity_id": raw_id,
                                "property": prop_name,
                                "target_entity_type": target_entity_type,
                                "missing_id": value,
                            }
                        )
                elif isinstance(value, list):
                    for ref_id in value:
                        if not isinstance(ref_id, str):
                            errors.append(
                                f"Invalid non-string reference in '{entity_type}:{raw_id}.{prop_name}': {ref_id}"
                            )
                            continue
                        if ref_id not in target_bucket:
                            msg = (
                                f"Broken reference in '{entity_type}:{raw_id}.{prop_name}' -> "
                                f"'{ref_id}' (target entity type: {target_entity_type})"
                            )
                            errors.append(msg)
                            missing_references.append(
                                {
                                    "source_entity_type": entity_type,
                                    "source_entity_id": raw_id,
                                    "property": prop_name,
                                    "target_entity_type": target_entity_type,
                                    "missing_id": ref_id,
                                }
                            )
                else:
                    errors.append(
                        f"Invalid reference value type in '{entity_type}:{raw_id}.{prop_name}'. "
                        f"Expected string or string list, got {type(value).__name__}."
                    )

    # 4) Explicit relationship validation
    for idx, rel in enumerate(explicit_relationships):
        rid = rel.get("relationship_id")
        f_type = rel.get("from_entity_type")
        f_id = rel.get("from_entity_id")
        t_type = rel.get("to_entity_type")
        t_id = rel.get("to_entity_id")

        if not rid:
            msg = f"Relationship #{idx} is missing 'relationship_id'."
            errors.append(msg)
            invalid_relationships.append({"relationship_id": "", "reason": msg})
            continue

        if rel_schema and rid not in rel_schema:
            msg = f"Relationship #{idx} uses undefined relationship_id '{rid}'."
            errors.append(msg)
            invalid_relationships.append({"relationship_id": rid, "reason": msg})

        if not f_type or not t_type:
            msg = f"Relationship #{idx} ('{rid}') must specify from_entity_type and to_entity_type."
            errors.append(msg)
            invalid_relationships.append({"relationship_id": rid, "reason": msg})
            continue

        if f_type not in kg_entities:
            msg = f"Relationship #{idx} ('{rid}') references unknown from_entity_type '{f_type}'."
            errors.append(msg)
            invalid_relationships.append({"relationship_id": rid, "reason": msg})
        elif f_id not in kg_entities[f_type]:
            msg = f"Relationship #{idx} ('{rid}') references missing from entity '{f_type}:{f_id}'."
            errors.append(msg)
            invalid_relationships.append({"relationship_id": rid, "reason": msg})

        if t_type not in kg_entities:
            msg = f"Relationship #{idx} ('{rid}') references unknown to_entity_type '{t_type}'."
            errors.append(msg)
            invalid_relationships.append({"relationship_id": rid, "reason": msg})
        elif t_id not in kg_entities[t_type]:
            msg = f"Relationship #{idx} ('{rid}') references missing to entity '{t_type}:{t_id}'."
            errors.append(msg)
            invalid_relationships.append({"relationship_id": rid, "reason": msg})

        if rid in rel_schema:
            expected_from = rel_schema[rid].get("from_entity")
            expected_to = rel_schema[rid].get("to_entity")
            if expected_from and f_type != expected_from:
                msg = (
                    f"Relationship #{idx} ('{rid}') has from_entity_type '{f_type}', "
                    f"expected '{expected_from}'."
                )
                errors.append(msg)
                invalid_relationships.append({"relationship_id": rid, "reason": msg})
            if expected_to and t_type != expected_to:
                msg = (
                    f"Relationship #{idx} ('{rid}') has to_entity_type '{t_type}', "
                    f"expected '{expected_to}'."
                )
                errors.append(msg)
                invalid_relationships.append({"relationship_id": rid, "reason": msg})

    validation_report = {
        "passed": len(errors) == 0,
        "errors": errors,
        "missing_references": missing_references,
        "invalid_relationships": invalid_relationships,
    }

    if errors and raise_on_error:
        raise ValueError("KG completeness validation failed:\n- " + "\n- ".join(errors))

    return (
        kg_entities,
        entity_primary_id_fields,
        property_ref_targets,
        explicit_relationships,
        validation_report,
    )


# =========================================================
# Build nodes + relationships
# =========================================================

def build_nodes_from_kg(kg_entities: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    nodes: Dict[str, Dict[str, Any]] = {}
    for entity_type, instances in kg_entities.items():
        for raw_id, entity in instances.items():
            uid = f"{entity_type}:{raw_id}"
            nodes[uid] = {
                "id": uid,
                "raw_id": raw_id,
                "type": entity_type,
                "data": entity,
            }
    return nodes


def explicit_relationships_to_uids(explicit_relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for rel in explicit_relationships:
        out.append(
            {
                **rel,
                "from_uid": f"{rel['from_entity_type']}:{rel['from_entity_id']}",
                "to_uid": f"{rel['to_entity_type']}:{rel['to_entity_id']}",
            }
        )
    return out


def infer_property_relationships(
    kg_entities: Dict[str, Dict[str, Dict[str, Any]]],
    property_ref_targets: Dict[str, Dict[str, str]]
) -> List[Dict[str, Any]]:
    inferred: List[Dict[str, Any]] = []

    for entity_type, instances in kg_entities.items():
        ref_props = property_ref_targets.get(entity_type, {})
        for raw_id, instance in instances.items():
            src_uid = f"{entity_type}:{raw_id}"

            for prop_name, target_entity_type in ref_props.items():
                if prop_name not in instance:
                    continue

                value = instance[prop_name]
                if value in ("", None, []):
                    continue

                if isinstance(value, str):
                    inferred.append(
                        {
                            "relationship_id": prop_name,
                            "from_entity_type": entity_type,
                            "from_entity_id": raw_id,
                            "to_entity_type": target_entity_type,
                            "to_entity_id": value,
                            "description": f"Inferred from property '{prop_name}'.",
                            "evidence": "",
                            "source": "property",
                            "from_uid": src_uid,
                            "to_uid": f"{target_entity_type}:{value}",
                        }
                    )
                elif isinstance(value, list):
                    for target_id in value:
                        inferred.append(
                            {
                                "relationship_id": prop_name,
                                "from_entity_type": entity_type,
                                "from_entity_id": raw_id,
                                "to_entity_type": target_entity_type,
                                "to_entity_id": target_id,
                                "description": f"Inferred from property '{prop_name}'.",
                                "evidence": "",
                                "source": "property",
                                "from_uid": src_uid,
                                "to_uid": f"{target_entity_type}:{target_id}",
                            }
                        )
    return inferred


def deduplicate_relationships(relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    result = []
    for rel in relationships:
        key = (
            rel.get("relationship_id"),
            rel.get("from_uid"),
            rel.get("to_uid"),
            rel.get("source"),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(rel)
    return result


# =========================================================
# Visualization
# =========================================================

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
        src = rel["from_uid"]
        dst = rel["to_uid"]
        if src not in nodes or dst not in nodes:
            continue

        rel_type = rel.get("relationship_id", "RELATED_TO")
        desc = rel.get("description", "")
        evidence = rel.get("evidence", "")
        source = rel.get("source", "")

        tooltip_parts = [f"<b>{rel_type}</b>"]
        if source:
            tooltip_parts.append(f"<b>source</b>: {source}")
        if desc:
            tooltip_parts.append(f"<b>description</b>: {desc}")
        if evidence:
            ev = evidence if len(evidence) <= 300 else evidence[:300] + "..."
            tooltip_parts.append(f"<b>evidence</b>: {ev}")

        net.add_edge(
            src,
            dst,
            label=rel_type,
            title="<br>".join(tooltip_parts),
            arrows="to",
            smooth=False,
            dashes=(source == "property"),
        )

    net.set_options(
        """
        const options = {
          "nodes": {
            "font": { "size": 14, "multi": "html" }
          },
          "edges": {
            "font": { "size": 11, "align": "middle" },
            "color": { "color": "#888888", "highlight": "#333333" }
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
        if r.get("from_uid") in kept_nodes and r.get("to_uid") in kept_nodes
    ]
    return kept_nodes, kept_rels


# =========================================================
# UI
# =========================================================

st.title("Ontology-Driven KG Viewer")

st.markdown(
    """
Upload:
1. an ontology JSON
2. a KG JSON

Workflow:
- validate KG completeness against ontology
- output ontology reference graph summary
- output missing reference / invalid relationship reports if validation fails
- only visualize after validation passes
- infer property-based relationships from ontology-defined id references
"""
)

col1, col2 = st.columns(2)
with col1:
    ontology_file = st.file_uploader("Upload ontology JSON", type=["json"], key="ontology")
with col2:
    kg_file = st.file_uploader("Upload KG JSON", type=["json"], key="kg")

ontology_text = st.text_area("Or paste ontology JSON here", height=180)
kg_text = st.text_area("Or paste KG JSON here", height=180)

ontology_data = None
kg_data = None

try:
    if ontology_file is not None:
        ontology_data = json.load(ontology_file)
    elif ontology_text.strip():
        ontology_data = json.loads(ontology_text)

    if kg_file is not None:
        kg_data = json.load(kg_file)
    elif kg_text.strip():
        kg_data = json.loads(kg_text)
except Exception as e:
    st.error(f"JSON loading error: {e}")
    st.stop()

if ontology_data and kg_data:
    try:
        ontology_summary = summarize_ontology_reference_graph(ontology_data)
    except Exception as e:
        st.error(f"Ontology parsing error: {e}")
        st.stop()

    (
        kg_entities,
        entity_primary_id_fields,
        property_ref_targets,
        explicit_relationships,
        validation_report,
    ) = validate_kg_completeness(
        kg_data,
        ontology_data,
        raise_on_error=False
    )

    st.subheader("Ontology Reference Graph Summary")
    st.caption("Automatically inferred from ontology property types.")
    st.dataframe(ontology_summary["entity_summary"], use_container_width=True, height=220)
    st.dataframe(ontology_summary["reference_edges"], use_container_width=True, height=260)

    if not validation_report["passed"]:
        st.error("KG completeness validation failed.")

        if validation_report["missing_references"]:
            st.subheader("Missing Reference Report")
            st.dataframe(validation_report["missing_references"], use_container_width=True, height=260)

        if validation_report["invalid_relationships"]:
            st.subheader("Invalid Relationship Report")
            st.dataframe(validation_report["invalid_relationships"], use_container_width=True, height=220)

        with st.expander("All validation errors"):
            for err in validation_report["errors"]:
                st.write(f"- {err}")

        st.stop()

    st.success("KG completeness validation passed.")

    st.subheader("Validation Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Ontology entity types", len(ontology_summary["entity_summary"]))
    c2.metric("Ontology reference edges", len(ontology_summary["reference_edges"]))
    c3.metric("Validation errors", len(validation_report["errors"]))

    nodes = build_nodes_from_kg(kg_entities)
    explicit_with_uid = explicit_relationships_to_uids(explicit_relationships)
    property_relationships = infer_property_relationships(kg_entities, property_ref_targets)
    all_relationships = deduplicate_relationships(explicit_with_uid + property_relationships)

    all_types = sorted({n["type"] for n in nodes.values()})

    with st.sidebar:
        st.header("Filters")
        selected_types = st.multiselect("Entity types", all_types, default=all_types)
        keyword = st.text_input("Keyword")
        show_explicit = st.checkbox("Show explicit relationships", value=True)
        show_property = st.checkbox("Show property-inferred relationships", value=True)
        show_entity_table = st.checkbox("Show entity table", value=True)
        show_rel_table = st.checkbox("Show relationship table", value=True)

    relationships_to_show = []
    if show_explicit:
        relationships_to_show.extend([r for r in all_relationships if r.get("source") == "explicit"])
    if show_property:
        relationships_to_show.extend([r for r in all_relationships if r.get("source") == "property"])

    filtered_nodes, filtered_relationships = filter_graph(
        nodes, relationships_to_show, selected_types, keyword
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", len(filtered_nodes))
    c2.metric("Edges", len(filtered_relationships))
    c3.metric("Explicit edges", len([r for r in filtered_relationships if r.get("source") == "explicit"]))
    c4.metric("Property edges", len([r for r in filtered_relationships if r.get("source") == "property"]))

    net = build_network(filtered_nodes, filtered_relationships)
    html = net.generate_html()
    components.html(html, height=800, scrolling=True)

    if show_entity_table:
        st.subheader("Entities")
        entity_rows = []
        for uid, node in filtered_nodes.items():
            row = {
                "uid": uid,
                "type": node["type"],
                "raw_id": node["raw_id"],
            }
            row.update(node["data"])
            entity_rows.append(row)
        st.dataframe(entity_rows, use_container_width=True, height=320)

    if show_rel_table:
        st.subheader("Relationships")
        st.dataframe(filtered_relationships, use_container_width=True, height=300)