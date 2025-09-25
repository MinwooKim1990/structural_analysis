"""Utilities for interpreting GNN-based structural analysis outputs with an LLM.

The module targets 2D building frames whose physics are approximated by a graph
neural network (GNN). It supplies three core capabilities for classroom use:

1. Provide a rich dummy payload representing a 2-bay, 4-story frame subjected to
   strong lateral wind and gravity loads.
2. Draw a consistent stick diagram (nodes + members) with wind direction and load
   arrows using NetworkX and Matplotlib.
3. Package derived engineering metrics and prompt templates so an OpenAI model
   (``gpt-4o-mini`` or ``gpt-5-mini``) can deliver a structured explanation.

Set the ``OPENAI_API_KEY`` environment variable before calling ``generate_report``.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  

# --- Helper utilities -----------------------------------------------------


def _vector_magnitude(components: Dict[str, float]) -> float:
    """Compute the Euclidean magnitude of a dictionary of vector components."""

    return math.sqrt(sum(value**2 for value in components.values()))


def _safe_dict(data: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    """Traverse nested dictionaries with defaults."""

    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key, {})
    return current if isinstance(current, dict) else {}


def _ensure_client(client: Optional[OpenAI]) -> OpenAI:
    """Return an OpenAI client, instantiating one when an API key is present."""

    if client is not None:
        return client

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; cannot contact the LLM.")

    return OpenAI()


# --- Data classes ---------------------------------------------------------


@dataclass
class DerivedNodeMetric:
    node_id: str
    label: str
    elevation_m: float
    displacement_mag_mm: float
    horizontal_reaction_kN: float
    vertical_reaction_kN: float
    utilization: float
    load_vector_kN: Dict[str, float]


@dataclass
class DerivedElementMetric:
    element_id: str
    label: str
    element_type: str
    stress_ratio: float
    axial_force_kN: Optional[float]
    shear_force_kN: Optional[float]
    moment_kNm: Optional[float]


# --- Core class -----------------------------------------------------------


class StructuralInsightGenerator:
    """Prepare visualizations and LLM-ready prompts from GNN structural outputs."""

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        client: Optional[OpenAI] = None,
        llm_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.llm_model = llm_model
        self._client = client
        self.llm_options = llm_options or self._default_llm_options(llm_model)

    # ------------------------------------------------------------------
    # Data processing

    def derive_structural_metrics(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract node/element summaries and global highlights."""

        node_metrics: List[DerivedNodeMetric] = []
        elevations: Dict[str, float] = {}

        for node in payload.get("nodes", []):
            disp = _safe_dict(node, "predictions", "displacement_mm")
            reactions = _safe_dict(node, "predictions", "reaction_kN")
            loads = node.get("loads", {})
            _, y, _ = node.get("position", [0.0, 0.0, 0.0])
            elevations[node.get("id", "")] = y
            metric = DerivedNodeMetric(
                node_id=node.get("id", ""),
                label=node.get("label", node.get("id", "")),
                elevation_m=y,
                displacement_mag_mm=_vector_magnitude(disp) if disp else 0.0,
                horizontal_reaction_kN=reactions.get("Fx", 0.0),
                vertical_reaction_kN=reactions.get("Fy", 0.0),
                utilization=_safe_dict(node, "predictions").get("utilization", 0.0),
                load_vector_kN=loads,
            )
            node_metrics.append(metric)

        element_metrics: List[DerivedElementMetric] = []
        for element in payload.get("elements", []):
            preds = _safe_dict(element, "predictions")
            element_metrics.append(
                DerivedElementMetric(
                    element_id=element.get("id", ""),
                    label=element.get("label", element.get("id", "")),
                    element_type=element.get("type", "unknown"),
                    stress_ratio=preds.get("stress_ratio", 0.0),
                    axial_force_kN=preds.get("axial_force_kN"),
                    shear_force_kN=preds.get("shear_force_kN"),
                    moment_kNm=preds.get("bending_moment_kNm"),
                )
            )

        most_deformed = max(node_metrics, key=lambda m: m.displacement_mag_mm, default=None)
        highest_utilized_member = max(element_metrics, key=lambda m: m.stress_ratio, default=None)

        load_paths = self._infer_load_paths(payload)
        story_drifts = self._derive_story_drifts(payload)
        roof_drift = self._roof_drift(payload)
        base_reactions = self._aggregate_support_reactions(payload)

        summary = {
            "max_node_displacement_mm": {
                "node": most_deformed.node_id if most_deformed else None,
                "label": most_deformed.label if most_deformed else None,
                "value": round(most_deformed.displacement_mag_mm, 3) if most_deformed else None,
            },
            "max_element_utilization": {
                "element": highest_utilized_member.element_id if highest_utilized_member else None,
                "label": highest_utilized_member.label if highest_utilized_member else None,
                "type": highest_utilized_member.element_type if highest_utilized_member else None,
                "stress_ratio": round(highest_utilized_member.stress_ratio, 3) if highest_utilized_member else None,
            },
            "global_predictions": payload.get("global_predictions", {}),
            "total_applied_loads": self._aggregate_loads(node_metrics),
            "base_reaction_totals_kN": base_reactions,
            "load_paths": load_paths,
            "story_drifts_mm": story_drifts,
            "roof_drift_mm": roof_drift,
            "max_story_drift_mm": max(story_drifts, key=lambda d: abs(d["drift_mm"])) if story_drifts else None,
        }

        # Ensure max story drift values are rounded for readability
        if summary["max_story_drift_mm"]:
            summary["max_story_drift_mm"] = {
                "top_elevation_m": summary["max_story_drift_mm"]["top_elevation_m"],
                "drift_mm": round(summary["max_story_drift_mm"]["drift_mm"], 3),
            }

        if summary["roof_drift_mm"] is not None:
            summary["roof_drift_mm"] = round(summary["roof_drift_mm"], 3)

        return {
            "nodes": [self._dataclass_to_dict(metric) for metric in node_metrics],
            "elements": [self._dataclass_to_dict(metric) for metric in element_metrics],
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Prompting

    def build_prompt(
        self,
        payload: Dict[str, Any],
        derived: Dict[str, Any],
        diagram_path: Optional[Path] = None,
    ) -> str:
        """Compose the engineering prompt fed to the LLM."""

        metadata = payload.get("metadata", {})
        diagram_text = (
            f"The generated diagram is saved at '{diagram_path}'. Reference the filename when you talk about visual cues."
            if diagram_path
            else "A diagram path was not supplied. Explain geometries descriptively."
        )

        required_sections = """
1. Structure Overview – concise geometry, wind direction, support conditions, and overall load summary.
2. Node-level Response Summary – highlight every node's displacement, support role, and applied load; explain the mechanical reasoning for notable values.
3. Load Path Explanation – describe how gravity and lateral loads travel through members to the foundations.
4. Visualization Notes – help students read the diagram; mention node IDs, colour/arrow conventions, and qualitative deflection cues.
5. Critical Weaknesses – identify overstressed members or serviceability concerns (story drift, brace buckling, base shear).
6. Strengthening Strategies – propose retrofit ideas and explain how they mitigate the identified weaknesses.
7. Additional Observations – note vibration, redundancy, inspection tips, or model limitations relevant to civil engineering practice.
"""

        prompt = f"""
You are an expert structural engineering instructor. Analyse the following GNN-based surrogate structural analysis output and explain it to graduate students.

Project metadata: {json.dumps(metadata, indent=2)}

Raw structural response data:
{json.dumps(payload.get('nodes', []), indent=2)}

Member predictions:
{json.dumps(payload.get('elements', []), indent=2)}

Derived engineering metrics:
{json.dumps(derived, indent=2)}

{diagram_text}

Produce an English explanation using the numbered sections below.
{required_sections}

Keep the tone instructional, cite numerical values with units, and connect results back to structural mechanics intuition.
"""

        return prompt.strip()

    def generate_report(
        self,
        payload: Dict[str, Any],
        diagram_path: Optional[Path] = None,
        client: Optional[OpenAI] = None,
        temperature: float = 0.2,
        llm_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Call the LLM API and return the engineered explanation."""

        derived = self.derive_structural_metrics(payload)
        prompt = self.build_prompt(payload, derived, diagram_path)

        llm_client = _ensure_client(client or self._client)
        response = llm_client.responses.create(
            **self._build_llm_request(prompt, temperature, llm_overrides)
        )

        return response.output_text

    # ------------------------------------------------------------------
    # Visualisation

    def visualize_structure(
        self,
        payload: Dict[str, Any],
        output_path: Path,
        show_displacements: bool = True,
        figsize: Tuple[int, int] = (7, 8),
    ) -> Path:
        """Generate a 2D stick-diagram PNG using the provided payload."""

        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        graph = nx.Graph()
        positions: Dict[str, Tuple[float, float]] = {}
        node_colors: List[str] = []
        loads: Dict[str, Dict[str, float]] = {}

        for node in payload.get("nodes", []):
            node_id = node.get("id")
            if node_id is None:
                continue
            x, y, *_ = node.get("position", [0.0, 0.0, 0.0])
            graph.add_node(node_id, label=node.get("label", node_id), supports=node.get("supports", []))
            positions[node_id] = (x, y)
            loads[node_id] = node.get("loads", {})
            node_colors.append(self._colour_for_node(node))

        for element in payload.get("elements", []):
            start, end = element.get("nodes", [None, None])
            if start in graph.nodes and end in graph.nodes:
                graph.add_edge(start, end, label=element.get("label", element.get("id", "")))

        plt.figure(figsize=figsize)
        nx.draw_networkx_edges(graph, positions, width=2.0, edge_color="#4A7FB5")
        nx.draw_networkx_nodes(
            graph,
            positions,
            node_color=node_colors,
            node_size=420,
            edgecolors="#1F1F1F",
        )
        nx.draw_networkx_labels(
            graph,
            positions,
            labels={n: f"{n}\n{graph.nodes[n]['label']}" for n in graph.nodes},
            font_size=7,
        )

        self._annotate_loads(positions, loads)
        self._annotate_wind_direction(payload, positions)

        if show_displacements:
            self._draw_displacement_arrows(payload, positions)

        plt.title("GNN Surrogate – 2-Bay 4-Story Frame")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=220)
        plt.close()

        return output_path

    # ------------------------------------------------------------------
    # Internals

    def _draw_displacement_arrows(self, payload: Dict[str, Any], positions: Dict[str, Tuple[float, float]]) -> None:
        """Overlay exaggerated displacement arrows for intuition."""

        scale = 0.01  # exaggerated scale for readability
        for node in payload.get("nodes", []):
            node_id = node.get("id")
            disp = _safe_dict(node, "predictions", "displacement_mm")
            if not node_id or not disp or node_id not in positions:
                continue
            x, y = positions[node_id]
            dx = disp.get("ux", 0.0) * scale
            dy = disp.get("uy", 0.0) * scale
            if dx == 0 and dy == 0:
                continue
            plt.arrow(
                x,
                y,
                dx,
                dy,
                head_width=0.15,
                head_length=0.25,
                fc="#E15759",
                ec="#E15759",
                length_includes_head=True,
                alpha=0.75,
            )

    def _annotate_loads(self, positions: Dict[str, Tuple[float, float]], loads: Dict[str, Dict[str, float]]) -> None:
        """Draw arrows for applied loads, distinguishing horizontal wind and gravity."""

        scale_fx = 0.03
        scale_fy = 0.012
        for node_id, loadvec in loads.items():
            if node_id not in positions:
                continue
            x, y = positions[node_id]

            fx = loadvec.get("Fx", 0.0)
            if abs(fx) > 1e-6:
                plt.arrow(
                    x,
                    y,
                    fx * scale_fx,
                    0,
                    head_width=0.15,
                    head_length=0.25,
                    fc="#2CA02C",
                    ec="#2CA02C",
                    length_includes_head=True,
                    alpha=0.7,
                )
                plt.text(
                    x + fx * scale_fx * 1.05,
                    y + 0.2,
                    f"{fx:.0f} kN",
                    fontsize=7,
                    color="#2CA02C",
                )

            fy = loadvec.get("Fy", 0.0)
            if abs(fy) > 1e-6:
                plt.arrow(
                    x,
                    y,
                    0,
                    fy * scale_fy,
                    head_width=0.15,
                    head_length=0.25,
                    fc="#7F7F7F",
                    ec="#7F7F7F",
                    length_includes_head=True,
                    alpha=0.6,
                )
                plt.text(
                    x - 0.5,
                    y + fy * scale_fy * 1.05,
                    f"{fy:.0f} kN",
                    fontsize=7,
                    color="#7F7F7F",
                )

    def _annotate_wind_direction(
        self,
        payload: Dict[str, Any],
        positions: Dict[str, Tuple[float, float]],
    ) -> None:
        """Draw a global wind arrow if the net horizontal load is non-zero."""

        total_fx = sum(node.get("loads", {}).get("Fx", 0.0) for node in payload.get("nodes", []))
        if abs(total_fx) < 1e-6:
            return

        xs = [pos[0] for pos in positions.values()]
        ys = [pos[1] for pos in positions.values()]
        if not xs or not ys:
            return

        x_start = min(xs) - (max(xs) - min(xs)) * 0.25
        y_level = max(ys) + 1.0
        direction = 1 if total_fx > 0 else -1
        length = (max(xs) - min(xs)) * 0.6 * direction

        plt.arrow(
            x_start,
            y_level,
            length,
            0,
            head_width=0.4,
            head_length=0.6,
            fc="#1f77b4",
            ec="#1f77b4",
            linewidth=2.0,
        )
        plt.text(
            x_start + length * 0.5,
            y_level + 0.6,
            "Typhoon wind →",
            fontsize=9,
            color="#1f77b4",
            ha="center",
        )

    def _colour_for_node(self, node: Dict[str, Any]) -> str:
        """Colour supports differently for quick recognition."""

        supports = node.get("supports", [])
        if "fixed" in supports:
            return "#1E88E5"
        if "pinned" in supports:
            return "#43A047"
        if "roller" in supports:
            return "#FB8C00"
        return "#8E24AA"

    def _dataclass_to_dict(self, metric: Any) -> Dict[str, Any]:
        return {field: getattr(metric, field) for field in metric.__dataclass_fields__}

    def _aggregate_loads(self, metrics: Iterable[DerivedNodeMetric]) -> Dict[str, float]:
        totals = {"Fx": 0.0, "Fy": 0.0, "Fz": 0.0}
        for metric in metrics:
            for component, value in metric.load_vector_kN.items():
                totals[component] = totals.get(component, 0.0) + value
        return {key: round(value, 3) for key, value in totals.items()}

    def _aggregate_support_reactions(self, payload: Dict[str, Any]) -> Dict[str, float]:
        totals = {"Fx": 0.0, "Fy": 0.0, "Fz": 0.0}
        for node in payload.get("nodes", []):
            if not node.get("supports"):
                continue
            reactions = _safe_dict(node, "predictions", "reaction_kN")
            for component, value in reactions.items():
                totals[component] = totals.get(component, 0.0) + value
        return {key: round(value, 3) for key, value in totals.items()}

    def _infer_load_paths(self, payload: Dict[str, Any]) -> List[List[str]]:
        """Basic load path estimation from loaded nodes to supports."""

        graph = nx.Graph()
        for node in payload.get("nodes", []):
            graph.add_node(node.get("id"), supports=node.get("supports", []))
        for element in payload.get("elements", []):
            start, end = element.get("nodes", [None, None])
            if start and end:
                graph.add_edge(start, end)

        load_nodes = [node for node in payload.get("nodes", []) if _vector_magnitude(node.get("loads", {})) > 0]
        support_nodes = [node.get("id") for node in payload.get("nodes", []) if node.get("supports")]

        paths: List[List[str]] = []
        for load_node in load_nodes:
            start_id = load_node.get("id")
            if start_id not in graph:
                continue
            best_path: Optional[List[str]] = None
            for support_id in support_nodes:
                try:
                    candidate = nx.shortest_path(graph, start_id, support_id)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                if best_path is None or len(candidate) < len(best_path):
                    best_path = candidate
            if best_path:
                paths.append(best_path)

        return paths

    def _derive_story_drifts(self, payload: Dict[str, Any]) -> List[Dict[str, float]]:
        """Compute average diaphragm drifts between successive levels."""

        level_displacements: Dict[float, List[float]] = {}
        for node in payload.get("nodes", []):
            _, elevation, _ = node.get("position", [0.0, 0.0, 0.0])
            disp = _safe_dict(node, "predictions", "displacement_mm")
            if not disp:
                continue
            level_displacements.setdefault(elevation, []).append(disp.get("ux", 0.0))

        if not level_displacements:
            return []

        story_drifts: List[Dict[str, float]] = []
        sorted_levels = sorted(level_displacements.items())
        prev_level, prev_avg = sorted_levels[0][0], sum(sorted_levels[0][1]) / len(sorted_levels[0][1])

        for elevation, ux_values in sorted_levels[1:]:
            avg_ux = sum(ux_values) / len(ux_values)
            drift = avg_ux - prev_avg
            story_drifts.append({"top_elevation_m": elevation, "drift_mm": round(drift, 3)})
            prev_level, prev_avg = elevation, avg_ux

        return story_drifts

    def _roof_drift(self, payload: Dict[str, Any]) -> Optional[float]:
        """Average roof diaphragm drift relative to ground."""

        level_displacements: Dict[float, List[float]] = {}
        for node in payload.get("nodes", []):
            _, elevation, _ = node.get("position", [0.0, 0.0, 0.0])
            disp = _safe_dict(node, "predictions", "displacement_mm")
            if not disp:
                continue
            level_displacements.setdefault(elevation, []).append(disp.get("ux", 0.0))

        if not level_displacements:
            return None

        sorted_levels = sorted(level_displacements.items())
        base_avg = sum(sorted_levels[0][1]) / len(sorted_levels[0][1])
        roof_avg = sum(sorted_levels[-1][1]) / len(sorted_levels[-1][1])
        return roof_avg - base_avg

    def _default_llm_options(self, model: str) -> Dict[str, Any]:
        """Return model-specific default request options."""

        if model == "gpt-5-mini":
            return {
                "reasoning": {"effort": "medium"},
                "max_output_tokens": 1500,
                "modalities": ["text"],
            }
        return {}

    def _build_llm_request(
        self,
        prompt: str,
        temperature: float,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        options = {**self.llm_options, **(overrides or {})}
        request = {
            "model": self.llm_model,
            "input": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        request.update(options)
        return request


# --- Demonstration --------------------------------------------------------


def demo(diagram_filename: str = "demo_structural_diagram.png", model: str = "gpt-4o-mini") -> None:
    """Example workflow using the sample payload."""

    json_filepath = "SAMPLE_GNN_OUTPUT.json"
    with open(json_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    generator = StructuralInsightGenerator(llm_model=model)
    derived = generator.derive_structural_metrics(data)
    
    diagram_path = Path(diagram_filename)
    generator.visualize_structure(data, diagram_path)

    print("Derived metrics:")
    print(json.dumps(derived, indent=2))

    try:
        report = generator.generate_report(data, diagram_path=diagram_path)
        print("\nLLM report:\n")
        print(report)
    except RuntimeError as err:
        print("\nSkipping LLM call –", err)


if __name__ == "__main__":
    demo()
