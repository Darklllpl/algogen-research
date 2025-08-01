import json
import os
from pathlib import Path
import copy
import subprocess

class SVLRenderer:
    """
    A generic renderer for parsing SVL 4.0 JSON data and rendering it as a TikZ .tex file.
    """

    def __init__(self, svl_data, output_dir="output_frames"):
        self._validate_version(svl_data.get("svl_version"))

        self.svl_data = svl_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.algorithm_info = svl_data["algorithm"]
        self.initial_frame = svl_data["initial_frame"]
        self.styles = self.initial_frame["styles"]
        self.pseudocode = self.initial_frame.get("pseudocode", [])
        self.variables_schema = self.initial_frame.get("variables_schema", [])

        self.elem_spacing = 2.0

        self.current_variables = {
            var["name"]: "-" for var in self.variables_schema
        }

        self.current_array_state = []
        for elem in self.initial_frame["data_state"]:
            self.current_array_state.append(
                {
                    "value": elem["value"],
                    "styleKey": elem["state"],
                    "tikz_node_name": f"node-{elem['index']}",
                    "position": (elem["index"] * self.elem_spacing, 0),
                }
            )

        self.temp_elements = []
        self.boundaries = []

    def _validate_version(self, version):
        if version != "5.0":
            print(f"Warning: This renderer is designed for SVL 5.0, but the input file version is {version}. Compatibility issues may occur.")

    def render(self):
        print("Starting to render SVL 5.0 visualization sequence...")
        self._render_and_save_frame(frame_index=0, delta_info=None)

        for i, delta in enumerate(self.svl_data["deltas"]):
            self._apply_delta(delta)
            self._render_and_save_frame(frame_index=i + 1, delta_info=delta)

        print(
            f"Rendering completed! {len(self.svl_data['deltas']) + 1} frames saved to {self.output_dir}"
        )

    def _apply_delta(self, delta):
        meta = delta.get("meta", {})
        for var_name, var_value in meta.items():
            if var_name in self.current_variables:
                self.current_variables[var_name] = var_value

        self.temp_elements.clear()
        
        operations = delta.get("operations", [])
        for op_group in operations:
            if isinstance(op_group, dict):
                self._process_operation(op_group)
            elif isinstance(op_group, list):
                for op in op_group:
                    self._process_operation(op)

    def _process_operation(self, op):
        name = op.get("op")
        params = op.get("params", {})

        op_map = {
            "updateStyle": self._apply_update_style,
            "drawTemp": self._apply_draw_temp,
            "removeTemp": self._apply_remove_temp,
            "moveElements": self._apply_move_elements,
            "updateValues": self._apply_update_values,
            "shiftElements": self._apply_shift_elements,
            "updateBoundary": self._apply_update_boundary,
            "removeBoundary": self._apply_remove_boundary,
        }
        
        handler = op_map.get(name)
        if handler:
            handler(params)
        else:
            print(f"Warning: Unknown operation type '{name}'")

    def _apply_update_style(self, params):
        indices = params.get("indices", [])
        style_key = params.get("styleKey")
        for i in indices:
            if 0 <= i < len(self.current_array_state):
                self.current_array_state[i]["styleKey"] = style_key

    def _apply_draw_temp(self, params):
        self.temp_elements.append(params)

    def _apply_remove_temp(self, params):
        t = params.get("type")
        self.temp_elements = [e for e in self.temp_elements if e.get("type") != t]

    def _apply_move_elements(self, params):
        pairs = params.get("pairs", [])
        if not pairs: return

        snapshot = copy.deepcopy(self.current_array_state)
        if isinstance(pairs, dict):
             pairs = [pairs]
        
        for p in pairs:
            src = p.get("fromIndex", p.get("from"))
            dst = p.get("toIndex", p.get("to"))
            if src is not None and dst is not None and (0 <= src < len(snapshot) and 0 <= dst < len(snapshot)):
                self.current_array_state[dst] = snapshot[src]
        
        for idx, elem in enumerate(self.current_array_state):
            elem["position"] = (idx * self.elem_spacing, 0)
    
    def _apply_shift_elements(self, params):
        shifts = params.get("shifts", [])
        if not shifts: return
        
        snapshot = copy.deepcopy(self.current_array_state)
        for s in shifts:
            src, dst = s.get("fromIndex"), s.get("toIndex")
            if (0 <= src < len(snapshot) and 0 <= dst < len(snapshot)):
                self.current_array_state[dst] = snapshot[src]

        for idx, elem in enumerate(self.current_array_state):
            elem["position"] = (idx * self.elem_spacing, 0)

    def _apply_update_values(self, params):
        for u in params.get("updates", []):
            idx, val = u.get("index"), u.get("value")
            if idx is not None and 0 <= idx < len(self.current_array_state):
                self.current_array_state[idx]["value"] = val
    
    def _apply_update_boundary(self, params):
        t = params.get("type")
        self.boundaries = [b for b in self.boundaries if b.get("type") != t]
        self.boundaries.append(params)

    def _apply_remove_boundary(self, params):
        t = params.get("type")
        self.boundaries = [b for b in self.boundaries if b.get("type") != t]

    def _render_and_save_frame(self, frame_index, delta_info=None):
        tex_code = self._generate_tex_for_frame(frame_index, delta_info)
        path = self.output_dir / f"frame_{frame_index:04d}.tex"
        path.write_text(tex_code, encoding="utf-8")

    def _generate_tex_for_frame(self, frame_index, delta_info=None):
        preamble = self._generate_tex_preamble()
        pic = []
        
        total_w = (len(self.current_array_state) - 1) * self.elem_spacing
        center_x = total_w / 2

        title = self._escape_latex(self.algorithm_info.get("name", "Visualization"))
        family = self._escape_latex(self.algorithm_info.get("family", ""))
        pic.append(f"\\node[font=\\large\\bfseries] at ({center_x}, 6) {{{title}}};")
        if family:
            pic.append(f"\\node[font=\\small] at ({center_x}, 5.3) {{{family}}};")
        
        var_labels = []
        for var_def in self.variables_schema:
            var_name = var_def['name']
            var_value = self.current_variables.get(var_name, '-')
            if var_value != '-':
                var_labels.append(f"\\texttt{{{var_name}}} = {var_value}")
        
        if var_labels:
            var_str = " \\;\\; ".join(var_labels)
            var_y = 4.5
            pic.append(f"\\node[font=\\normalsize, inner sep=2pt, rounded corners=2pt, fill=black!5] at ({center_x}, {var_y}) {{{var_str}}};")

        frame_label = self._build_frame_label(frame_index, delta_info)
        pic.append(f"\\node[font=\\bfseries] at ({center_x}, 3.5) {{{frame_label}}};")

        for i, elem in enumerate(self.current_array_state):
            actual_x, actual_y = i * self.elem_spacing, 0
            node_name = f"node-{i}"
            elem["tikz_node_name"] = node_name
            style = f"element_{elem['styleKey']}"
            pic.append(f"\\node[{style}] ({node_name}) at ({actual_x}, {actual_y}) {{{elem['value']}}};")
            pic.append(f"\\node[below=0.1cm of {node_name}] {{{i}}};")

        for b in self.boundaries:
            start_idx, end_idx = b.get("range", [0, 0])
            if 0 <= start_idx < len(self.current_array_state) and 0 <= end_idx < len(self.current_array_state):
                node_left = self.current_array_state[start_idx]["tikz_node_name"]
                node_right = self.current_array_state[end_idx]["tikz_node_name"]
                skey = b.get("styleKey", "default_boundary")
                label = self._escape_latex(b.get("label", ""))
                pic.append(
                    f"\\begin{{pgfonlayer}}{{background}}\n"
                    f"  \\node[temp_{skey}, fit={{({node_left}.north west) ({node_right}.south east)}}, inner sep=0.4cm] (box) {{}};\n"
                    f"\\end{{pgfonlayer}}\n"
                    f"\\node[anchor=north, font=\\small\\itshape] at (box.south) {{{label}}};"
                )

        for t in self.temp_elements:
            if t["type"] == "swap_arrow":
                a, b = t["indices"]
                node_a, node_b = self.current_array_state[a]["tikz_node_name"], self.current_array_state[b]["tikz_node_name"]
                skey = t["styleKey"]
                pic.append(f"\\draw[temp_{skey}] ({node_a}.north) to[bend left=45] ({node_b}.north);")
            elif t["type"] == "key_holder":
                val = self._escape_latex(str(t.get("value", "")))
                skey = t.get("styleKey")
                pic.append(f"\\node[temp_{skey}] at ({center_x}, 2) {{key: {val}}};")

        code_start_x, code_start_y = total_w + 3, 2.5
        if self.pseudocode:
            current_highlight = delta_info.get("code_highlight") if delta_info else self.initial_frame.get("code_highlight")
            pic.append(f"\\node[font=\\bfseries, anchor=west] at ({code_start_x}, {code_start_y}) {{Pseudocode:}};")
            for line_num, line_text in enumerate(self.pseudocode, 1):
                raw_spaces = len(line_text) - len(line_text.lstrip(" "))
                indent = f"\\hspace*{{{raw_spaces * 0.5}em}}" if raw_spaces else ""
                style = "code_highlight" if line_num == current_highlight else "code_line"
                y_pos = code_start_y - (line_num * 0.7)
                pic.append(f"\\node[{style}] at ({code_start_x}, {y_pos}) {{\\texttt{{{indent}{self._escape_latex(line_text.lstrip(' '))}}}}};")

        body = "\n".join(pic)
        return f"{preamble}\n\\begin{{document}}\n\\begin{{tikzpicture}}\n{body}\n\\end{{tikzpicture}}\n\\end{{document}}\n"

    def _generate_tex_preamble(self):
        header = (
            "\\documentclass[tikz, border=10pt]{standalone}\n"
            "\\usepackage{xcolor}\n"
            "\\usepackage{tikz}\n"
            "\\usetikzlibrary{arrows.meta, decorations.pathmorphing, "
            "decorations.pathreplacing, positioning, fit, backgrounds}\n"
        )
        color_map = {}  

        def map_color(raw):
            if isinstance(raw, str):
                if raw.startswith("rgba("):
                    import re
                    match = re.match(r'rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)', raw)
                    if match:
                        r, g, b, a = match.groups()
                        hex_color = f"{int(r):02X}{int(g):02X}{int(b):02X}"
                        if hex_color not in color_map:
                            color_map[hex_color] = f"c{len(color_map)+1}"
                        color_name = color_map[hex_color]
                        opacity = float(a)
                        if opacity < 1.0:
                            return f"{color_name}!{int(opacity*100)}"
                        return color_name
                elif raw.startswith("#") and len(raw) == 7:
                    hex_code = raw[1:]
                    if hex_code not in color_map:
                        color_map[hex_code] = f"c{len(color_map)+1}"
                    return color_map[hex_code]
            return raw

        tikz_styles = [
            "every node/.style={font=\\sffamily}",
            "code_line/.style={anchor=west}",
            "code_highlight/.style={code_line, fill=yellow!20, rounded corners=2pt, inner sep=2pt}",
        ]
        for key, st in self.styles.get("elementStyles", {}).items():
            fill, stroke, w = map_color(st.get("fill","white")), map_color(st.get("stroke","black")), st.get("strokeWidth",1)
            tikz_styles.append(f"element_{key}/.style={{draw={stroke}, fill={fill}, minimum size=1cm, line width={w}pt, rounded corners=2pt}}")
        
        for key, st in self.styles.get("tempStyles", {}).items():
            if st.get("shape") == "box" or key.endswith("_box") or key.endswith("_boundary"):
                fill = map_color(st.get("fill", "gray!10"))
                stroke = map_color(st.get("stroke", "black"))
                w = st.get("strokeWidth", 1)
                tikz_styles.append(f"temp_{key}/.style={{draw={stroke}, fill={fill}, line width={w}pt, rounded corners=3pt}}")
            else:
                col, w = map_color(st.get("color", "black")), st.get("strokeWidth", 1)
                tikz_styles.append(f"temp_{key}/.style={{->, >={{Stealth[length=3mm]}}, line width={w}pt, color={col}}}")

        color_defs = [f"\\definecolor{{{name}}}{{HTML}}{{{hexcode}}}" for hexcode, name in color_map.items()]
        tikzset_block = "\\tikzset{\n  " + ",\n  ".join(tikz_styles) + "\n}"
        return header + "\n" + "\n".join(color_defs) + "\n" + tikzset_block + "\n"

    def _escape_latex(self, text: str) -> str:
        if not isinstance(text, str): text = str(text)
        return (text.replace("\\", r"\textbackslash{}").replace("{", r"\{").replace("}", r"\}").replace("_", r"\_").replace("^", r"\^").replace("&", r"\&").replace("%", r"\%").replace("$", r"\$").replace("#", r"\#").replace("~", r"\textasciitilde{}"))

    def _build_frame_label(self, idx, delta):
        if idx == 0 or not delta: return f"Frame {idx}: Initial State"
        return f"Frame {idx}"

if __name__ == "__main__":
    json_path = "local_model_generated_output/bubble_sort_nearly_reversed_0023.json" 

    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        print("Please run the corresponding tracker script (e.g., quicksort_tracker.py) first.")
    else:
        print(f"Loading SVL 5.0 file: {json_path}...")
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        
        output_dir_name = Path("sort") / (Path(json_path).stem.replace("_svl_4.0", "") + "_frames_4.0")
        renderer = SVLRenderer(data, output_dir=output_dir_name)
        renderer.render()