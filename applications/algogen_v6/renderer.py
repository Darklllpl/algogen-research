# renderer.py
import json
import os
from pathlib import Path
import copy
import math
import sys

class SVLRenderer:
    """
    Universal renderer for parsing SVL 5.0 JSON data and rendering to TikZ .tex files.
    Final version aiming to completely and accurately implement SVL 5.0 specification.
    """

    def __init__(self, svl_data, output_dir="output_frames", debug_layout=False):
        # 1. Initialize and version validation
        self._validate_version(svl_data.get("svl_version"))

        self.svl_data = svl_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.debug_layout = debug_layout  # Add debug option

        # --- 2. Load metadata and initial state ---
        self.algorithm_info = svl_data["algorithm"]
        self.initial_frame = svl_data["initial_frame"]
        self.styles = self.initial_frame["styles"]
        self.pseudocode = self.initial_frame.get("pseudocode", [])
        self.variables_schema = self.initial_frame.get("variables_schema", [])
        self.data_schema = self.initial_frame.get("data_schema", {})

        # --- 3. Initialize dynamic state variables ---
        self.current_data_state = copy.deepcopy(self.initial_frame["data_state"])
        self.data_type = self.current_data_state["type"]
        self.current_aux_views = copy.deepcopy(self.initial_frame.get("auxiliary_views", []))
        self.current_variables = {var["name"]: "-" for var in self.variables_schema}
        
        self.temp_elements = [] 
        self.dependencies = []

        self.elem_spacing = 1.8  # Array element spacing

    def _validate_version(self, version):
        if version != "5.0":
            print(f"Warning: Renderer designed for SVL 5.0, but file version is {version}.")

    def render(self):
        """Main rendering loop, generates all frames."""
        print(f"Starting SVL 5.0 ({self.algorithm_info.get('name', '')}) visualization sequence rendering...")
        self._render_and_save_frame(frame_index=0, delta_info=None)

        for i, delta in enumerate(self.svl_data["deltas"]):
            self._apply_delta(delta)
            self._render_and_save_frame(frame_index=i + 1, delta_info=delta)
        
        print(f"Rendering complete! {len(self.svl_data['deltas']) + 1} frames saved to {self.output_dir}")

    # =================================================================
    # Delta application logic
    # =================================================================
    def _apply_delta(self, delta):
        # Update global variables
        meta = delta.get("meta", {})
        for var_name, var_value in meta.items():
            if var_name in self.current_variables:
                self.current_variables[var_name] = var_value

        # Clear temporary elements from previous frame, but keep boundary boxes
        boundary_boxes = [e for e in self.temp_elements if e.get("type") == "boundary_box"]
        self.temp_elements.clear()
        # Keep all boundary boxes, not just one
        self.temp_elements.extend(boundary_boxes)
        self.dependencies.clear()
        
        # Process operations
        operations = delta.get("operations", [])
        for op_group in operations:
            op_list = op_group if isinstance(op_group, list) else [op_group]
            for op in op_list:
                if isinstance(op, dict) and op.get("op"):
                    self._process_operation(op)

    def _process_operation(self, op):
        """Operation dispatcher, maps operation names to specific implementation functions."""
        op_name = op.get("op")
        params = op.get("params", {})
        
        op_map = {
            "updateStyle": self._apply_update_style, "moveElements": self._apply_move_elements,
            "shiftElements": self._apply_shift_elements, "updateValues": self._apply_update_values,
            "updateNodeStyle": self._apply_update_node_style, "updateNodeProperties": self._apply_update_node_properties,
            "updateEdgeStyle": self._apply_update_edge_style, "updateEdgeProperties": self._apply_update_edge_properties,
            "updateTableCell": self._apply_update_table_cell, "highlightTableCell": self._apply_highlight_table_cell,
            "showDependency": self._apply_show_dependency, "drawTemp": self._apply_draw_temp,
            "removeTemp": self._apply_remove_temp, "updateBoundary": self._apply_update_boundary,
            "removeBoundary": self._apply_remove_boundary, "addAuxView": self._apply_add_aux_view,
            "removeAuxView": self._apply_remove_aux_view,
        }
        
        handler = op_map.get(op_name)
        if handler: handler(params)
        else: print(f"Warning: Unknown operation type '{op_name}'")

    # --- Atomic operation implementations ---
    def _apply_add_aux_view(self, params):
        view_data = params.get('view')
        if view_data: self.current_aux_views.append(view_data)

    def _apply_remove_aux_view(self, params):
        view_id_to_remove = params.get('view_id')
        self.current_aux_views = [v for v in self.current_aux_views if v.get('view_id') != view_id_to_remove]

    def _apply_update_boundary(self, params):
        boundary_element = {"type": "boundary_box", "original_type": params.get("type"), **params}
        # Ensure type field is not overwritten by params type
        boundary_element["type"] = "boundary_box"
        self.temp_elements.append(boundary_element)

    def _apply_remove_boundary(self, params):
        type_to_remove = params.get("type")
        self.temp_elements = [e for e in self.temp_elements if not (e.get("type") == "boundary_box" and e.get("original_type") == type_to_remove)]

    def _apply_update_style(self, params):
        if self.data_type == "array":
            for i in params.get("indices", []):
                if 0 <= i < len(self.current_data_state["data"]): self.current_data_state["data"][i]["state"] = params.get("styleKey")
    
    def _apply_move_elements(self, params):
        if self.data_type == "array":
            pairs = params.get("pairs", [])
            snapshot = copy.deepcopy(self.current_data_state["data"])
            for p in pairs:
                src, dst = p.get("fromIndex"), p.get("toIndex")
                if src is not None and dst is not None and 0 <= src < len(snapshot) and 0 <= dst < len(snapshot):
                    self.current_data_state["data"][dst] = snapshot[src]

    def _apply_shift_elements(self, params):
        if self.data_type == "array":
            shifts = params.get("shifts", [])
            snapshot = copy.deepcopy(self.current_data_state["data"])
            for s in shifts:
                src, dst = s.get("fromIndex"), s.get("toIndex")
                if src is not None and dst is not None and 0 <= src < len(snapshot) and 0 <= dst < len(snapshot):
                    self.current_data_state["data"][dst] = snapshot[src]

    def _apply_update_values(self, params):
         if self.data_type == "array":
            for u in params.get("updates", []):
                idx, val = u.get("index"), u.get("value")
                if idx is not None and 0 <= idx < len(self.current_data_state["data"]): self.current_data_state["data"][idx]["value"] = val
    
    def _apply_update_node_style(self, params):
        if self.data_type == "graph":
            for node in self.current_data_state["structure"]["nodes"]:
                if node["id"] in params.get("ids", []): node["styleKey"] = params.get("styleKey")

    def _apply_update_node_properties(self, params):
        if self.data_type == "graph":
            for update in params.get("updates", []):
                for node in self.current_data_state["structure"]["nodes"]:
                    if node["id"] == update.get("id"): 
                        if "properties" not in node: node["properties"] = {}
                        node["properties"].update(update.get("properties", {}))
    
    def _apply_update_edge_style(self, params):
        if self.data_type == "graph":
            for edge in self.current_data_state["structure"]["edges"]:
                for etu in params.get("edges", []):
                    # Support bidirectional matching for undirected edges
                    if (edge["from"] == etu.get("from") and edge["to"] == etu.get("to")) or \
                       (not edge.get("directed") and edge["from"] == etu.get("to") and edge["to"] == etu.get("from")):
                        edge["styleKey"] = params.get("styleKey")
    
    # [BUG FIX] Complete _apply_update_edge_properties logic
    def _apply_update_edge_properties(self, params):
        if self.data_type == "graph":
            for update in params.get("updates", []):
                for edge in self.current_data_state["structure"]["edges"]:
                    # Support bidirectional matching for undirected edges
                    if (edge["from"] == update.get("from") and edge["to"] == update.get("to")) or \
                       (not edge.get("directed") and edge["from"] == update.get("to") and edge["to"] == update.get("from")):
                        if "properties" not in edge: edge["properties"] = {}
                        edge["properties"].update(update.get("properties", {}))

    def _apply_update_table_cell(self, params):
        for view in self.current_aux_views:
            if view.get("view_id") == params.get("view_id") and view.get("type") == "table":
                for u in params.get("updates", []):
                    r, c, val = u.get("row"), u.get("col"), u.get("value")
                    if r is not None and c is not None and 0 <= r < len(view["data"]) and 0 <= c < len(view["data"][r]):
                        view["data"][r][c] = val
    
    def _apply_draw_temp(self, params): self.temp_elements.append(params)
    def _apply_remove_temp(self, params): self.temp_elements = [e for e in self.temp_elements if e.get("type") != params.get("type")]
    def _apply_highlight_table_cell(self, params): self.temp_elements.append({"type": "table_highlight", **params})
    def _apply_show_dependency(self, params): self.dependencies.append(params)

    # =================================================================
    # Rendering logic
    # =================================================================
    
    # [BUG FIX] 重新添加缺失的 _render_and_save_frame 方法
    def _render_and_save_frame(self, frame_index, delta_info):
        """为单个帧生成TeX代码并将其保存到文件。"""
        tex_code = self._generate_tex_for_frame(frame_index, delta_info)
        path = self.output_dir / f"frame_{frame_index:04d}.tex"
        path.write_text(tex_code, encoding="utf-8")

    def _generate_tex_for_frame(self, frame_index, delta_info=None):
        """生成单帧的 LaTeX 代码。

        该版本将所有位置参数统一收敛到 layout 字典中，便于后续整体调优。
        layout 说明：
            layout["top"]      → 顶部信息区（算法名、动作描述、变量）
            layout["main"]     → 主视图（视图本身与标题）
            layout["aux"]      → 辅助视图（统一 x，首视图 y，间距 spacing，标题相对 y）
            layout["pseudo"]   → 伪代码区
        """

        # -----------------------------------------------
        # 0. 生成前置内容（导言区）
        # -----------------------------------------------
        preamble = self._generate_tex_preamble()
        pic: list[str] = []  # 保存 tikz 片段

        # -----------------------------------------------
        # 1. 智能布局计算
        # -----------------------------------------------
        layout_info = self._calculate_adaptive_layout()
        
        # -----------------------------------------------
        # 2. 顶部信息区
        # -----------------------------------------------
        top = layout_info["top"]
        pic.append(
            f"\\node[font=\\huge\\bfseries] at (0, {top['title_y']}) {{{self._escape_latex(self.algorithm_info.get('name', ''))}}};"
        )

        action_desc = self._generate_action_description(delta_info)
        frame_prefix = f"Frame {frame_index}: " if frame_index > 0 else ""
        pic.append(
            f"\\node[font=\\small\\itshape, color=black!70] at (0, {top['subtitle_y']}) {{{self._escape_latex(frame_prefix + action_desc)}}};"
        )

        var_labels = [
            f"\\texttt{{{self._escape_latex(n)}}}={self._escape_latex(v)}"
            for n, v in self.current_variables.items()
            if v != "-"
        ]
        if var_labels:
            pic.append(
                f"\\node[font=\\normalsize,fill=black!5,rounded corners=2pt, inner sep=3pt] at (0, {top['var_y']}) {{{', '.join(var_labels)}}};"
            )

        # -----------------------------------------------
        # 3. 主视图（含标题）
        # -----------------------------------------------
        main = layout_info["main"]
        if self.data_type == "array":
            if self.current_data_state.get("title"):
                pic.append(
                    f"\\node[font=\\bfseries, anchor=south] at ({main['x']}, {main['y'] + main['title_rel_y']}) {{{self._escape_latex(self.current_data_state.get('title', 'Array'))}}};"
                )
            pic.extend(self._render_array_view(x_offset=main["x"], y_offset=main["y"]))
        elif self.data_type == "graph":
            if self.current_data_state.get("title"):
                pic.append(
                    f"\\node[font=\\bfseries, anchor=south] at ({main['x']}, {main['y'] + main['title_rel_y']}) {{{self._escape_latex(self.current_data_state.get('title', 'Graph'))}}};"
                )
            pic.extend(self._render_graph_view(x_offset=main["x"], y_offset=main["y"]))
        elif self.data_type == "table":
            # 表格主视图
            table_title = self.current_data_state.get("title")
            if table_title:
                pic.append(
                    f"\\node[font=\\bfseries, anchor=south] at ({main['x']}, {main['y'] + main['title_rel_y']}) {{{self._escape_latex(table_title)}}};"
                )
            data = self.current_data_state.get("data", [])
            view_id = self.current_data_state.get("view_id", "main_table")
            matrix_content = [" & ".join([str(self._escape_latex(c)) for c in row]) for row in data]
            matrix_lines = [row + " \\\\" for row in matrix_content]
            matrix_str = "\n".join(matrix_lines)
            matrix_style = "matrix of nodes, nodes={minimum size=0.8cm, anchor=center}, column sep=1pt, row sep=1pt"
            matrix_y_pos = main["y"] - 1.5  # 固定向下偏移
            pic.append(
                f"\\matrix ({view_id}) at ({main['x']}, {matrix_y_pos}) [{matrix_style}] {{\n{matrix_str}\n}};"
            )

        # -----------------------------------------------
        # 4. 辅助视图
        # -----------------------------------------------
        if self.current_aux_views:
            aux_cfg = layout_info["aux"]
            pic.extend(
                self._render_aux_views(
                    x_offset=aux_cfg["x"],
                    first_y_offset=aux_cfg["first_y"],
                    spacing=aux_cfg["spacing"],
                    title_relative_y=aux_cfg["title_rel_y"],
                )
            )

        # -----------------------------------------------
        # 5. 伪代码区
        # -----------------------------------------------
        if self.pseudocode:
            pseudo = layout_info["pseudo"]
            pic.extend(self._render_pseudocode(delta_info, x_offset=pseudo["x"], y_offset=pseudo["y"]))

        # -----------------------------------------------
        # 6. 汇总生成 tikzpicture（使用动态边界框）
        # -----------------------------------------------
        bounding_box = layout_info["bounding_box"]
        pic.insert(0, f"\\useasboundingbox ({bounding_box['left']},{bounding_box['bottom']}) rectangle ({bounding_box['right']},{bounding_box['top']});")
        body = (
            "\\begin{tikzpicture}[x=1cm, y=1cm, every node/.style={transform shape}]\n"
            + "\n".join(pic)
            + "\n\\end{tikzpicture}"
        )
        return f"{preamble}\n\\begin{{document}}\n{body}\n\\end{{document}}\n"

    def _calculate_adaptive_layout(self):
        """智能计算自适应布局参数"""
        
        # -----------------------------------------------
        # 1. 计算各组件尺寸
        # -----------------------------------------------
        # 主视图尺寸
        main_view_width = self._calculate_main_view_width()
        main_view_height = self._calculate_main_view_height()
        
        # 伪代码尺寸
        pseudo_width = self._calculate_pseudocode_width()
        pseudo_height = self._calculate_pseudocode_height()
        
        # 辅助视图尺寸
        aux_views_info = self._calculate_aux_views_info()
        
        # 顶部信息区高度
        top_height = 3.0  # 标题 + 副标题 + 变量
        
        # 调试信息
        if hasattr(self, 'debug_layout') and self.debug_layout:
            print(f"布局调试信息:")
            print(f"  主视图: {main_view_width:.1f}cm × {main_view_height:.1f}cm")
            print(f"  伪代码: {pseudo_width:.1f}cm × {pseudo_height:.1f}cm")
            print(f"  辅助视图: {aux_views_info['total_width']:.1f}cm × {aux_views_info['total_height']:.1f}cm")
        
        # -----------------------------------------------
        # 2. 智能水平布局计算
        # -----------------------------------------------
        # 计算左右两侧需要的空间
        left_space_needed = pseudo_width + 3  # 伪代码 + 边距
        right_space_needed = main_view_width + 3  # 主视图 + 边距
        
        # 如果有辅助视图，需要考虑它们的宽度
        if aux_views_info["total_width"] > 0:
            right_space_needed = max(right_space_needed, aux_views_info["total_width"] + 3)
        
        # 计算总宽度和中心偏移
        total_width = left_space_needed + right_space_needed + 6  # 增加额外间距
        center_offset = (right_space_needed - left_space_needed) / 2
        
        # 如果伪代码太宽，调整布局策略
        if pseudo_width > main_view_width * 1.5:
            # 伪代码过宽时，采用上下布局
            if hasattr(self, 'debug_layout') and self.debug_layout:
                print(f"  采用上下布局（伪代码过宽）")
            return self._calculate_vertical_layout(main_view_width, main_view_height, 
                                                 pseudo_width, pseudo_height, 
                                                 aux_views_info, top_height)
        
        if hasattr(self, 'debug_layout') and self.debug_layout:
            print(f"  采用左右布局")
            print(f"  总宽度: {total_width:.1f}cm, 中心偏移: {center_offset:.1f}cm")
        
        # -----------------------------------------------
        # 3. 计算垂直布局
        # -----------------------------------------------
        # 计算各部分的高度
        content_height = max(
            main_view_height,
            aux_views_info["total_height"] if aux_views_info["total_height"] > 0 else 0,
            pseudo_height
        )
        
        # 计算总高度
        total_height = top_height + content_height + 6  # 增加额外间距
        
        # -----------------------------------------------
        # 4. 确定各组件位置
        # -----------------------------------------------
        # 顶部信息区
        top_y = total_height / 2 - 1
        top = {
            "title_y": top_y + 1.5,
            "subtitle_y": top_y + 0.8,
            "var_y": top_y + 0.0,
        }
        
        # 主视图位置（右侧）
        main_x = center_offset + main_view_width / 2
        main_y = top_y - content_height / 2
        main = {
            "x": main_x,
            "y": main_y,
            "title_rel_y": 1.5,
        }
        
        # 伪代码位置（左侧）
        pseudo_x = -center_offset - pseudo_width / 2
        pseudo_y = top_y - content_height / 2
        pseudo = {
            "x": pseudo_x,
            "y": pseudo_y,
        }

        # --------- 强制主视图和伪代码之间最小间距 ---------
        min_gap = 3.0  # cm
        main_left = main["x"] - main_view_width / 2
        pseudo_right = pseudo["x"] + pseudo_width / 2
        actual_gap = main_left - pseudo_right
        if actual_gap < min_gap:
            shift = (min_gap - actual_gap) / 2
            main["x"] += shift
            pseudo["x"] -= shift
        # -----------------------------------------------
        
        # 辅助视图位置（右侧，与主视图对齐）
        aux = {
            "x": main_x,
            "first_y": top_y + content_height / 2 - 1,
            "spacing": 1.5,
            "title_rel_y": 0.5,
        }
        
        # -----------------------------------------------
        # 5. 计算动态边界框
        # -----------------------------------------------
        # 添加安全边距
        margin = 3.0  # 增加边距
        bounding_box = {
            "left": -total_width / 2 - margin,
            "right": total_width / 2 + margin,
            "top": total_height / 2 + margin,
            "bottom": -total_height / 2 - margin,
        }
        
        if hasattr(self, 'debug_layout') and self.debug_layout:
            print(f"  边界框: ({bounding_box['left']:.1f}, {bounding_box['bottom']:.1f}) to ({bounding_box['right']:.1f}, {bounding_box['top']:.1f})")
        
        layout_result = {
            "top": top,
            "main": main,
            "aux": aux,
            "pseudo": pseudo,
            "bounding_box": bounding_box,
        }
        
        # 验证并调整布局边界
        return self._validate_layout_bounds(layout_result)

    def _validate_layout_bounds(self, layout_info):
        """验证布局边界是否合理，如果超出范围则进行调整"""
        bounding_box = layout_info["bounding_box"]
        
        # 检查边界框是否过大
        max_reasonable_size = 30  # 最大合理尺寸
        width = bounding_box["right"] - bounding_box["left"]
        height = bounding_box["top"] - bounding_box["bottom"]
        
        if width > max_reasonable_size or height > max_reasonable_size:
            if hasattr(self, 'debug_layout') and self.debug_layout:
                print(f"  警告: 布局过大 ({width:.1f}cm × {height:.1f}cm)，进行调整")
            
            # 缩放布局
            scale_factor = min(max_reasonable_size / width, max_reasonable_size / height)
            
            # 调整所有位置
            for section in ["top", "main", "aux", "pseudo"]:
                if section in layout_info:
                    for key, value in layout_info[section].items():
                        if isinstance(value, (int, float)):
                            layout_info[section][key] = value * scale_factor
            
            # 调整边界框
            center_x = (bounding_box["left"] + bounding_box["right"]) / 2
            center_y = (bounding_box["bottom"] + bounding_box["top"]) / 2
            new_width = width * scale_factor
            new_height = height * scale_factor
            
            layout_info["bounding_box"] = {
                "left": center_x - new_width / 2,
                "right": center_x + new_width / 2,
                "top": center_y + new_height / 2,
                "bottom": center_y - new_height / 2,
            }
        
        return layout_info

    def _calculate_vertical_layout(self, main_width, main_height, pseudo_width, pseudo_height, aux_info, top_height):
        """当伪代码过宽时，采用上下布局"""
        
        # 计算总宽度（取最宽的部分）
        total_width = max(main_width, pseudo_width, aux_info["total_width"]) + 6
        
        # 计算总高度
        content_height = main_height + pseudo_height + aux_info["total_height"] + 4
        total_height = top_height + content_height + 6
        
        # 顶部信息区
        top_y = total_height / 2 - 1
        top = {
            "title_y": top_y + 1.5,
            "subtitle_y": top_y + 0.8,
            "var_y": top_y + 0.0,
        }
        
        # 主视图位置（居中）
        main_y = top_y - content_height / 2 + main_height / 2
        main = {
            "x": 0,
            "y": main_y,
            "title_rel_y": 1.5,
        }
        
        # 伪代码位置（主视图下方）
        pseudo_y = main_y - main_height / 2 - pseudo_height / 2 - 2
        pseudo = {
            "x": 0,
            "y": pseudo_y,
        }
        
        # 辅助视图位置（伪代码下方）
        aux_y = pseudo_y - pseudo_height / 2 - aux_info["total_height"] / 2 - 2
        aux = {
            "x": 0,
            "first_y": aux_y + aux_info["total_height"] / 2,
            "spacing": 1.5,
            "title_rel_y": 0.5,
        }
        
        # 边界框
        margin = 3.0
        bounding_box = {
            "left": -total_width / 2 - margin,
            "right": total_width / 2 + margin,
            "top": total_height / 2 + margin,
            "bottom": -total_height / 2 - margin,
        }
        
        layout_result = {
            "top": top,
            "main": main,
            "aux": aux,
            "pseudo": pseudo,
            "bounding_box": bounding_box,
        }
        
        # 验证并调整布局边界
        return self._validate_layout_bounds(layout_result)

    def _calculate_main_view_width(self):
        """计算主视图宽度"""
        if not self.current_data_state or not self.current_data_state.get("data"):
            return 8  # 默认宽度
            
        if self.data_type == "array":
            return len(self.current_data_state["data"]) * self.elem_spacing
        elif self.data_type == "table":
            rows = len(self.current_data_state["data"])
            cols = len(self.current_data_state["data"][0]) if rows > 0 else 0
            return max(cols * 1.2, 8)
        elif self.data_type == "graph" and self.current_data_state.get("structure"):
            nodes = self.current_data_state["structure"].get("nodes", [])
            return max(12, len(nodes) * 2.5)
        
        return 8

    def _calculate_main_view_height(self):
        """计算主视图高度"""
        if not self.current_data_state:
            return 4  # 默认高度
            
        if self.data_type == "array":
            return 3  # 数组视图高度
        elif self.data_type == "table":
            rows = len(self.current_data_state.get("data", []))
            return max(2, rows * 0.8 + 1)  # 表格高度
        elif self.data_type == "graph":
            return 8  # 图视图高度
        
        return 4

    def _calculate_pseudocode_width(self):
        """计算伪代码宽度"""
        if not self.pseudocode:
            return 0
        
        # 计算最长行的长度
        max_line_length = 0
        for line in self.pseudocode:
            # 更精确的字符宽度估算
            line_width = 0
            for char in line:
                if ord(char) > 127:  # 中文字符
                    line_width += 2.2  # 中文字符稍宽
                elif char in 'WwMm':  # 宽字符
                    line_width += 1.2
                elif char in 'ijl|':  # 窄字符
                    line_width += 0.6
                elif char in '0123456789':  # 数字
                    line_width += 0.8
                else:  # 普通字符
                    line_width += 1.0
            max_line_length = max(max_line_length, line_width)
        
        # 转换为厘米（假设每个字符0.25cm，更保守的估算）
        return max(8, max_line_length * 0.25)  # 最小宽度8cm

    def _calculate_pseudocode_height(self):
        """计算伪代码高度"""
        if not self.pseudocode:
            return 0
        
        # 每行0.8cm高度，标题1.2cm
        return len(self.pseudocode) * 0.8 + 1.2

    def _calculate_aux_views_info(self):
        """计算辅助视图信息"""
        if not self.current_aux_views:
            return {"total_width": 0, "total_height": 0}
        
        total_width = 0
        total_height = 0
        
        for view in self.current_aux_views:
            if view.get("type") == "table":
                data = view.get("data", [])
                if data:
                    cols = len(data[0])
                    rows = len(data)
                    view_width = max(6, cols * 1.2)
                    view_height = rows * 0.8 + 1.5  # 标题 + 内容
                else:
                    view_width = 6
                    view_height = 2
            elif view.get("type") == "list":
                data = view.get("data", {})
                if isinstance(data, list):
                    lines = len(data)
                else:
                    lines = sum(len(v) for v in data.values())
                view_width = 8
                view_height = max(1, lines * 0.6) + 1.5
            else:
                view_width = 6
                view_height = 2
            
            total_width = max(total_width, view_width)
            total_height += view_height + 1.5  # 间距
        
        return {
            "total_width": total_width,
            "total_height": total_height,
        }

    # --- 各视图的具体渲染实现 ---
    def _render_array_view(self, x_offset=0, y_offset=0):
        pic = []
        array_data = self.current_data_state.get("data", [])
        center_offset = (len(array_data) - 1) * self.elem_spacing / 2
        for i, elem in enumerate(array_data):
            pos_x = i * self.elem_spacing - center_offset
            style = f"element_{elem.get('state', 'idle')}"
            node_name = f"node-{i}"
            pic.append(f"\\node[{style}] ({node_name}) at ({pos_x + x_offset}, {y_offset}) {{{self._escape_latex(elem.get('value'))}}};")
            pic.append(f"\\node[below=2pt of {node_name}, font=\\tiny] {{{i}}};")
        
        for temp in self.temp_elements:
            if temp.get("type") == "boundary_box":
                start_idx, end_idx = temp.get("range", [0, 0])
                if start_idx <= end_idx < len(array_data):
                    style = f"box_{temp.get('styleKey', 'default_boundary')}"
                    label = self._escape_latex(temp.get('label', ''))
                    fit_nodes = f"(node-{start_idx}.north west) (node-{end_idx}.south east)"
                    pic.append(f"\\begin{{pgfonlayer}}{{background}} \\node[{style}, fit={{{fit_nodes}}}] (box) {{}}; \\end{{pgfonlayer}}")
                    if label: pic.append(f"\\node[above=2pt of box.north, font=\\small] {{{label}}};")
        return pic

    def _render_graph_view(self, x_offset=0, y_offset=0):
        pic, nodes = [], self.current_data_state.get("structure", {}).get("nodes", [])
        if not nodes: return pic
        num_nodes = len(nodes)
        radius = len(nodes) * 1.8 / math.pi if len(nodes) > 1 else 1
        for i, node in enumerate(nodes):
            angle = 360 / num_nodes * i
            pos_x, pos_y = radius * math.cos(math.radians(angle)), radius * math.sin(math.radians(angle))
            style, label = f"element_{node.get('styleKey', 'idle_node')}", self._build_node_label(node)
            pic.append(f"\\node[{style}] ({node['id']}) at ({pos_x + x_offset}, {pos_y + y_offset}) {{{label}}};")
        
        for edge in self.current_data_state.get("structure", {}).get("edges", []):
            style_opt = f"edge_{edge.get('styleKey', 'normal_edge')}"
            directed = edge.get("directed", False)
            opts_items = [style_opt]
            if directed:
                opts_items.append("->")
            opts = f"[{', '.join(opts_items)}]"

            label_txt = edge.get("label")
            label_node = ""
            if label_txt is not None and str(label_txt) != "-":
                label_node = f" node[midway, font=\\tiny, fill=white, auto=left, sloped, inner sep=1pt] {{{self._escape_latex(label_txt)}}} "

            pic.append(f"\\draw{opts} ({edge['from']}) to{label_node} ({edge['to']});")
        return pic

    def _render_aux_views(self, x_offset=0, first_y_offset=3, spacing=3, title_relative_y=1.5):
        """按顺序垂直堆叠所有辅助视图，自动避免相互重叠。

        参数说明：
            x_offset        所有辅助视图统一的水平偏移
            first_y_offset  第一块辅助视图（标题行下缘）的基准 Y 坐标
            spacing         不同辅助视图之间的额外间距，单位 cm
            title_relative_y 标题相对于视图内容的相对Y偏移（默认1.5cm上方）
        """
        pic = []

        current_y_offset = first_y_offset  # 当前视图主体（非标题）放置的 Y 坐标
        for view in self.current_aux_views:
            # 对于每一个视图，标题放在主体上方 title_relative_y 处
            title_y_offset = current_y_offset + title_relative_y

            if view.get("type") == "table":
                pic.extend(self._render_table_view(view, x_offset, current_y_offset, title_y_offset))
                # 表格高度：行数 * 0.8cm，再加 1cm 标题与 spacing
                rows = len(view.get("data", []))
                view_height = rows * 0.8 + 1  # 简化估算
            elif view.get("type") == "list":
                pic.extend(self._render_list_view(view, x_offset, current_y_offset, title_y_offset))
                # 列表高度：项目数量 * 0.6cm，再加 1cm 标题
                data = view.get("data", {})
                if isinstance(data, list):
                    lines = len(data)
                else:
                    lines = sum(len(v) for v in data.values())
                view_height = max(1, lines * 0.6) + 1
            else:
                # 未知类型，跳过，但仍下移固定间距
                view_height = 2

            # 更新下一视图的 y 坐标（当前视图底部再减 spacing）
            current_y_offset = current_y_offset - view_height - spacing

        return pic
        
    def _render_table_view(self, view, x_offset, y_offset, title_y_offset=None):
        pic, data, view_id = [], view.get("data", []), view.get("view_id")
        # 如果没有提供标题Y偏移，则使用视图Y偏移
        if title_y_offset is None:
            title_y_offset = y_offset
        pic.append(f"\\node[font=\\bfseries, anchor=south] at ({x_offset}, {title_y_offset}) {{{self._escape_latex(view.get('title',''))}}};")
        matrix_content = [" & ".join([str(self._escape_latex(c)) for c in row]) for row in data]
        # 为每行内容添加 LaTeX 换行符\\，确保最后一行末尾也有\\
        matrix_lines = [row + " \\\\" for row in matrix_content]
        matrix_str = "\n".join(matrix_lines)
        matrix_style = "matrix of nodes, nodes={minimum size=0.8cm, anchor=center}, column sep=1pt, row sep=1pt"
        # 使用固定的垂直间距，避免动态计算导致的相对定位问题
        matrix_y_pos = y_offset - 1.5  # 固定间距，可根据需要调整
        pic.append(f"\\matrix ({view_id}) at ({x_offset}, {matrix_y_pos}) [{matrix_style}] {{\n{matrix_str}\n}};")
        
        for temp in self.temp_elements:
            if temp.get("type") == "table_highlight" and temp.get("view_id") == view_id:
                for cell in temp.get("cells", []):
                    r, c = cell['row'] + 1, cell['col'] + 1
                    style = f"cell_{temp['styleKey']}"
                    pic.append(f"\\begin{{pgfonlayer}}{{background}}\n  \\node[{style}, fit=({view_id}-{r}-{c})] {{}};\n\\end{{pgfonlayer}}")
        
        for dep in self.dependencies:
            if dep.get("view_id") == view_id:
                to_r, to_c = dep["to_cell"]["row"] + 1, dep["to_cell"]["col"] + 1
                style = f"dep_{dep['styleKey']}"
                for from_cell in dep["from_cells"]:
                    from_r, from_c = from_cell["row"] + 1, from_cell["col"] + 1
                    pic.append(f"\\draw[->, {style}] ({view_id}-{from_r}-{from_c}.center) to ({view_id}-{to_r}-{to_c}.center);")
        return pic
        
    def _render_list_view(self, view, x_offset, y_offset, title_y_offset=None):
        pic = []
        # 如果没有提供标题Y偏移，则使用视图Y偏移
        if title_y_offset is None:
            title_y_offset = y_offset
        pic.append(f"\\node[font=\\bfseries, anchor=south] at ({x_offset}, {title_y_offset}) {{{self._escape_latex(view.get('title',''))}}};")
        y_pos = y_offset - 0.6
        data = view.get("data", {})
        if isinstance(data, list): # 支持简单列表
             data = {"List": data}
        for list_name, item_list in data.items():
            list_str = ", ".join(map(str, item_list))
            pic.append(f"\\node[anchor=west, font=\\small] at ({x_offset - 1}, {y_pos}) {{\\bfseries {list_name}:}};")
            pic.append(f"\\node[anchor=west, font=\\tt] at ({x_offset + 0.5}, {y_pos}) {{{self._escape_latex(list_str)}}};")
            y_pos -= 0.6
        return pic

    def _render_pseudocode(self, delta_info, x_offset=-10, y_offset=4):
        pic = []
        current_highlight = delta_info.get("code_highlight") if delta_info else self.initial_frame.get("code_highlight")
        # 添加背景框，使伪代码区域更加清晰
        pic.append(f"\\node[font=\\bfseries, anchor=west, fill=black!3, rounded corners, inner sep=5pt] at ({x_offset}, {y_offset+0.5}) {{Pseudocode:}};")
        for line_num, line_text in enumerate(self.pseudocode, 1):
            indent = len(line_text) - len(line_text.lstrip(" "))
            indent_cmd = f"\\hspace*{{{indent*0.5}em}}"
            escaped_text = self._escape_latex(line_text.lstrip(" "))
            style = "code_highlight" if line_num == current_highlight else "code_line"
            # 增大行间距
            y_pos = y_offset - (line_num * 0.8)
            pic.append(f"\\node[{style}, font=\\ttfamily] at ({x_offset}, {y_pos}) {{{indent_cmd}{escaped_text}}};")
        return pic
        
    def _build_node_label(self, node):
        """[增强] 正确渲染节点内部的每一个变量"""
        lines = [f"\\bfseries {self._escape_latex(node.get('label', node['id']))}"]
        node_props_schema = self.data_schema.get("node_properties_schema", [])
        for prop_def in node_props_schema:
            prop_name = prop_def["name"]
            prop_val = node.get("properties", {}).get(prop_name)
            if prop_val is not None and str(prop_val) != "-":
                lines.append(f"\\tiny {self._escape_latex(prop_name)}: {self._escape_latex(prop_val)}")
        return "\\\\".join(lines)

    def _escape_latex(self, text):
        """转义 LaTeX 特殊字符，避免编译错误"""
        mapping = {
            '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',  
            '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
            '\\': r'\textbackslash{}',
        }
        s = str(text)
        escaped_chars = []
        for ch in s:
            escaped_chars.append(mapping.get(ch, ch))
        return ''.join(escaped_chars)

    def _generate_action_description(self, delta):
        if not delta: return "Initial State"
        operations = delta.get("operations", [])
        if not operations: return "Process"
        
        flat_ops = [op for group in operations for op in (group if isinstance(group, list) else [group]) if isinstance(op, dict)]
        descriptions = []
        for op in flat_ops:
            op_name, params = op.get("op"), op.get("params", {})
            if op_name == "updateStyle" and params.get("styleKey") == "compare":
                descriptions.append(f"Compare at indices {params.get('indices')}")
            elif op_name == "moveElements":
                pairs = params.get("pairs", [])
                involved = sorted(list(set([p.get('fromIndex') for p in pairs] + [p.get('toIndex') for p in pairs])))
                if len(involved) == 2: descriptions.append(f"Swap indices {tuple(involved)}")
        
        return " | ".join(dict.fromkeys(descriptions)) or "Process"

    def _generate_tex_preamble(self):
        header = "\\documentclass[tikz, border=20pt]{standalone}\n\\usepackage{xcolor}\n\\usepackage{tikz}\n\\usetikzlibrary{arrows.meta, positioning, fit, backgrounds, shapes.misc, matrix}\n"
        color_map = {}
        def map_color(raw):
            # Handle null or invalid values
            if not raw:
                return "black"
            
            # Handle hex colors #RRGGBB
            if isinstance(raw, str) and raw.startswith("#"):
                hex_code = raw[1:].upper()
                if hex_code not in color_map:
                    color_map[hex_code] = f"svlcolor{len(color_map)}"
                return color_map[hex_code]
            
            # Handle rgba(r,g,b,a) format colors
            if isinstance(raw, str) and raw.startswith("rgba("):
                try:
                    # Extract rgba values
                    rgba = raw.replace("rgba(", "").replace(")", "").split(",")
                    r, g, b = int(rgba[0].strip()), int(rgba[1].strip()), int(rgba[2].strip())
                    opacity = float(rgba[3].strip())
                    
                    # Convert to hex and map
                    hex_code = f"{r:02X}{g:02X}{b:02X}"
                    if hex_code not in color_map:
                        color_map[hex_code] = f"svlcolor{len(color_map)}"
                    
                    # Return color with opacity
                    return f"{color_map[hex_code]}!{int(opacity*100)}"
                except (IndexError, ValueError):
                    print(f"Warning: Cannot parse RGBA color '{raw}', using black instead")
                    return "black"
            
            # Default return black
            return "black"

        styles_def = ["code_line/.style={anchor=west, font=\\ttfamily}", "code_highlight/.style={code_line, fill=yellow!20, rounded corners=2pt, inner sep=2pt}"]
        for key, st in self.styles.get("elementStyles", {}).items():
            fill, stroke, lw = map_color(st.get("fill","white")), map_color(st.get("stroke","black")), st.get("strokeWidth",1)
            if "node" in key:
                styles_def.append(f"element_{key}/.style={{circle, draw={stroke}, fill={fill}, minimum size=1.4cm, line width={lw}pt, align=center, text depth=0.5ex}}")
            elif "cell" in key:
                styles_def.append(f"cell_{key}/.style={{fill={fill}, rounded corners=1pt}}")
            else:
                styles_def.append(f"element_{key}/.style={{draw={stroke}, fill={fill}, minimum size=1cm, line width={lw}pt, rounded corners=2pt}}")
            # New: Generate highlight styles for table cells to avoid cell_<key> undefined error
            styles_def.append(f"cell_{key}/.style={{fill={fill}, draw={stroke}, rounded corners=1pt}}")
        
        # Generate corresponding edge styles edge_<key> for each element style
        for key, st in self.styles.get("elementStyles", {}).items():
            # Use stroke as edge color, strokeWidth as line width
            color, lw = map_color(st.get("stroke", "black")), st.get("strokeWidth", 1.5)
            styles_def.append(f"edge_{key}/.style={{line width={lw}pt, color={color}}}")
        
        for key, st in self.styles.get("tempStyles", {}).items():
             if "arrow" in key or "link" in key:
                color, lw = map_color(st.get("color", "red")), st.get("strokeWidth", 1.5)
                styles_def.append(f"dep_{key}/.style={{color={color}, line width={lw}pt, ->, >=Stealth}}")
             elif "box" in key or "boundary" in key:
                fill = map_color(st.get("fill", "none"))
                stroke = map_color(st.get("stroke", "black"))
                dash = f", dashed, dash pattern=on {st['strokeDash'].replace(' ', 'pt on ')}pt" if "strokeDash" in st else ""
                styles_def.append(f"box_{key}/.style={{draw={stroke}, fill={fill}{dash}, rounded corners=3pt, inner sep=8pt}}")

        color_defs = "\n".join([f"\\definecolor{{{name}}}{{HTML}}{{{hex_code}}}" for hex_code, name in color_map.items()])
        tikz_set = "\\tikzset{\n  " + ",\n  ".join(styles_def) + "\n}"
        return f"{header}\n{color_defs}\n{tikz_set}"

# --- Main program entry (adopting your command line interface design) ---
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SVL 5.0 Renderer")
    parser.add_argument("json_file", help="SVL JSON file path")
    parser.add_argument("--output-dir", "-o", help="Output directory", default=None)
    parser.add_argument("--debug-layout", "-d", action="store_true", help="Enable layout debug info")
    
    args = parser.parse_args()
    json_path = args.json_file

    if not os.path.exists(json_path):
        print(f"Error: File not found '{json_path}'")
        sys.exit(1)
    
    print(f"Loading file: {json_path}...")
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        output_dir = args.output_dir or (Path(json_path).parent / (Path(json_path).stem + "_frames"))
        renderer = SVLRenderer(data, output_dir=output_dir, debug_layout=args.debug_layout)
        renderer.render()
    except Exception as e:
        print(f"Unknown error during processing: {e}")
        raise e
