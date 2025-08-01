# renderer.py
import json
import os
from pathlib import Path
import copy
import math
import sys

class SVLRenderer:
    """
    一个通用的渲染器，用于解析 SVL 5.0 JSON 数据并渲染为 TikZ .tex 文件。
    此为最终版本，旨在完整、准确地实现 SVL 5.0 规范。
    """

    def __init__(self, svl_data, output_dir="output_frames"):
        # 1. 初始化和版本验证
        self._validate_version(svl_data.get("svl_version"))

        self.svl_data = svl_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # --- 2. 加载元数据和初始状态 ---
        self.algorithm_info = svl_data["algorithm"]
        self.initial_frame = svl_data["initial_frame"]
        self.styles = self.initial_frame["styles"]
        self.pseudocode = self.initial_frame.get("pseudocode", [])
        self.variables_schema = self.initial_frame.get("variables_schema", [])
        self.data_schema = self.initial_frame.get("data_schema", {})

        # --- 3. 初始化动态状态变量 ---
        self.current_data_state = copy.deepcopy(self.initial_frame["data_state"])
        self.data_type = self.current_data_state["type"]
        self.current_aux_views = copy.deepcopy(self.initial_frame.get("auxiliary_views", []))
        self.current_variables = {var["name"]: "-" for var in self.variables_schema}
        
        self.temp_elements = [] 
        self.dependencies = []

        self.elem_spacing = 1.8  # 数组元素的间距

    def _validate_version(self, version):
        if version != "5.0":
            print(f"警告: 渲染器专为 SVL 5.0 设计，但文件版本为 {version}。")

    def render(self):
        """主渲染循环，生成所有帧。"""
        print(f"开始渲染 SVL 5.0 ({self.algorithm_info.get('name', '')}) 可视化序列...")
        self._render_and_save_frame(frame_index=0, delta_info=None)

        for i, delta in enumerate(self.svl_data["deltas"]):
            self._apply_delta(delta)
            self._render_and_save_frame(frame_index=i + 1, delta_info=delta)
        
        print(f"渲染完成！{len(self.svl_data['deltas']) + 1} 帧已保存至 {self.output_dir}")

    # =================================================================
    # Delta 应用逻辑
    # =================================================================
    def _apply_delta(self, delta):
        # 更新全局变量
        meta = delta.get("meta", {})
        for var_name, var_value in meta.items():
            if var_name in self.current_variables:
                self.current_variables[var_name] = var_value

        # 清空上一帧的临时元素
        self.temp_elements.clear()
        self.dependencies.clear()
        
        # 处理操作
        operations = delta.get("operations", [])
        for op_group in operations:
            op_list = op_group if isinstance(op_group, list) else [op_group]
            for op in op_list:
                if isinstance(op, dict) and op.get("op"):
                    self._process_operation(op)

    def _process_operation(self, op):
        """操作分发器，将操作名映射到具体的实现函数。"""
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
        else: print(f"警告: 未知的操作类型 '{op_name}'")

    # --- 原子操作的具体实现 ---
    def _apply_add_aux_view(self, params):
        view_data = params.get('view')
        if view_data: self.current_aux_views.append(view_data)

    def _apply_remove_aux_view(self, params):
        view_id_to_remove = params.get('view_id')
        self.current_aux_views = [v for v in self.current_aux_views if v.get('view_id') != view_id_to_remove]

    def _apply_update_boundary(self, params):
        self.temp_elements.append({"type": "boundary_box", "original_type": params.get("type"), **params})

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
                    # 支持无向边的双向匹配
                    if (edge["from"] == etu.get("from") and edge["to"] == etu.get("to")) or \
                       (not edge.get("directed") and edge["from"] == etu.get("to") and edge["to"] == etu.get("from")):
                        edge["styleKey"] = params.get("styleKey")
    
    # [BUG FIX] 补完 _apply_update_edge_properties 的逻辑
    def _apply_update_edge_properties(self, params):
        if self.data_type == "graph":
            for update in params.get("updates", []):
                for edge in self.current_data_state["structure"]["edges"]:
                    # 支持无向边的双向匹配
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
    # 渲染逻辑
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
        # 1. 计算整体布局
        # -----------------------------------------------
        # 1.1 主视图宽度，用于推算左右留白
        main_view_width = 8 # 默认宽度
        if self.current_data_state and self.current_data_state.get("data"):
            if self.data_type == "array":
                main_view_width = len(self.current_data_state["data"]) * self.elem_spacing
            elif self.data_type == "table":
                rows = len(self.current_data_state["data"])
                cols = len(self.current_data_state["data"][0]) if rows > 0 else 0
                main_view_width = max(cols * 1.2, 8)
        elif self.data_type == "graph" and self.current_data_state and self.current_data_state.get("structure"):
            nodes = self.current_data_state["structure"].get("nodes", [])
            main_view_width = max(12, len(nodes) * 2.5)

        # 1.2 统一的左右偏移。视图区域整体以 (0,0) 为中心展开
        x_half_extent = max(8, main_view_width / 2 + 4)

        # 1.3 汇总所有布局参数
        layout = {
            "top": {
                "title_y": 8.5,
                "subtitle_y": 7.8,
                "var_y": 7.0,
            },
            "main": {
                "x": x_half_extent-5,
                "y": 0,
                "title_rel_y": 1.5,  # 标题相对主视图内容上方偏移
            },
            "aux": {
                "x": x_half_extent - 2,
                "first_y": 5,  # 第一块辅助视图主体 y
                "spacing": 1,  # 不同辅助视图间距
                "title_rel_y": 1.5,
            },
            "pseudo": {
                "x": -x_half_extent-1,
                "y": 3,
            },
        }

        # -----------------------------------------------
        # 2. 顶部信息区
        # -----------------------------------------------
        top = layout["top"]
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
        main = layout["main"]
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
            aux_cfg = layout["aux"]
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
            pseudo = layout["pseudo"]
            pic.extend(self._render_pseudocode(delta_info, x_offset=pseudo["x"], y_offset=pseudo["y"]))

        # -----------------------------------------------
        # 6. 汇总生成 tikzpicture
        # -----------------------------------------------
        pic.insert(0, "\\useasboundingbox (-12,-12) rectangle (12,12);")
        body = (
            "\\begin{tikzpicture}[x=1cm, y=1cm, every node/.style={transform shape}]\n"
            + "\n".join(pic)
            + "\n\\end{tikzpicture}"
        )
        return f"{preamble}\n\\begin{{document}}\n{body}\n\\end{{document}}\n"

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
            # 处理空值或无效值
            if not raw:
                return "black"
            
            # 处理十六进制颜色 #RRGGBB
            if isinstance(raw, str) and raw.startswith("#"):
                hex_code = raw[1:].upper()
                if hex_code not in color_map:
                    color_map[hex_code] = f"svlcolor{len(color_map)}"
                return color_map[hex_code]
            
            # 处理 rgba(r,g,b,a) 格式的颜色
            if isinstance(raw, str) and raw.startswith("rgba("):
                try:
                    # 提取 rgba 值
                    rgba = raw.replace("rgba(", "").replace(")", "").split(",")
                    r, g, b = int(rgba[0].strip()), int(rgba[1].strip()), int(rgba[2].strip())
                    opacity = float(rgba[3].strip())
                    
                    # 转换为十六进制并映射
                    hex_code = f"{r:02X}{g:02X}{b:02X}"
                    if hex_code not in color_map:
                        color_map[hex_code] = f"svlcolor{len(color_map)}"
                    
                    # 返回带透明度的颜色
                    return f"{color_map[hex_code]}!{int(opacity*100)}"
                except (IndexError, ValueError):
                    print(f"警告: 无法解析 RGBA 颜色 '{raw}'，使用黑色代替")
                    return "black"
            
            # 默认返回黑色
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
            # 新增：为表格单元格生成高亮样式，避免 cell_<key> 未定义报错
            styles_def.append(f"cell_{key}/.style={{fill={fill}, draw={stroke}, rounded corners=1pt}}")
        
        # 为每个元素样式生成对应的边样式 edge_<key>
        for key, st in self.styles.get("elementStyles", {}).items():
            # 使用 stroke 作为边的颜色，strokeWidth 作为线宽
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

# --- 主程序入口 (采纳您的命令行接口设计) ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python renderer.py <json_file_path>")
        sys.exit(1)

    json_path = sys.argv[1]

    if not os.path.exists(json_path):
        print(f"错误: 找不到文件 '{json_path}'")
        sys.exit(1)
    
    print(f"正在加载文件: {json_path}...")
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        output_dir = Path(json_path).parent / (Path(json_path).stem + "_frames")
        renderer = SVLRenderer(data, output_dir=output_dir)
        renderer.render()
    except Exception as e:
        print(f"处理时发生未知错误: {e}")
        raise e
