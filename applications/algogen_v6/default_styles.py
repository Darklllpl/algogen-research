# default_styles.py
#
# SVL 5.0 Standard Default Style Library
# All tracker scripts should import DEFAULT_STYLES object from here.
# This version has been refactored to improve semantic clarity and reduce redundancy.

DEFAULT_STYLES = {

  "elementStyles": {
    "idle":             {"fill": "#FFFFFF", "stroke": "#424242", "strokeWidth": 1.5},
    "compare":          {"fill": "#FFECB3", "stroke": "#FFB300", "strokeWidth": 2},
    "sorted":           {"fill": "#E3F2FD", "stroke": "#2196F3", "strokeWidth": 1.5},
    "swapping":         {"fill": "#FFCDD2", "stroke": "#D32F2F", "strokeWidth": 2.5}, 

    "pivot":            {"fill": "#E0F2F1", "stroke": "#00897B", "strokeWidth": 2.5}, 
    "key_element":      {"fill": "#D1F2EB", "stroke": "#009688", "strokeWidth": 2.5}, 
    "placeholder":      {"fill": "#F5F5F5", "stroke": "#BDBDBD"},                   
    "shifting":         {"fill": "#BBDEFB", "stroke": "#1976D2"},                   
    "sub_array_active": {"fill": "#F9FBE7", "stroke": "#AFB42B", "strokeWidth": 1.5}, 
    "partition_area":   {"fill": "#E8F5E9", "stroke": "#66BB6A", "strokeWidth": 1.5}, 
    "heap_area":        {"fill": "#F5F5F5", "stroke": "#616161"},                    
    
    "idle_node":        {"fill": "#FAFAFA", "stroke": "#616161", "strokeWidth": 2},
    "current_node":     {"fill": "#FFF9C4", "stroke": "#FBC02D", "strokeWidth": 3},
    "visited_node":     {"fill": "#E8EAF6", "stroke": "#3F51B5", "strokeWidth": 2},
    "in_path_node":     {"fill": "#C8E6C9", "stroke": "#4CAF50", "strokeWidth": 2.5},
    
    "normal_edge":      {"color": "#9E9E9E", "strokeWidth": 1.5},
    "traversed_edge":   {"color": "#7E57C2", "strokeWidth": 3},
    "in_path_edge":     {"color": "#66BB6A", "strokeWidth": 3.5},

    "current_cell":     {"fill": "rgba(255, 249, 196, 0.7)"}, 
    "dependency_cell":  {"fill": "rgba(225, 245, 254, 0.7)"}, 
    "updated_cell":     {"fill": "rgba(200, 230, 201, 0.7)"}  
  },

  "variableStyles": {
    "default_pointer": {"color": "#D32F2F", "shape": "arrow"},
    "default_value":   {"color": "#212121"}
  },

  "tempStyles": {
    "swap_arrow":         {"color": "#D32F2F", "strokeWidth": 2, "tip": "Stealth"},
    "dependency_arrow":   {"color": "#FF7043", "strokeWidth": 1.5, "tip": "Latex"},
    "partition_box":      {"fill": "rgba(179, 229, 252, 0.2)", "stroke": "#0288D1", "strokeDash": "4 2"},
    "key_holder_box":     {"fill": "#F3E5F5", "stroke": "#8E24AA"},
    "parent_child_link":  {"color": "#009688", "strokeWidth": 1.5, "tip": "none"},
    "heap_boundary":      {"fill": "rgba(200, 230, 201, 0.2)", "stroke": "#4CAF50", "strokeDash": "4 2"}
  },

  "commentStyles": {
    "default_comment": {"fill": "#EEEEEE", "fontColor": "#212121", "borderColor": "#BDBDBD"}
  },
  
  "animationStyles": {
      "default_move": {"type": "ease-in-out", "duration": 500}
  }
}