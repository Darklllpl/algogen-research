
import streamlit as st
import time
import subprocess
import os
import re
import json
import glob
from pathlib import Path
import traceback


def display_animation_player():
    """
    A standalone function for displaying and controlling the animation player.
    It is now the core of the image playback page and optimized to prevent flickering.
    This function is now fully controlled by control_object and deltas in session_state.
    """
    if not st.session_state.get("png_files"):
        st.warning("No image files available for playback.")
        return

    png_files = st.session_state.png_files
    control = st.session_state.get("control_object", {})
    deltas = st.session_state.get("deltas", [])
    current_frame = st.session_state.current_frame

    # --- UI controls ---
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button('▶️ 播放/暂停', key='play_pause_button'):
            st.session_state.playing = not st.session_state.get('playing', False)

    with col2:
        frame_selector = st.slider(
            '选择帧', 0, len(png_files) - 1, current_frame, key='frame_slider'
        )
        if frame_selector != current_frame:
            st.session_state.current_frame = frame_selector
            st.session_state.playing = False
    
    def match_condition(delta_operations, condition):
        """
        Check if the given delta operation list satisfies the condition.
        Now supports matching 'op' or 'styleKey'.
        - delta_operations: Expects a list of operations (e.g., from deltas[i]['operations']).
        - condition: Expects a dictionary (e.g., {"op": "moveElements"} or {"styleKey": "compare"}).
        """
        if not isinstance(delta_operations, list) or not isinstance(condition, dict) or not condition:
            return False

        try:
            key_to_match = list(condition.keys())[0]
            value_to_match = condition[key_to_match]
        except IndexError:
            return False 

        stack = list(delta_operations)
        
        while stack:
            item = stack.pop()
            
            if isinstance(item, dict):
                if key_to_match == "op":
                    if item.get("op") == value_to_match:
                        return True
                
                elif key_to_match == "styleKey":
                    if item.get("params", {}).get("styleKey") == value_to_match:
                        return True
            
            elif isinstance(item, list):
                stack.extend(reversed(item))
                
        return False

    top_annotation = st.empty()
    image_placeholder = st.empty()
    bottom_annotation = st.empty()

    annotation_texts = {'top': [], 'bottom': []}
    if current_frame > 0 and current_frame <= len(deltas):
        current_delta_dict = deltas[current_frame - 1]
        operations_list = current_delta_dict.get('operations', [])
        for ann in control.get('annotations', []):
            if match_condition(operations_list, ann.get('condition', {})):
                pos = ann.get('position', 'top')
                if pos in ['top', 'bottom']:
                    annotation_texts[pos].append(ann['text'])

    box_style = "text-align: center; font-size: 1.1em; font-weight: bold; color: #333; min-height: 2.5em; padding: 0.5em; border: 1px solid #ddd; border-radius: 8px; margin-top: 1em; margin-bottom: 1em;"
    
    top_text = '<br>'.join(annotation_texts['top'])
    bottom_text = '<br>'.join(annotation_texts['bottom'])

    top_annotation.markdown(f"<div style='{box_style}'>{top_text if top_text else '&nbsp;'}</div>", unsafe_allow_html=True)
    
    try:
        current_image_path = png_files[current_frame]
        image_placeholder.image(
            current_image_path, 
            caption=f'Frame: {Path(current_image_path).stem}'
        )
    except IndexError:
        st.error("Error: Unable to access the specified frame. Please reload the page.")
    except Exception as e:
        st.error(f'Error loading image: {e}')

    bottom_annotation.markdown(f"<div style='{box_style}'>{bottom_text if bottom_text else '&nbsp;'}</div>", unsafe_allow_html=True)


    if st.session_state.get('playing', False):
        base_delay = 0.5 
        speed_multiplier = control.get('speed_multiplier', 1.0)
        delay = base_delay / max(speed_multiplier, 0.1)  

        if current_frame > 0 and current_frame <= len(deltas):
            current_delta_dict = deltas[current_frame - 1]
            operations_list = current_delta_dict.get('operations', [])
            for pause in control.get('pauses', []):
                if match_condition(operations_list, pause.get('condition', {})):
                    delay += pause.get('duration', 0) / 1000.0
        
        time.sleep(delay)

        next_frame = (current_frame + 1) % len(png_files)
        st.session_state.current_frame = next_frame
        
        if next_frame == 0:
            st.session_state.playing = False

        st.rerun()

def extract_last_json(text):
    """
    Try to extract the last valid JSON object from Markdown code blocks, plain text, or single lines.
    """
    import json
    import re

    md_json_blocks = re.findall(r"```json\s*(\{[\s\S]*?\})\s*```", text)
    for block in reversed(md_json_blocks):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue

    fallback_blocks = list(re.finditer(r'({[\s\S]*?})|(\[[\s\S]*?\])', text))
    for match in reversed(fallback_blocks):
        candidate = match.group().replace('\u00A0', ' ')
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    try:
        return json.loads(text)
    except Exception:
        return None


# ---------------------------------
# 2. Streamlit application main interface
# ---------------------------------

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISPATCH_OUTPUT_DIR = os.path.join(BASE_DIR, "dispatch_output")
CONTROL_OBJECT_FILE = os.path.join(DISPATCH_OUTPUT_DIR, "control_object.json")

# --- Top mode selection ---
mode = st.selectbox("Please select mode:", ["Main flow", "Image playback"])

# --- Mode one: Image playback ---
if mode == "Image playback":
    st.title("Image frame animation player")

    # --- FIX: Add CSS to prevent flickering and scrolling ---
    st.markdown("""
    <style>
        .stApp {
            min-height: 100vh;
        }
        .block-container {
            min-height: 95vh;
        }
    </style>
    """, unsafe_allow_html=True)
    # --- END FIX ---

    # --- NEW: Add anchor and JS to keep scroll position at player ---
    st.markdown('<div id="player-anchor"></div>', unsafe_allow_html=True)
    st.markdown("""
        <script>
        window.onload = function() {
          const anchor = document.getElementById("player-anchor");
          if (anchor) {
            anchor.scrollIntoView({behavior: "auto", block: "start"});
          }
        }
        </script>
    """, unsafe_allow_html=True)
    # --- END NEW ---

    # --- NEW: Load data from files instead of relying on session state from other mode ---
    control_object = {}
    deltas = []
    
    try:
        with open(CONTROL_OBJECT_FILE, 'r', encoding='utf-8') as f:
            control_object = json.load(f)
    except FileNotFoundError:
        st.warning(f"Cannot find control file '{CONTROL_OBJECT_FILE}'. Please run 'Main flow' first.")
    except json.JSONDecodeError:
        st.error(f"Error parsing control file '{CONTROL_OBJECT_FILE}'. File may be corrupted.")

    # Find the latest svl_5.0.json file
    try:
        svl_files = glob.glob(os.path.join(DISPATCH_OUTPUT_DIR, "*_svl_5.0.json"))
        if not svl_files:
            raise FileNotFoundError
        latest_svl_file = max(svl_files, key=os.path.getctime)
        with open(latest_svl_file, 'r', encoding='utf-8') as f:
            svl_data = json.load(f)
            deltas = svl_data.get("deltas", [])
    except FileNotFoundError:
        st.warning(f"No SVL files found in '{DISPATCH_OUTPUT_DIR}'. Please run 'Main flow' first.")
    except json.JSONDecodeError:
        st.error(f"Error parsing SVL file '{latest_svl_file}'. File may be corrupted.")
    
    # Save loaded data to session_state
    st.session_state.control_object = control_object
    st.session_state.deltas = deltas

    if "current_frame" not in st.session_state:
        st.session_state.current_frame = 0
    if "playing" not in st.session_state:
        st.session_state.playing = False

    st.caption("This mode is used to directly load and play the image sequence in the 'image' folder.")
    
    image_dir = os.path.join(BASE_DIR, "image")
    png_files = sorted(glob.glob(os.path.join(image_dir, "frame_*.png")))

    if png_files:
        if 'png_files' not in st.session_state or st.session_state.png_files != png_files:
            st.session_state.png_files = png_files
            st.session_state.current_frame = 0
            st.session_state.playing = False
        
        display_animation_player()
    else:
        st.warning("No 'frame_*.png' image files found in the 'image' directory! Please run 'Main flow' first.")

# --- Mode two: Main flow ---
else:
    st.set_page_config(page_title="SVL 5.0 Visual Generator", page_icon="🔮")
    st.title("Algorithm Visual Generator (SVL 5.0)")
    st.caption("A smart application driven by SVL 5.0 and large language models")

    with st.expander("💡 如何使用？", expanded=False):
        st.markdown("""
        1.  Input the algorithm and data you want to visualize in the **Core instruction area**.
        2.  (Optional) Input the visual style you want in the **Style area**.
        3.  (Optional) Add breakpoints or annotations in the **Control area**.
        4.  Click the **Generate visualization** button!
        5.  After successful generation, switch to the **Image playback** mode to view the animation.
        """)
    st.markdown("---")

    core_prompt = st.text_area("Core instruction area (Algorithm & Data)", height=150, placeholder="Input algorithm and data here...")
    style_prompt = st.text_area("Style & presentation area (Style & Presentation)", height=100, placeholder="(Optional) Input the style you want here...")
    control_prompt = st.text_area("Control & annotation area (Control & Annotation)", height=100, placeholder="(Optional) Add breakpoints or annotations here...")

    PROMPT_FILE = os.path.join(BASE_DIR, "prompt", "llm_intent_recognition_prompt.txt")
    RESULT_FILE = os.path.join(BASE_DIR, "llm_result", "llm_intent_recognition_result.txt")
    FULL_PROMPT_FILE = os.path.join(BASE_DIR, "prompt", "llm_intent_recognition_prompt_full.txt")
    CREATIVE_PROMPT_FILE = os.path.join(BASE_DIR, "prompt", "llm_creative_prompt.txt")
    CREATIVE_PROMPT_FULL_FILE = os.path.join(BASE_DIR, "prompt", "llm_creative_prompt_full.txt")
    CREATIVE_RESULT_FILE = os.path.join(BASE_DIR, "llm_result", "llm_creative_result.txt")

    if st.button("生成可视化", type="primary"):
        if not core_prompt:
            st.warning("请输入核心指令（算法和数据）！")
        else:
            status_placeholder = st.empty()
            intent_json = None
            creative_json = None
            sample_intent = None
            process_ok = True
            dispatch_output_file = ""
            with st.spinner("执行中，请稍候..."):
                # --- Step 1: Call llm_intent_recognition ---
                status_placeholder.info("Step 1: Start intent recognition...")
                try:
                    with open(PROMPT_FILE, "r", encoding="utf-8-sig") as f:
                        base_prompt = f.read()
                    user_prompt = f"Core instruction area:\n{core_prompt}\nStyle & presentation area:\n{style_prompt}\nControl & annotation area:\n{control_prompt}\n"
                    full_prompt = base_prompt.strip() + "\n\n" + user_prompt
                    with open(FULL_PROMPT_FILE, "w", encoding="utf-8") as f:
                        f.write(full_prompt)
                    subprocess.run(["python3", os.path.join(BASE_DIR, "llm_intent_recognition.py"), FULL_PROMPT_FILE], check=True, cwd=BASE_DIR)
                    
                    with open(RESULT_FILE, "r", encoding="utf-8-sig") as f:
                        llm_result = f.read()
                    intent_json = extract_last_json(llm_result)
                    if intent_json:
                        status_placeholder.success("Step 1: Intent recognition completed, JSON extraction successful")
                    else:
                        status_placeholder.error("Step 1 failed: Unable to extract intent JSON from the result")
                        process_ok = False
                except Exception as e:
                    status_placeholder.error(f"Step 1 error: {str(e)}\n{traceback.format_exc()}")  # Improved error
                    process_ok = False

                # --- Step 2: Call llm_creative ---
                if process_ok:
                    status_placeholder.info("Step 2: Start creative generation...")
                    try:
                        style_prompt_from_intent = intent_json.get("style_prompt", "")
                        control_prompt_from_intent = intent_json.get("control_prompt", "")
                        with open(CREATIVE_PROMPT_FILE, "r", encoding="utf-8-sig") as f:
                            creative_base = f.read()
                        # 在模板末尾添加用户指令
                        creative_user = f'\n\nStyle instruction:\n"{style_prompt_from_intent}"\n\nControl instruction:\n"{control_prompt_from_intent}"\n'
                        creative_full = creative_base.strip() + creative_user
                        with open(CREATIVE_PROMPT_FULL_FILE, "w", encoding="utf-8") as f:
                            f.write(creative_full)
                        subprocess.run(["python3", os.path.join(BASE_DIR, "llm_creative.py"), CREATIVE_PROMPT_FULL_FILE], check=True, cwd=BASE_DIR)
                        
                        with open(CREATIVE_RESULT_FILE, "r", encoding="utf-8-sig") as f:
                            creative_result = f.read()
                        creative_json = extract_last_json(creative_result)
                        if creative_json:
                            # 空的创意结果是可以接受的，因为用户可能没有提供样式或控制指令
                            status_placeholder.success("Step 2: Creative generation completed, JSON extraction successful")
                        else:
                            status_placeholder.error("Step 2 failed: Unable to extract creative JSON from the result")
                            process_ok = False
                    except Exception as e:
                        status_placeholder.error(f"Step 2 error: {str(e)}\n{traceback.format_exc()}")  # Improved error
                        process_ok = False

                # --- Step 3: Merge intent and creative ---
                if process_ok:
                    status_placeholder.info("Step 3: Start merging intent and creative...")
                    try:
                        sample_intent = intent_json.copy()
                        if isinstance(creative_json, dict):
                            sample_intent.update(creative_json)
                        else:
                            sample_intent['creative'] = creative_json
                        status_placeholder.success("Step 3: Merge completed")
                    except Exception as e:
                        status_placeholder.error(f"Step 3 error: {str(e)}\n{traceback.format_exc()}")  # Add try-except
                        process_ok = False

                # --- Step 4 & 5: Dispatch generation of SVL ---
                if process_ok:
                    status_placeholder.info("Step 4 & 5: Start dispatching generation of SVL...")
                    try:
                        os.makedirs(DISPATCH_OUTPUT_DIR, exist_ok=True)
                        intent_file = os.path.join(BASE_DIR, "dispatch_input", f"{sample_intent['algorithm_id']}_intent.json")
                        with open(intent_file, "w", encoding="utf-8") as f:
                            json.dump(sample_intent, f, ensure_ascii=False, indent=2)
                        dispatcher_script = os.path.join(BASE_DIR, "dispatcher.py")
                        result = subprocess.run(["python3", dispatcher_script, intent_file], check=True, cwd=BASE_DIR, capture_output=True, text=True)  # Capture output
                        dispatch_output_file = os.path.join(DISPATCH_OUTPUT_DIR, f"{sample_intent['algorithm_id']}_svl_5.0.json")
                        if os.path.exists(dispatch_output_file):
                            status_placeholder.success(f"Step 5: Dispatcher execution completed, SVL file generated")
                        else:
                            status_placeholder.error(f"Step 5 failed: Dispatcher output file not found")
                            process_ok = False
                    except subprocess.CalledProcessError as e:
                        status_placeholder.error(f"Step 4/5 error: {str(e)}\nOutput: {e.output}\nStderr: {e.stderr}")  # Improved error
                        process_ok = False
                    except Exception as e:
                        status_placeholder.error(f"Step 4/5 error: {str(e)}\n{traceback.format_exc()}")  # Improved error
                        process_ok = False

                # --- Step 6: Call renderer.py to generate TeX file ---
                if process_ok:
                    status_placeholder.info("Step 6: Start rendering TeX file...")
                    try:
                        renderer_script = os.path.join(BASE_DIR, "renderer.py")
                        result = subprocess.run(["python3", renderer_script, dispatch_output_file], check=True, cwd=BASE_DIR, capture_output=True, text=True)  # Capture
                        status_placeholder.success("Step 6: TeX file rendering completed")
                    except subprocess.CalledProcessError as e:
                        status_placeholder.error(f"Step 6 error: {str(e)}\nOutput: {e.output}\nStderr: {e.stderr}")
                        process_ok = False
                    except Exception as e:
                        status_placeholder.error(f"Step 6 error: {str(e)}\n{traceback.format_exc()}")
                        process_ok = False

                # --- Step 7: Call tex_to_png.py to convert to PNG ---
                if process_ok:
                    status_placeholder.info("Step 7: Start converting TeX to PNG...")
                    try:
                        tex_output_dir = os.path.join(DISPATCH_OUTPUT_DIR, f"{sample_intent['algorithm_id']}_svl_5.0_frames")
                        image_dir = os.path.join(BASE_DIR, "image")
                        tex_to_png_script = os.path.join(BASE_DIR, "tex_to_png.py")
                        result = subprocess.run(["python3", tex_to_png_script, tex_output_dir, image_dir], check=True, cwd=BASE_DIR, capture_output=True, text=True)  # Capture
                        status_placeholder.success("Step 7: PNG conversion completed, moved to image directory")
                    except subprocess.CalledProcessError as e:
                        status_placeholder.error(f"Step 7 error: {str(e)}\nOutput: {e.output}\nStderr: {e.stderr}")
                        process_ok = False
                    except Exception as e:
                        status_placeholder.error(f"Step 7 error: {str(e)}\n{traceback.format_exc()}")
                        process_ok = False
            
            if process_ok:
                status_placeholder.success("All steps completed! Please switch to 'Image playback' mode to view the result.")
                
                # --- NEW FIX: Save control_object to a file for persistence ---
                control_to_save = {}
                if creative_json and isinstance(creative_json, dict):
                    control_to_save = creative_json.get("control_object", {})
                
                with open(CONTROL_OBJECT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(control_to_save, f, ensure_ascii=False, indent=2)
            else:
                # Remove [DEBUG] related outputs
                status_placeholder.error("Process interrupted, please check the error information.")
