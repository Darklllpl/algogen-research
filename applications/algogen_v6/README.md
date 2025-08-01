# AlgoGen v6.0 - Algorithm Visualization Application

AlgoGen v6.0 是一个完整的算法可视化应用系统，基于SVL 5.0规范构建，提供算法执行过程的可视化展示和交互功能。

## 系统架构

### 核心组件

#### 1. Web应用 (`web/`)
- **app.py** - Flask Web应用主程序
- **dispatcher.py** - 请求分发器
- **renderer.py** - 算法可视化渲染器
- **llm_creative.py** - LLM创意生成模块
- **llm_intent_recognition.py** - LLM意图识别模块
- **deepseek.py** - DeepSeek模型集成
- **tex_to_png.py** - LaTeX转PNG工具

#### 2. 数据处理 (`data_processing/`)
- **create_gemma_dataset_v5.py** - 生成Gemma v5数据集
- **generate_instruction_pairs.py** - 生成指令对数据集
- **generate_gold_standards.py** - 生成黄金标准数据
- **shuffle_jsonl.py** - JSONL文件随机化
- **validate.py** - 数据验证工具

#### 3. 评估系统 (`evaluation/`)
- **evaluate_accuracy.py** - 准确率评估
- **evaluate_id_and_input_only.py** - ID和输入评估
- **test_llm_to_dataset.py** - LLM到数据集测试

#### 4. 可视化渲染 (`visualization/`)
- **renderer.py** - 主渲染器
- **final_styles.py** - 最终样式定义
- **default_styles.py** - 默认样式
- **style_merger.py** - 样式合并工具
- **convert_frames.sh** - 帧转换脚本
- **render_all.sh** - 批量渲染脚本

## 数据集

### 黄金标准数据集 (`gold_standards/`)
包含100个算法可视化黄金标准，涵盖：
- 排序算法（冒泡排序、快速排序等）
- 图论算法（Dijkstra、BFS、DFS等）
- 动态规划算法
- 其他经典算法

### 指令对数据集
- **instruction_pairs_dataset_with_model_output.jsonl** - 包含模型输出的指令对数据集
- **wrong.jsonl** - 错误案例数据集

## 快速开始

### 环境要求
```bash
pip install flask transformers torch numpy matplotlib
```

### 启动Web应用
```bash
cd web
python app.py
```

### 生成数据集
```bash
python create_gemma_dataset_v5.py
python generate_instruction_pairs.py
python generate_gold_standards.py
```

### 运行评估
```bash
python evaluate_accuracy.py
```

### 渲染可视化
```bash
bash render_all.sh
bash convert_frames.sh
```

## 使用示例

### 1. 创建算法可视化
```python
from renderer import SVLRenderer

# 创建渲染器
renderer = SVLRenderer()

# 加载SVL数据
with open('gold_standard_000.json', 'r') as f:
    svl_data = json.load(f)

# 渲染可视化
frames = renderer.render(svl_data)
```

### 2. Web界面使用
1. 访问 `http://localhost:5000`
2. 上传SVL文件或选择预设算法
3. 查看算法执行过程
4. 调整播放速度和样式

### 3. 数据集生成
```python
from create_gemma_dataset_v5 import create_dataset

# 生成Gemma数据集
create_dataset(
    input_dir='gold_standards',
    output_file='gemma_dataset_v5.jsonl'
)
```

## 配置选项

### 样式配置 (`final_styles.py`)
- 定义各种算法元素的视觉样式
- 支持自定义颜色、字体、动画效果

### 播放器配置
- 默认播放速度
- 自动播放选项

## 扩展开发

### 添加新算法
1. 创建SVL 5.0格式的数据文件
2. 定义算法特定的样式
3. 添加到黄金标准数据集

### 自定义渲染器
1. 继承SVLRenderer类
2. 实现自定义渲染逻辑
3. 注册新的操作类型

## 故障排除

### 常见问题
1. **LaTeX渲染错误**: 检查tex_to_png.py配置
2. **样式加载失败**: 验证final_styles.py语法
3. **数据集格式错误**: 使用validate.py验证

### 调试模式
```bash
export FLASK_DEBUG=1
python app.py
```