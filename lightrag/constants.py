"""
Centralized configuration constants for LightRAG.

This module defines default values for configuration constants used across
different parts of the LightRAG system. Centralizing these values ensures
consistency and makes maintenance easier.
"""

# Default values for server settings
DEFAULT_WOKERS = 2
DEFAULT_MAX_GRAPH_NODES = 1000

# Default values for extraction settings
DEFAULT_SUMMARY_LANGUAGE = "English"  # Default language for document processing
DEFAULT_MAX_GLEANING = 1
DEFAULT_ENTITY_NAME_MAX_LENGTH = 256

# Number of description fragments to trigger LLM summary
DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE = 8
# Max description token size to trigger LLM summary
DEFAULT_SUMMARY_MAX_TOKENS = 1200
# Recommended LLM summary output length in tokens
DEFAULT_SUMMARY_LENGTH_RECOMMENDED = 600
# Maximum token size sent to LLM for summary
DEFAULT_SUMMARY_CONTEXT_SIZE = 12000
# Default entities to extract if ENTITY_TYPES is not specified in .env
# DEFAULT_ENTITY_TYPES = [
#     "Person",
#     "Creature",
#     "Organization",
#     "Location",
#     "Event",
#     "Concept",
#     "Method",
#     "Content",
#     "Data",
#     "Artifact",
#     "NaturalObject",
# ]

DEFAULT_ENTITY_TYPES = {
    # ===== 文献与元数据层 =====
    "Person": "论文作者、研究人员或相关专家个体",
    "Organization": "高校、研究机构、核电企业、实验室等组织",
    "Project": "科研项目、计划或课题名称",
    "Document": "论文、报告、专利、技术文档等",
    "Standard": "标准、规范、导则或法规文件",
    "Location": "实验地点、机构所在地或工程场址",
    "Time": "时间信息，如年份、实验周期或运行阶段",

    # ===== 科技论文核心语义层 =====
    "ResearchTopic": "论文研究的核心问题或主题",
    "DomainConcept": "领域内的专业术语或理论概念",
    "Method": "通用研究方法或技术路线",
    "ExperimentalMethod": "实验或测试方法",
    "SimulationMethod": "数值模拟或计算方法（如CFD、蒙特卡洛）",
    "Algorithm": "用于计算、优化或反演的算法",
    "Model": "理论模型、物理模型或经验模型",
    "Equation": "数学方程、控制方程或经验公式",

    # ===== 材料与化学层 =====
    "Material": "工程材料或功能材料（如锆合金、不锈钢）",
    "Substance": "具体物质、元素、同位素或化合物",
    "Property": "材料或介质的物理或化学性质",
    "Composition": "材料成分、配比或浓度信息",
    "Microstructure": "材料微观结构（晶粒、析出相等）",
    "Degradation": "材料性能退化过程（如腐蚀、疲劳、蠕变）",
    "FailureMode": "材料或部件的失效形式（如断裂、开裂）",

    # ===== 核电工程层 =====
    "ReactorType": "核反应堆类型（如压水堆、沸水堆）",
    "System": "核电厂系统（如一回路、安注系统）",
    "Component": "设备或部件（如蒸汽发生器、管道）",
    "CoreComponent": "堆芯关键部件（如燃料组件、控制棒）",
    "Fuel": "核燃料类型或形式",
    "Coolant": "反应堆冷却剂",
    "AccidentType": "事故类型（如失水事故LOCA）",
    "SafetyFunction": "核安全功能（如停堆、余热导出）",

    # ===== 工况与物理量 =====
    "Condition": "运行工况或边界条件",
    "PhysicalQuantity": "物理量（如温度、压力、流量）",
    "Parameter": "模型或实验中的关键参数",
    "Phenomenon": "物理或化学现象（如沸腾、空化）",
    "Mechanism": "现象或过程背后的作用机理",
    "Result": "实验或仿真得到的结果或结论",
    "Metric": "评价指标或性能指标",
    "Dataset": "实验或仿真数据集合",

    # ===== 工具与检测 =====
    "Software": "仿真软件、计算平台或代码系统",
    "Instrument": "实验或测量仪器设备",
    "Sensor": "传感器或检测元件",

    # ===== 核物理与辐射 =====
    "Nuclide": "核素或同位素",
    "RadiationEffect": "辐照效应或辐射相关影响"
}

# Separator for: description, source_id and relation-key fields(Can not be changed after data inserted)
GRAPH_FIELD_SEP = "<SEP>"

# Query and retrieval configuration defaults
DEFAULT_TOP_K = 40
DEFAULT_CHUNK_TOP_K = 20
DEFAULT_MAX_ENTITY_TOKENS = 6000
DEFAULT_MAX_RELATION_TOKENS = 8000
DEFAULT_MAX_TOTAL_TOKENS = 30000
DEFAULT_COSINE_THRESHOLD = 0.2
DEFAULT_RELATED_CHUNK_NUMBER = 5
DEFAULT_KG_CHUNK_PICK_METHOD = "VECTOR"

# TODO: Deprated. All conversation_history messages is send to LLM.
DEFAULT_HISTORY_TURNS = 0

# Rerank configuration defaults
DEFAULT_MIN_RERANK_SCORE = 0.0
DEFAULT_RERANK_BINDING = "null"

# Default source ids limit in meta data for entity and relation
DEFAULT_MAX_SOURCE_IDS_PER_ENTITY = 300
DEFAULT_MAX_SOURCE_IDS_PER_RELATION = 300
### control chunk_ids limitation method: FIFO, FIFO
###    FIFO: First in first out
###    KEEP: Keep oldest (less merge action and faster)
SOURCE_IDS_LIMIT_METHOD_KEEP = "KEEP"
SOURCE_IDS_LIMIT_METHOD_FIFO = "FIFO"
DEFAULT_SOURCE_IDS_LIMIT_METHOD = SOURCE_IDS_LIMIT_METHOD_FIFO
VALID_SOURCE_IDS_LIMIT_METHODS = {
    SOURCE_IDS_LIMIT_METHOD_KEEP,
    SOURCE_IDS_LIMIT_METHOD_FIFO,
}
# Maximum number of file paths stored in entity/relation file_path field (For displayed only, does not affect query performance)
DEFAULT_MAX_FILE_PATHS = 100

# Field length of file_path in Milvus Schema for entity and relation (Should not be changed)
# file_path must store all file paths up to the DEFAULT_MAX_FILE_PATHS limit within the metadata.
DEFAULT_MAX_FILE_PATH_LENGTH = 32768
# Placeholder for more file paths in meta data for entity and relation (Should not be changed)
DEFAULT_FILE_PATH_MORE_PLACEHOLDER = "truncated"

# Default temperature for LLM
DEFAULT_TEMPERATURE = 1.0

# Async configuration defaults
DEFAULT_MAX_ASYNC = 4  # Default maximum async operations
DEFAULT_MAX_PARALLEL_INSERT = 2  # Default maximum parallel insert operations

# Embedding configuration defaults
DEFAULT_EMBEDDING_FUNC_MAX_ASYNC = 8  # Default max async for embedding functions
DEFAULT_EMBEDDING_BATCH_NUM = 10  # Default batch size for embedding computations

# Gunicorn worker timeout
DEFAULT_TIMEOUT = 300

# Default llm and embedding timeout
DEFAULT_LLM_TIMEOUT = 180
DEFAULT_EMBEDDING_TIMEOUT = 30

# Logging configuration defaults
DEFAULT_LOG_MAX_BYTES = 10485760  # Default 10MB
DEFAULT_LOG_BACKUP_COUNT = 5  # Default 5 backups
DEFAULT_LOG_FILENAME = "lightrag.log"  # Default log filename

# Ollama server configuration defaults
DEFAULT_OLLAMA_MODEL_NAME = "lightrag"
DEFAULT_OLLAMA_MODEL_TAG = "latest"
DEFAULT_OLLAMA_MODEL_SIZE = 7365960935
DEFAULT_OLLAMA_CREATED_AT = "2024-01-15T00:00:00Z"
DEFAULT_OLLAMA_DIGEST = "sha256:lightrag"
