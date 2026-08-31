# LightRAG 查询接口压力测试

`query_stress_test.py` 是独立运行的异步压力测试脚本。它不会导入或修改
LightRAG 运行状态，只依赖项目已有的 `aiohttp`。

每个请求都会分别从 `dynamic_parameters` 的每个数组随机取一个值，并从
`query_generation.templates` 及其变量中随机生成查询。两个端点默认以轮询
方式均衡请求：

- `POST /api/query/data`
- `POST /api/query`

## 快速开始

先复制示例配置并按实际 workspace、问题和上下文修改：

```bash
cp scripts/query_stress_config.example.json query_stress_config.json
python scripts/query_stress_test.py query_stress_config.json --dry-run
python scripts/query_stress_test.py query_stress_config.json
```

可从命令行临时覆盖请求数、并发数和随机种子：

```bash
python scripts/query_stress_test.py query_stress_config.json \
  --requests 1000 --concurrency 50 --seed 42
```

执行后会在 `output.directory` 中生成：

- `requests_*.jsonl`：每个请求的状态码、耗时、workspace 和错误；
- `summary_*.json`：实际压测设置、总体及分端点的吞吐、成功率和延迟分位数。

默认只要有请求失败，进程就返回状态码 1。若只想采集结果，可设置
`load.fail_on_error` 为 `false`。`load.requests_per_second` 为 `null` 时不主动
限速；设置为正数后会限制全局请求启动速率。

## 动态参数

`dynamic_parameters` 支持任意请求字段，每次请求对各字段独立随机取值，
从而形成随机笛卡尔组合。例如：

```json
{
  "dynamic_parameters": {
    "workspace": ["default", "project_a"],
    "top_k": [5, 10, 20],
    "enable_rerank": [true, false]
  }
}
```

数组中的元素会原样放入请求体，所以数组、对象等复合参数也可以随机化：

```json
{
  "dynamic_parameters": {
    "hl_keywords": [
      ["架构", "模块"],
      ["部署", "性能"]
    ],
    "conversation_history": [
      [],
      [{"role": "user", "content": "请关注查询性能"}]
    ]
  }
}
```

`query_generation` 会为每个模板变量随机取值，然后随机选择模板进行渲染：

```json
{
  "query_generation": {
    "templates": ["问题：{question}\n检索上下文：{agent_context}"],
    "variables": {
      "question": ["系统如何工作？", "如何降低查询延迟？"],
      "agent_context": ["关注模块调用链。", "关注缓存和重排序。"]
    }
  }
}
```

也可以从 UTF-8 CSV 文件中为每个请求独立随机抽取一个问题。以下配置读取
`scripts/questions.csv` 的 `question` 列，并把抽到的内容作为模板中的
`{question}` 变量：

```json
{
  "query_generation": {
    "csv": {
      "path": "scripts/questions.csv",
      "column": "question",
      "variable": "question"
    },
    "templates": ["{question}"],
    "variables": {}
  }
}
```

采样采用有放回方式，因此请求数可以大于 CSV 行数。CSV 路径会先相对于配置
文件所在目录解析，再相对于当前工作目录解析。若不配置 `templates`，抽到的
CSV 内容会直接作为请求的 `query`。

如果不需要模板，也可以删除 `query_generation.templates`，改用：

```json
{
  "dynamic_parameters": {
    "query": ["问题一", "问题二", "问题三"]
  }
}
```

## `agent_search` 上下文说明

当前 LightRAG `QueryRequest.agent_search` 是布尔开关，不能直接承载上下文
字符串。示例配置保持 `"agent_search": true`，并通过 `agent_context` 模板变量
把随机上下文注入 `query`，该 query 会进入 agent_search 的检索目标。若把
`agent_search` 配置成文本，脚本会在发压前直接报告配置错误。

如果服务启用了 API Key，可在 `headers` 中添加：

```json
{
  "headers": {
    "Accept": "application/json",
    "X-API-Key": "替换为实际密钥"
  }
}
```

不要提交包含真实密钥的配置文件。
