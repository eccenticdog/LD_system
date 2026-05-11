# 实验室文档系统
该项目是面向实验室技术文档，设备说明文档以及使用手册等资料，构建智能问答Agent系统，覆盖文档上传、私有知识库问答、实时信息补充及复杂问题处理等核心能力。

## 核心能力

- 支持 `PDF` / `Markdown` 文档导入
- 基于 `MinerU` 进行 PDF 转 Markdown 与版面解析
- 基于 `BGE-M3` 生成稠密向量 + 稀疏向量，进行混合检索
- 使用 `Milvus` 存储切片与向量数据
- 使用 `Neo4j` 抽取并存储知识图谱三元组
- 使用 `MongoDB` 存储会话历史
- 使用 `MinIO` 存储原始文件与图片资源
- 查询阶段支持多路召回：
  - 本地向量检索
  - `HyDE` 假设文档检索
  - 知识图谱事实检索
  - MCP 联网搜索
- 使用 `RRF` 做召回融合，使用 `Reranker` 做精排
- 支持普通返回和 `SSE` 流式返回
- 提供基础评测脚本，用于验证 `HyDE + Vector` 检索效果

## 整体流程

### 1. 导入流程

文档导入服务会把上传文件送入一条 LangGraph 工作流：

`文件上传 -> PDF/MD 解析 -> 图片处理 -> 文档切分 -> 产品名识别 -> 向量生成 -> 写入 Milvus -> 抽取三元组写入 Neo4j`

如果上传的是 Markdown，会直接进入 Markdown 处理节点；如果上传的是 PDF，会先走 PDF 转 Markdown。

### 2. 查询流程

查询服务会执行另一条 LangGraph 工作流：

`问题理解/产品确认 -> 多路召回 -> RRF 融合 -> Rerank 精排 -> LLM 生成答案 -> 写入 Mongo 历史`

其中“多路召回”包含：

- 本地向量检索
- HyDE 检索
- 知识图谱检索
- MCP 联网搜索

## 技术栈

- 后端框架：`FastAPI`
- 编排框架：`LangGraph`
- 大模型接入：OpenAI 兼容接口
- 向量模型：`BGE-M3`
- 重排模型：`BGE Reranker`
- 向量数据库：`Milvus`
- 图数据库：`Neo4j`
- 历史会话：`MongoDB`
- 对象存储：`MinIO`
- 文档解析：`MinerU`
- 评测工具：`ragas`

## 目录结构

```text
app/
  clients/                # Milvus / Neo4j / Mongo / MinIO 客户端封装
  conf/                   # 环境配置读取
  core/                   # 日志、Prompt 加载等基础模块
  import_process/         # 文档导入服务与导入工作流
  lm/                     # LLM、Embedding、Reranker 封装
  query_process/          # 查询服务与问答工作流
  utils/                  # 工具函数
doc/                      # 示例文档、手册资料
evaluation/               # 检索评测脚本
output/                   # 导入过程输出目录
prompts/                  # 提示词模板
test/                     # 本地测试脚本
```

## 运行依赖

启动前建议先准备以下基础设施：

- `Python 3.11+`
- `Milvus`
- `MongoDB`
- `Neo4j`（如需知识图谱能力）
- `MinIO`（推荐）
- `MinerU` 服务
- 一个 OpenAI 兼容的大模型接口
- 一个可选的 MCP 搜索服务

## 环境变量

项目通过根目录 `.env` 读取配置。放到 GitHub 前，建议不要提交真实密钥，改为保留示例配置。

常用变量如下：

```env
# LLM
OPENAI_API_KEY=
OPENAI_BASE_URL=
LLM_DEFAULT_MODEL=
VL_MODEL=
LLM_DEFAULT_TEMPERATURE=

# Embedding / Reranker
BGE_M3_PATH=
BGE_M3=
BGE_DEVICE=
BGE_FP16=
BGE_RERANKER_LARGE=
BGE_RERANKER_DEVICE=
BGE_RERANKER_FP16=

# Milvus
MILVUS_URL=
CHUNKS_COLLECTION=
ITEM_NAME_COLLECTION=
ENTITY_NAME_COLLECTION=

# Neo4j
NEO4J_URI=
NEO4J_DATABASE=
NEO4J_USERNAME=
NEO4J_PASSWORD=

# MongoDB
MONGO_URL=
MONGO_DB_NAME=

# MinIO
MINIO_ENDPOINT=
MINIO_ACCESS_KEY=
MINIO_SECRET_KEY=
MINIO_BUCKET_NAME=
MINIO_IMG_DIR=
MINIO_SECURE=
MINIO_PDF_DIR=

# MinerU
MINERU_BASE_URL=
MINERU_API_TOKEN=

# MCP Web Search
MCP_DASHSCOPE_BASE_URL=
```

## 快速开始

### 1. 安装依赖

项目根目录已经包含 `pyproject.toml` 和 `uv.lock`，推荐使用 `uv`：

```bash
uv sync
```

### 2. 启动文档导入服务

```bash
uv run python -m app.import_process.api.file_import_service
```

默认地址：

- `http://127.0.0.1:8000/import.html`
- `http://127.0.0.1:8000/docs`

### 3. 启动查询服务

```bash
uv run python -m app.query_process.api.query_service
```

默认地址：

- `http://127.0.0.1:8001/chat.html`
- `http://127.0.0.1:8001/docs`

## 主要接口

### 文档导入

- `POST /upload`
  - 支持多文件上传
  - 返回 `task_ids`

- `GET /status/{task_id}`
  - 查询导入任务状态

示例：

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "files=@doc/example.pdf"
```

### 查询问答

- `POST /query`
  - 入参：`query`、`session_id`、`is_stream`

- `GET /stream/{session_id}`
  - SSE 流式接收回答

- `GET /history/{session_id}`
  - 查询历史记录

- `DELETE /history/{session_id}`
  - 清空历史记录

示例：

```bash
curl -X POST "http://127.0.0.1:8001/query" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"HAK 180 怎么操作？\",\"is_stream\":false}"
```

## 评测

`evaluation/README.md` 中提供了 `HyDE + Vector` 检索链路的本地评测说明，可用于：

- 评估召回质量
- 比较不同参数配置
- 观察 `hit_rate`、`recall@k`、`mrr` 等指标

## 适合继续完善的方向

- 增加 `.env.example`，方便开源后快速部署
- 增加 `docker-compose`，统一管理 Milvus / MongoDB / Neo4j / MinIO
- 增加更完整的接口鉴权与权限管理
- 增加前端页面说明和效果截图
- 补充自动化测试与 CI 配置

## 项目特点总结

这不是一个单纯的“向量检索 Demo”，而是一个已经具备知识库导入、结构化存储、多路召回、重排生成、历史会话和基础评测能力的完整雏形。对于做企业文档问答、产品知识助手、售后手册助手这类场景，它已经有比较清晰的工程结构和扩展路径。
