# Codex 开发笔记

## 记忆后端重构计划

### 背景

当前项目一开始主要使用 LangGraph 的 `InMemoryStore` 作为默认记忆存储。现在项目里已经引入了 `mem0ai` SDK，后续也可能继续测试其他记忆模块。

需要明确一点：`LangGraph InMemoryStore` 和 `mem0` 不能当成同一种接口的两个兼容实现。

- `InMemoryStore` 更像 LangGraph 的通用 key/value 和向量检索存储，核心概念是 namespace、key、value。
- `mem0` 更像完整的记忆 SDK，自己带有记忆抽取、更新、去重、搜索和特定返回格式。

所以重构方向不是让它们互相兼容，而是让业务代码只依赖我们自己的 `MemoryBackend` 抽象。LangGraph store、mem0、以后其他记忆系统都作为 adapter 接到这个抽象后面。

### 当前方向

- 长期用户记忆统一走 `MemoryBackend`。
- 邮件线程上下文可以继续使用 LangGraph store，因为它更像工作流状态，不是长期用户记忆。
- graph 节点和 tool 不应该直接依赖 mem0 或 LangGraph store 的长期记忆 API。
- 各个后端的差异应该封装在 adapter 类内部。

### 当前实现状态

- 已新增 `memory_agent.memory_backends`。
- 已新增 `MemoryBackend` 协议。
- 已新增 `MemoryRecord`，作为后端无关的记忆搜索结果。
- 已新增 `LangGraphMemoryBackend`。
- 已新增初版 `Mem0MemoryBackend`，并使用延迟导入，避免默认环境必须能 import mem0。
- 已更新 `Context`，支持传入可选的 `memory_backend`。
- 已把 graph 和 email graph 里的长期记忆读写改为通过 backend resolver。
- 已更新 `EmailProcessingService`，支持注入外部 memory backend。
- 已新增 LangGraph backend 和 tool 注入路径的单元测试。

### 后续重构计划

1. 梳理存储职责。
   - 区分长期用户记忆、邮件线程上下文、临时运行状态。
   - 只有长期用户记忆放到 `MemoryBackend` 后面。

2. 稳定 `MemoryBackend` 契约。
   - 明确 `upsert`、`search`、`list`，以及可能的 `delete` 行为。
   - 统一 `memory_id`、`metadata`、`score`、记忆正文的返回格式。
   - 决定各后端是否必须支持真正更新，还是允许通过新增记忆来模拟更新。

3. 加固 mem0 adapter。
   - 按 mem0 真实的 `add`、`search`、`get_all` 请求和返回格式适配。
   - 不假设 mem0 的行为和 LangGraph `InMemoryStore` 一样。
   - 使用假的 mem0 client 或 test double 增加测试。

4. 明确拆分 service 里的存储用途。
   - `EmailProcessingService._store` 保留给 LangGraph 图运行状态和邮件线程上下文。
   - 长期记忆只通过 `memory_backend` 访问。
   - JSON archive 行为只作为 LangGraph memory backend fallback 使用。

5. 增加 contract tests。
   - 为 `MemoryBackend` 定义一套共享行为测试。
   - 对 `LangGraphMemoryBackend` 跑这套测试。
   - 对 `Mem0MemoryBackend` 用 fake/local test double 跑同一套测试。
   - 后续接入其他 memory 模块时复用同一套测试。

6. 增加配置入口。
   - 允许通过 CLI 参数或环境变量选择 memory backend。
   - 建议可选值：`langgraph`、`mem0`。
   - 在 mem0 行为完全验证前，默认继续使用 LangGraph backend。

7. 补充使用文档。
   - 在 README 里增加手动注入 backend 的示例。
   - 等配置入口完成后，再补充 CLI 切换 backend 的示例。

### 待确认问题

- mem0 应该负责从原始 conversation messages 中抽取记忆，还是继续由现有 graph 先抽取结构化记忆，再写入 backend？
- 现在是否需要跨后端的 delete/update 支持，还是下一阶段只需要 add/search/list？
- 记忆 metadata 是否继续保持自由 dict，还是项目应该定义更严格的 email-derived memory schema？

