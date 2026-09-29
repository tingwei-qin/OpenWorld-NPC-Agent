# OpenWorld-NPC-Agent
基于RAG + Function Calling 构建的AI NPC智能体，支持知识库检索、对话问答，可作为通用Agent应用。

## ✨项目简介
本项目构建大模型RAG智能体，实现自定义知识库导入、向量存储、检索增强问答。
- 采用Chroma向量数据库，存储文档向量
- 支持Function Calling，拓展Agent能力
- 实现知识入库、连接测试、RAG问答核心模块

## 📁项目结构
- add_knowledge.py：文档向量化，存入向量库
- main.py：Agent主入口，对话交互逻辑
- test_connect.py：向量库连接测试
- test_rag.py：RAG检索问答测试

## 🛠️技术栈
Python | LangChain | Chroma | Ollama | RAG | Function Calling

## ▶️运行步骤
1. 安装依赖包
2. 执行 add_knowledge.py 导入知识库文档
3. 运行 main.py 启动NPC智能体，进行对话交互

## 📌项目亮点
将检索增强生成与Agent工具调用结合，解决大模型幻觉问题，适合业务场景的问答智能体开发，可拓展多工具调用能力。
