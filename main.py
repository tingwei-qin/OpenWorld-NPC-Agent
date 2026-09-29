import os
import time
import json
import logging
# 屏蔽chromadb遥测日志
logging.getLogger("chromadb").setLevel(logging.ERROR)
import redis
import ollama
import chromadb
# ===================== 初始化组件 =====================
# Redis 短期对话记忆（增加自动重连配置）
r = redis.Redis(
    host="127.0.0.1",
    port=6379,
    decode_responses=True,
    protocol=2,
    socket_connect_timeout=5,
    socket_keepalive=True,
    retry_on_timeout=True
)
REDIS_KEY = "npc_chat_history"
# ChromaDB 知识库，关闭匿名遥测
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=chromadb.Settings(anonymized_telemetry=False)
)
collection = chroma_client.get_or_create_collection(name="npc_knowledge")
# NPC 系统人设【已增加防幻觉约束】
NPC_SYSTEM_PROMPT = """
你是林默，小镇酒馆老板。
性格沉稳温和，熟悉小镇所有传闻，说话自然，像真人NPC，不要暴露你是AI。
回答简短自然，根据玩家对话和知识库信息进行回应。
重要约束：如果没有历史对话记录，绝对不要编造和玩家的过往聊天内容。当不清楚之前对话时，如实回答：我不记得刚才聊过什么了。
"""
# ===================== 【新增1：工具定义 Function Calling】 =====================
def get_player_favor(player_id: str) -> str:
    """查询酒馆老板林默对玩家的好感度"""
    favor_map = {"player01": "友好", "player02": "敌对", "player03": "普通"}
    return favor_map.get(player_id, "陌生")

def query_item_stock(item_name: str) -> str:
    """查询小镇酒馆物品库存"""
    stock_map = {"苹果": "充足", "麦酒": "充足", "药水": "短缺", "长剑": "售罄"}
    return stock_map.get(item_name, "酒馆没有该物品")
# 给大模型识别的工具描述
tools = [
    {
        "name": "get_player_favor",
        "description": "查询酒馆老板林默对玩家的好感度，参数player_id玩家编号",
        "parameters": {
            "type": "object",
            "properties": {"player_id": {"type": "string", "description": "玩家ID，例如player01"}},
            "required": ["player_id"]
        }
    },
    {
        "name": "query_item_stock",
        "description": "查询酒馆物品库存，参数item_name物品名称",
        "parameters": {
            "type": "object",
            "properties": {"item_name": {"type": "string", "description": "物品名称，例如麦酒"}},
            "required": ["item_name"]
        }
    }
]
tool_map = {
    "get_player_favor": get_player_favor,
    "query_item_stock": query_item_stock
}
# ===================== 【新增2：Agent1 知识库智能体】 =====================
class KnowledgeAgent:
    def __init__(self):
        self.collection = collection
    def search_knowledge(self, query:str, top_k=2):
        """知识库Agent，专门负责RAG检索，复用你原有get_embedding"""
        return rag_retrieve(query, top_k)
# ===================== 【新增3：Agent2 NPC对话智能体（支持工具调用）】 =====================
class NPCChatAgent:
    def __init__(self, tools, tool_map):
        self.tools = tools
        self.tool_map = tool_map
        self.model_name = "qwen2:7b"
        self.system_prompt = NPC_SYSTEM_PROMPT
    def chat(self, user_input:str, world_knowledge:str, history:list, player_id:str):
        messages = [{"role": "system", "content": self.system_prompt + f"\n【知识库参考信息】{world_knowledge}"}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        max_retries = 2
        reply = ""
        for attempt in range(max_retries):
            try:
                resp = ollama.chat(
                    model=self.model_name,
                    messages=messages,
                    tools=self.tools
                )
                msg = resp["message"]
                # 判断是否触发工具调用
                if msg.get("tool_calls"):
                    for tool_call in msg["tool_calls"]:
                        func_name = tool_call["function"]["name"]
                        args = tool_call["function"]["arguments"]
                        func = self.tool_map[func_name]
                        tool_result = func(**args)
                        # 工具返回结果送入大模型生成最终回答
                        final_messages = messages + [msg, {"role":"tool", "content":str(tool_result)}]
                        final_resp = ollama.chat(model=self.model_name, messages=final_messages)
                        reply = final_resp["message"]["content"]
                else:
                    reply = msg["content"]
                break
            except Exception as e:
                print(f"模型调用失败，第{attempt + 1}次重试：", e)
                if attempt == max_retries -1:
                    reply = "抱歉，我现在有点忙，稍后再和你聊吧。"
        return reply
# ===================== 【你原来所有函数全部保留，下面没有改动！】 =====================
def get_embedding(text):
    """调用ollama向量模型，增加异常处理"""
    try:
        res = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return res["embedding"]
    except Exception as e:
        print("向量模型调用失败：", e)
        return None
def rag_retrieve(query, top_k=2):
    """RAG检索知识库，失败时返回空字符串"""
    try:
        emb = get_embedding(query)
        if not emb:
            return ""
        res = collection.query(query_embeddings=[emb], n_results=top_k)
        docs = res["documents"][0]
        return "\n".join(docs)
    except Exception as e:
        print("知识库检索失败：", e)
        return ""
def load_chat_history(max_turns=8):
    """从redis读取历史对话，只保留最近max_turns轮对话，避免Token溢出"""
    raw = r.lrange(REDIS_KEY, 0, -1)
    history = []
    for item in raw:
        history.append(json.loads(item))
    # 只保留最近max_turns轮对话（每轮对话是user+assistant，所以乘以2）
    if len(history) > max_turns * 2:
        history = history[-max_turns * 2:]
    return history
def save_chat(user_msg, npc_reply):
    """把本轮对话存入redis"""
    r.rpush(REDIS_KEY, json.dumps({"role": "user", "content": user_msg}))
    r.rpush(REDIS_KEY, json.dumps({"role": "assistant", "content": npc_reply}))
def count_tokens(messages):
    """统计对话Token占用"""
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # 兼容qwen2的token统计
        total_tokens = sum(len(encoding.encode(msg["content"])) for msg in messages)
        return total_tokens
    except Exception as e:
        print("Token统计失败：", e)
        return 0
# ===================== 【升级：新版多智能体调度函数，替换原来npc_chat】 =====================
def npc_chat_multi_agent(user_input, player_id="player01"):
    start_time = time.time()
    # Step1：知识库Agent执行RAG检索
    knowledge_agent = KnowledgeAgent()
    knowledge = knowledge_agent.search_knowledge(user_input)
    # Step2：读取Redis历史对话
    history = load_chat_history(max_turns=8)
    # Step3：实例化对话Agent，执行推理+工具调用
    chat_agent = NPCChatAgent(tools, tool_map)
    reply = chat_agent.chat(user_input, knowledge, history, player_id)
    # Step4：保存对话到Redis
    save_chat(user_input, reply)
    # Step5：统计耗时
    end_time = time.time()
    latency = end_time - start_time
    print(f"本次对话响应延迟：{latency:.2f}秒")
    return reply
# ===================== 交互主循环【修改调用多智能体版本】 =====================
if __name__ == "__main__":
    print("===== 多智能体NPC酒馆老板林默 已上线，输入quit退出，输入/reset清空记忆 =====")
    while True:
        user_q = input("你：")
        if user_q.strip().lower() == "quit":
            print("林默：欢迎下次再来酒馆！")
            break
        if user_q.strip().lower() == "/reset":
            r.delete(REDIS_KEY)
            print("✅ 已清空对话记忆")
            continue
        answer = npc_chat_multi_agent(user_q)
        print("林默：", answer)