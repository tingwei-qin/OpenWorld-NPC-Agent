import redis
import ollama
# ========== 测试1：连接Redis ==========
try:
    r = redis.Redis(
        host="127.0.0.1",
        port=6379,
        decode_responses=True,
        protocol=2
    )
    r.ping()
    print("✅ Redis 连接成功！")
except Exception as e:
    print("❌ Redis 连接失败：", e)
# ========== 测试2：调用本地Ollama qwen2:7b ==========
try:
    resp = ollama.chat(
        model="qwen2:7b",
        messages=[{"role": "user", "content": "你好，请回复一句话证明你正常运行"}]
    )
    print("✅ Ollama 大模型调用成功！")
    print("模型返回：", resp["message"]["content"])
except Exception as e:
    print("❌ Ollama 调用失败：", e)