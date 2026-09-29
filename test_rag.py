import ollama
import chromadb
# ========== 测试1：调用 nomic-embed-text 生成向量 ==========
try:
    resp = ollama.embeddings(
        model="nomic-embed-text",
        prompt="这是NPC人设测试文本"
    )
    vector = resp["embedding"]
    print("✅ 向量模型调用成功！向量维度：", len(vector))
except Exception as e:
    print("❌ 向量模型失败：", e)
# ========== 测试2：ChromaDB 创建集合，插入+查询向量 ==========
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="npc_knowledge")

    # 插入测试文档
    collection.upsert(
        ids=["doc_001"],
        documents=["NPC名字是林默，是一名酒馆老板，性格沉稳，熟知小镇所有传闻"],
        embeddings=[vector]
    )
    # 相似检索测试
    query_resp = ollama.embeddings(
        model="nomic-embed-text",
        prompt="酒馆老板是谁"
    )
    results = collection.query(
        query_embeddings=[query_resp["embedding"]],
        n_results=1
    )
    print("✅ ChromaDB 检索成功！")
    print("检索结果：", results["documents"])
except Exception as e:
    print("❌ ChromaDB 测试失败：", e)