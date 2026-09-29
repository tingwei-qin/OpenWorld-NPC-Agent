import chromadb
import ollama
# 初始化向量库
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=chromadb.Settings(anonymized_telemetry=False)
)
collection = chroma_client.get_or_create_collection(name="npc_knowledge")
# 小镇知识库文本
knowledge_list = [
    "林默的酒馆坐落在黑石小镇的中心，酒馆后院有一口老井，传说深夜会听到井里传来低语。",
    "黑石小镇最近有传闻，森林深处出现了游荡的黑影，不少猎户不敢进山。",
    "林默早年是一名冒险者，后来受伤才开了这家酒馆，他见识广博，听过很多大陆的传闻。",
    "酒馆售卖麦酒和烤面包，晚上会有流浪的旅人、猎人来这里歇脚。"
]
# 批量写入向量库
for idx, text in enumerate(knowledge_list):
    emb = ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]
    collection.add(
        embeddings=[emb],
        documents=[text],
        ids=[f"doc_{idx}"]
    )
print("✅ 知识库写入完成！")