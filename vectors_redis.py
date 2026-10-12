from tqdm import tqdm
import json
import uuid
import time
import redis
from zai import ZhipuAiClient
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction


# redis处理模块
def get_redis_client():
    # 创建Redis连接, 使用连接池 (推荐用于生产环境)
    pool = redis.ConnectionPool(host='0.0.0.0', port=6379, db=0, password=None, max_connections=10)
    r = redis.StrictRedis(connection_pool=pool)
    # 测试连接
    try:
        r.ping()
        print("成功连接到 Redis !")
    except r.ConnectionError:
        print("无法连接到 Redis !")
    return r


# 将 (question, answer) 问答对, 存入 redis
def cache_set(r, question: str, answer: str):
    r.hset("qa", question, answer)
    r.expire("qa", 3600)


# 通过 question, 读取存在 redis 中的 answer
def cache_get(r, question: str):
    return r.hget("qa", question)


# 实例化智谱client对象
client = ZhipuAiClient(api_key="****************************************")


class ZhipuAIEmbeddings(Embeddings):
    def __init__(self, client):
        self.client = client

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            # 调用清华智谱最新版本的 embeddings 方法
            response = self.client.embeddings.create(
                model="embedding-3",
                input=[text],
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text):
        # 查询文档
        return self.embed_documents([text])[0]


class Milvus_vector():
    def __init__(self, client, uri="./milvus_agent.db"):
        self.URI = uri
        self.embeddings = ZhipuAIEmbeddings(client=client)
        # 定义索引类型
        self.dense_index = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
        }
        self.sparse_index = {
            "metric_type": "BM25",
            "index_type": "SPARSE_INVERTED_INDEX"
        }

    def create_vector_store(self, docs):
        init_docs = docs[:10]
        self.vectorstore = Milvus.from_documents(
            documents=init_docs,
            embedding=self.embeddings,
            builtin_function=BM25BuiltInFunction(),  # output_field_names="sparse",
            index_params=[self.dense_index, self.sparse_index],
            vector_field=["dense", "sparse"],
            connection_args={
                "uri": self.URI,
            },
            consistency_level="Bounded",  # 支持 ("Strong", "Session", "Bounded", "Eventually")
            drop_old=False,
        )

        print("已初始化创建 Milvus ‼")
        count = 10
        temp = []
        for doc in tqdm(docs[10:]):
            temp.append(doc)
            if len(temp) >= 5:
                self.vectorstore.aadd_documents(temp)
                count += len(temp)
                temp = []
                print(f"已插入 {count} 条数据......")
                time.sleep(1)
        print(f"总共插入 {count} 条数据......")
        print("已创建 Milvus 索引完成 ‼")
        return self.vectorstore


# 插入数据
def prepare_document(file_path=['./data/data.jsonl', './data/train.jsonl']):
    # 逐条取出文本数据, 创建嵌入张量, 然后将张量数据插入Milvus
    file_path1 = file_path[0]
    count = 0
    docs = []
    with open(file_path1, 'r', encoding='utf-8') as f:
        for line in f:
            content = json.loads(line.strip())
            prompt = content['query'] + "\n" + content['response']
            temp_doc = Document(page_content=prompt, metadata={"doc_id": str(uuid.uuid4())})
            docs.append(temp_doc)
            count += 1
    print(f"已加载 {count} 条数据!")
    return docs


if __name__ == "__main__":
    # 预处理即将插入 Milvus 的文档数据
    docs = prepare_document()
    print("预处理文档数据成功......")
    # 创建 Milvus 连接
    milvus_vectorstore = Milvus_vector(client)
    print("创建Milvus连接成功......")
    # 创建向量索引
    vectorstore = milvus_vectorstore.create_vector_store(docs)
    r = get_redis_client()
    print("创建Redis连接成功......")
    print(r)
    print("全部初始化完成, 可以开始问答了......")
