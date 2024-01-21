import chromadb
import os
import sys

import uuid
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
# client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chromadb"))
client = chromadb.Client()

# Create collection. get_collection, get_or_create_collection, delete_collection also available!
collection = client.create_collection("all-my-documents")

# Add docs to the collection. Can also update and delete. Row-based API coming soon!
collection.add(
    documents=["This is document1", "This is google", "This is a query document"], # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
    metadatas=[{"source": "notion"}, {"source": "google-docs"}, {'source': 'test'}], # filter on these!
    ids=["doc1", "doc2", 'doc3'], # unique for each doc
)

#collection.delete(where={"source": "notion"})

# Query/search 2 most similar results. You can also .get by id
results = collection.query(
    query_texts=["This is a query document"],
    n_results=1,
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)

print(results)





# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from transformers import AutoTokenizer
# from langchain.docstore.document import Document
# d1 = Document(page_content="""# 分布式训练技术原理
# - 数据并行
#     - FSDP
#         - FSDP算法是由来自DeepSpeed的ZeroRedundancyOptimizer技术驱动的，但经过修改的设计和实现与PyTorch的其他组件保持一致。FSDP将模型实例分解为更小的单元，然后将每个单元内的所有参数扁平化和分片。分片参数在计算前按需通信和恢复，计算结束后立即丢弃。这种方法确保FSDP每次只需要实现一个单元的参数，这大大降低了峰值内存消耗。(数据并行+Parameter切分)
#     - DDP
#         - DistributedDataParallel (DDP)， **在每个设备上维护一个模型副本，并通过向后传递的集体AllReduce操作同步梯度，从而确保在训练期间跨副本的模型一致性** 。为了加快训练速度， **DDP将梯度通信与向后计算重叠** ，促进在不同资源上并发执行工作负载。
#     - ZeRO
#         - Model state
#             - Optimizer->ZeRO1
#                 - 将optimizer state分成若干份，每块GPU上各自维护一份
#                 - 每块GPU上存一份完整的参数W,做完一轮foward和backward后，各得一份梯度,对梯度做一次 **AllReduce（reduce-scatter + all-gather）** ， **得到完整的梯度G,由于每块GPU上只保管部分optimizer states，因此只能将相应的W进行更新,对W做一次All-Gather**
#             - Gradient+Optimzer->ZeRO2
#                 - 每个GPU维护一块梯度
#                 - 每块GPU上存一份完整的参数W,做完一轮foward和backward后， **算得一份完整的梯度,对梯度做一次Reduce-Scatter，保证每个GPU上所维持的那块梯度是聚合梯度,每块GPU用自己对应的O和G去更新相应的W。更新完毕后，每块GPU维持了一块更新完毕的W。同理，对W做一次All-Gather，将别的GPU算好的W同步到自己这来**""")


# chunk_size=250
# chunk_overlap=50
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=chunk_size,
#     chunk_overlap=chunk_overlap
# )



# MODEL_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "_models")

# # 选用的 Embedding 名称
# EMBEDDING_MODEL = "m3e-base"

# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# hugging_face_embeddings = HuggingFaceEmbeddings(model_name=f'{MODEL_ROOT_PATH}/moka-ai/{EMBEDDING_MODEL}')


# for doc in text_splitter.split_documents([d1]):
#     collection.add(
#         ids=str(uuid.uuid1()),
#         documents=doc.page_content,
#         metadatas={"source": "d"},
#         embeddings=hugging_face_embeddings.embed_documents([doc.page_content])[0]
#     )


# print(collection.query(query_embeddings=hugging_face_embeddings.embed_query("RAG增强"), n_results=1))
# print(collection.query(query_embeddings=hugging_face_embeddings.embed_query("FSDP算法是由来自DeepSpeed的ZeroRedundancyOptimizer技术驱动的，但经过修改的设计和实现与PyTorch的其他组件保持一致。FSDP将模型实例分解为更小的单元，然后将每个单元内的所有参数扁平化和分片。分片参数在计算前按需通信和恢复，计算结束后立即丢弃。这种方法确保FSDP每次只需要实现一个单元的参数，这大大降低了峰值内存消耗。"), n_results=1))

