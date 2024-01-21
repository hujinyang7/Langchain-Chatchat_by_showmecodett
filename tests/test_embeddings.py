import os

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)

model_path = os.path.join(os.path.dirname(os.path.dirname(current_file_path)), '_models/moka-ai/m3e-base')

print('SentenceTransformer')
from sentence_transformers import SentenceTransformer

print('time to load model')

model = SentenceTransformer(model_path)

print('time to encode')
#Our sentences we like to encode
sentences = [
    '* Moka 此文本嵌入模型由 MokaAI 训练并开源，训练脚本使用 uniem',
    '* Massive 此文本嵌入模型通过**千万级**的中文句对数据集进行训练',
    '* Mixed 此文本嵌入模型支持中英双语的同质文本相似度计算，异质文本检索等功能，未来还会支持代码检索，ALL in one'
]


#Sentences are encoded by calling model.encode()
sentence_transformer_embeddings = model.encode(sentences)





print('HuggingFaceEmbeddings')
# test huggingface embeddings
MODEL_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "_models")

# 选用的 Embedding 名称
EMBEDDING_MODEL = "m3e-base"

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
hugging_face_embeddings = HuggingFaceEmbeddings(model_name=f'{MODEL_ROOT_PATH}/moka-ai/{EMBEDDING_MODEL}')


def compare_float_arrays(arr1, arr2):
    if len(arr1) != len(arr2):
        return False

    for a, b in zip(arr1, arr2):
        if round(a, 2) != round(b, 2):
            return False

    return True


for i in range(len(sentences)):
    print("Sentence:", sentences[i])
    print("SentenceTransformer:", sentence_transformer_embeddings[i][:5])
    print('is_same', compare_float_arrays(sentence_transformer_embeddings[i], hugging_face_embeddings.embed_query(sentences[i])))
    print("")

print('success')
