# -*- coding: utf-8 -*-

from transformers import AutoModel, AutoTokenizer

# 指定模型名称
model_name = "moka-ai/m3e-base"

# 下载并加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False, force_download=True)
model = AutoModel.from_pretrained(model_name, local_files_only=False, force_download=True)

# 保存到本地
local_model_path = "../_models"
tokenizer.save_pretrained(local_model_path)
model.save_pretrained(local_model_path)

print("模型和分词器已成功下载并保存到本地！")



if __name__ == '__main__':
    '''网络原因下载失败，最后通过 https://aistudio.baidu.com/datasetdetail/234251 下载'''