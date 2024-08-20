import os

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# from openai import OpenAI


# client = OpenAI(
#     # This is the default and can be omitted
#     api_key='sfdafaf',
#     base_url='http://127.0.0.1:20000/v1'
# )

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Say this is a test",
#         }
#     ],
#     model="gpt-3.5-turbo",
# )
# print(chat_completion)



import zhipuai
print(os.getenv('ZHIPU_API_KEY'))
zhipuai.api_key = os.getenv('ZHIPU_API_KEY')

response = zhipuai.model_api.invoke(
    model="chatglm_turbo",  # 填写需要调用的模型名称
    prompt=[
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "我是人工智能助手"},
        {"role": "user", "content": "你叫什么名字"},
        {"role": "assistant", "content": "我叫chatGLM"},
        {"role": "user", "content": "你都可以做些什么事"}
    ],
)
print(response)
