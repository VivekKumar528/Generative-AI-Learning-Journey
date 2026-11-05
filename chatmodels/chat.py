# from dotenv import load_dotenv

# load_dotenv()

# from langchain.chat_models import init_chat_model

# model = init_chat_model("gpt-4.1")

# response = model.invoke("What is cricket ?")

# print(response)

from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="groq"
)

response = model.invoke("What is cricket?")

print(response.content)

