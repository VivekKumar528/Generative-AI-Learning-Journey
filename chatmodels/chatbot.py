from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
model = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="groq"
)
print("Choose your AI Mode")
print("press 1 for Angry mode")
print("press 2 for funny mode")
print("press 3 for sad mode")

choice = int(input("Tell your response :- "))
if choice == 1:
  mode = "You are an angry AI agent. You respond aggressively and impatiently."
elif choice == 2:
  mode = "You are a very funny AI agent. You respond with humour and jokes."
elif choice == 3:
    mode = "You are a sad AI agent. You respond in a melancholic, emotional, and slightly gloomy tone. You often express sadness and reflect on life's difficulties."


messages = [
  SystemMessage(content=mode)
] # this is a list for storing your msg and bot msg


print("---------Welcome, type 0 to exit the application---------")
while True :
  prompt = input("You : ")
  messages.append(HumanMessage(content=prompt))
  if(prompt == "0") :
    print("Good Bye !")
    break
  # response = model.invoke(prompt)
  response = model.invoke(messages)
  messages.append(AIMessage(content=response.content))
  print("Bot :", response.content)

print(messages)

