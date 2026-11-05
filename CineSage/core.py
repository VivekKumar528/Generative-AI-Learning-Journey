from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="groq"
)
prompt = ChatPromptTemplate.from_messages([
  ("system",
    """
You are an intelligent information extraction assistant.

Your task is to carefully read the given paragraph and extract useful information.

Extract the following details whenever available:

1. Title / Name
2. Category (Movie, Book, Person, Company, Technology, etc.)
3. Genre
4. Director / Creator / Author
5. Cast / Main People
6. Release Year / Publication Year
7. Language
8. Main Topic
9. Plot / Description
10. Important Themes
11. Music Composer (if applicable)
12. Rating (if available)
13. Critical Reception / Public Opinion
14. Important Facts
15. Quick Summary

Instructions:
- Extract only information available in the paragraph.
- Do not make up facts.
- If information is missing, write "Not mentioned".
- Keep the extracted information concise and readable.
- The summary should be 2-3 sentences.
- Identify the content category based on the paragraph.
- If a field is not relevant to the content, write "Not applicable".

Paragraph:
{paragraph}

Extracted Information:
"""
  ), (
    "human",
    '''Extract information from this paragraph:
    {paragraph}
    '''
  )
])

para = input("Give your paragraph : ")

final_prompt = prompt.invoke(
  {"paragraph" : para}
)

response = model.invoke(final_prompt)

print(response.content)