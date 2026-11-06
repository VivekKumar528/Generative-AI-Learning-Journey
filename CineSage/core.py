from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="groq"
)


class Movie(BaseModel): # Schema
  title: str
  release_year : Optional[int]
  genre : List[str]
  director: Optional[str]
  cast: List[str]
  rating: Optional[float]
  summary: str

parser = PydanticOutputParser(pydantic_object=Movie)


prompt = ChatPromptTemplate.from_messages([
  ('system', """
   Extract movie information from the paragraph
   {format_instructions}
"""),
("human", "{paragraph}")]
)

para = input("Give your paragraph : ")

final_prompt = prompt.invoke(
  {"paragraph" : para,
   "format_instructions" : parser.get_format_instructions()
   }
)

response = model.invoke(final_prompt)
movie_data = parser.parse(response.content)

print(movie_data)