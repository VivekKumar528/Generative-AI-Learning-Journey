
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()

# Model
model = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="groq"
)

# Schema
class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str


parser = PydanticOutputParser(pydantic_object=Movie)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    Extract movie information from the paragraph.
    {format_instructions}
    """),
    ("human", "{paragraph}")
])

# Streamlit UI
st.title("Movie Information Extractor")

para = st.text_area("Give your paragraph:")

if st.button("Extract Information"):

    if para:

        final_prompt = prompt.invoke({
            "paragraph": para,
            "format_instructions": parser.get_format_instructions()
        })

        response = model.invoke(final_prompt)

        movie_data = parser.parse(response.content)

        st.subheader("Extracted Movie Information")
        st.write(movie_data)

