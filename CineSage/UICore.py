
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

load_dotenv()

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Information Extractor",
    page_icon="🔍",
    layout="centered"
)

# ---------------- MODEL ----------------

model = init_chat_model(
    "openai/gpt-oss-20b",
    model_provider="groq"
)

# ---------------- PROMPT ----------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an intelligent information extraction assistant.

        Carefully read the given paragraph and extract useful information.

        Extract the following details whenever available:

        1. Title / Name
        2. Category
        3. Genre
        4. Director / Creator / Author
        5. Cast / Main People
        6. Release Year / Publication Year
        7. Language
        8. Main Topic
        9. Plot / Description
        10. Important Themes
        11. Music Composer
        12. Rating
        13. Critical Reception / Public Opinion
        14. Important Facts
        15. Quick Summary

        Instructions:
        - Extract only information available in the paragraph.
        - Do not make up facts.
        - If information is missing, write "Not mentioned".
        - If a field is not relevant, write "Not applicable".
        - Keep the extracted information concise and readable.
        - The summary should be 2-3 sentences.
        - Identify the content category based on the paragraph.
        """
    ),
    (
        "human",
        "Extract useful information from this paragraph:\n\n{paragraph}"
    )
])

# ---------------- UI ----------------

st.title("🔍 Information Extractor")

st.write(
    "Extract useful information and generate a quick summary "
    "from any paragraph using AI."
)

st.divider()

paragraph = st.text_area(
    "Enter your paragraph",
    placeholder="Paste your paragraph here...",
    height=250
)

if st.button("Extract Information", use_container_width=True):

    if not paragraph.strip():
        st.warning("Please enter a paragraph first.")

    else:
        with st.spinner("Extracting information..."):

            try:
                final_prompt = prompt.invoke({
                    "paragraph": paragraph
                })

                response = model.invoke(final_prompt)

                st.subheader("Extracted Information")
                st.markdown(response.content)

            except Exception as e:
                st.error(f"Something went wrong: {e}")
