from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


import json


# Load environment variables
load_dotenv()

#to do: upload core documents (GHG factors)
#


def process_text(text):
    # Adjust the chunk_size and chunk_overlap if necessary
    chunk_size = 3000  # increase chunk size for more context
    chunk_overlap = 400  # increase overlap to ensure continuity

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    
    return knowledgeBase


def main():
    st.title("Chat with your PDF 💬")
    
    # Load prebaked questions from JSON file
    with open('prebaked_questions.json') as f:
        prebaked_questions = json.load(f)
    
    pdf = st.file_uploader('Upload your PDF Document', type='pdf')
    
    # Dropdown menu for prebaked questions
    question_options = [q['display_text'] for q in prebaked_questions]
    selected_question = st.selectbox("Choose a prebaked question:", options=question_options)
    # Find the actual query text for the selected question
    actual_query_text = next((q['query_text'] for q in prebaked_questions if q['display_text'] == selected_question), None)
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        knowledgeBase = process_text(text)
        
        if actual_query_text:  # Use the actual query text for the model
            query = actual_query_text
        else:  # Or use the manual input if no prebaked question is selected
            query = st.text_input('Ask a question to the PDF')
        
        cancel_button = st.button('Cancel')
        
        if cancel_button:
            st.stop()
        
        if query:
            docs = knowledgeBase.similarity_search(query)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type='stuff')
            
            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)
                
            st.write(response)
            
if __name__ == "__main__":
    main()