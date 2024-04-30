import openai
import streamlit as st
import json
import io
import os
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
import pandas as pd
import base64
import traceback
import xlsxwriter
from langchain_text_splitters import RecursiveCharacterTextSplitter



# insert css
with open("design.css") as source_des:
    st.markdown(f"<style>{source_des.read()}<style>", unsafe_allow_html=True)


load_dotenv()

mykey = os.getenv("OPENAI_API_KEY")

# OpenAI API

client = OpenAI(api_key=mykey)
# openai.api_key = mykey
# openai.api_key = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=openai.api_key)


def extract_facts(text):
    prompt = f'''
    You are an experienced HR with experience in the field of Data Science or Full Stack Web Development or
    DevOps or Big Data Engineering or Data Analyst or Business Analyst or QA Tester.

    Please extract the following information from the uploaded pdf and return it as a JSON object:
    1. Name of the candidate
    2. Qualification (most recent education)
    3. Passout Year
    4. Mobile Number
    5. Email ID
    4. Skills

    Once the pdf files are uploaded, go through each pdf file one by one and generate the mentioned information.
    generate output for all pdf resume files, dont even skip a single pdf,
    also, dont repeat output of any pdf file once generated.

    This is the body of text to extract the information from:
    {text}
    ''' 

    custom_function = [
        {
            'name': 'facts_extraction',
            'description': 'Extract the facts from the body of the input text',
            'parameters':{
                'type':'object',
                'properties':{
                    'Name': {
                        'type':'string',
                        'description':'Name candidate'
                    },
                    'Passout Year': {
                        'type':'string',
                        'description':'Passout Year'
                    },
                    'Qualification': {
                        'type':'string',
                        'description':'Qualification'
                    },
                    'Mobile Number': {
                        'type':'string',
                        'description':'Mobile Number'
                    },
                    'Email ID': {
                        'type':'string',
                        'description':'Email ID'
                    },
                    'Skills': {
                        'type':'string',
                        'description':'Skills'
                    }
                }
            }
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            functions=custom_function
        )

        output = response.choices[0].message.function_call.arguments
        result = json.loads(output)
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {e}")
        st.error(traceback.format_exc())
        result = None

    return result


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n'],
        chunk_size=1600,  # Adjust the chunk size as needed
        chunk_overlap=0,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def main():
    st.title("Resume Scanner")
    st.markdown('----', unsafe_allow_html=True)

    with st.sidebar:
        # File uploader widget for user to upload PDF documents
        uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    # # File uploader widget for user to upload PDF documents
    # uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        extracted_facts = []  # List to store extracted facts for all files
        for uploaded_file in uploaded_files:
            # st.write(f"### {uploaded_file.name}")
            try:
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text()
                
                chunks = get_text_chunks(text)
                # Extract facts for the current document
                facts = extract_facts(chunks)
                if facts:
                    extracted_facts.append(facts)
                # st.write(extracted_facts)
            except Exception as e:
                st.error(f"An error occurred while processing the PDF '{uploaded_file.name}': {e}")
                st.error(traceback.format_exc())
        # st.write(extracted_facts)
            # for key, value in facts.items():
            #     st.write(f"{key}: {value}")
            # st.write("")
            # html_content = '<hr class="dashed-line">'
            # st.markdown(html_content, unsafe_allow_html=True)
            

        if extracted_facts:
            # Convert extracted facts to DataFrame
            df = pd.DataFrame(extracted_facts)

            # Display the DataFrame
            # st.subheader("Extracted Facts:")
            st.write(df)

            # Display key-value pairs
            for extracted_fact in extracted_facts:
                for key, value in extracted_fact.items():
                    st.write(f"{key}: {value}")
                st.write("")
                html_content = '<hr class="dashed-line">'
                st.markdown(html_content, unsafe_allow_html=True)


        with st.sidebar:
            # Button to download the DataFrame as an Excel file
            st.markdown(get_table_download_link(df), unsafe_allow_html=True)




def get_table_download_link(df):
    # Generate a link to download the DataFrame as an Excel file
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False, engine="xlsxwriter")
    excel_buffer.seek(0)
    b64 = base64.b64encode(excel_buffer.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="extracted_facts.xlsx">Download Excel File</a>'
    return href


if __name__ == "__main__":
    main()
