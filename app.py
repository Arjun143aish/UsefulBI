import streamlit as st
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import openai


# Sidebar contents
with st.sidebar:
    st.title('UBI ChatBot🤖')
    st.markdown('''
    ## About
    This app enables users to upload their files (PDF,CSV,TXT) and get back the responses:
    - [UsefulBI -About Us](https://usefulbi.com/about-us/)
    - [Linkedin](https://www.linkedin.com/company/usefulbi-corporation/mycompany/verification/)
    - [Facebook](https://www.facebook.com/UsefulbiCorporation/)
    - [Twitter](https://twitter.com/usefulbicorp)
 
    ''')
    add_vertical_space(2)
    st.write('')

openai.api_key = os.environ.get("OPENAI_API_KEY")

def get_user_access_level():
    access_level = st.radio("Select your role:", ("Admin", "Non-Admin"))
    return access_level


def main():
    st.header("Chat with One or more file's 💬")    
    
    user_access_level = get_user_access_level()

    if user_access_level == "Admin":
        # Admin can upload or update path
        default_folder = st.text_input("Enter default folder path:", "C:\\Users\\DLP-I516-29\\Documents\\Useful BI\\UseFulBI\\2023\\Data Engineering\\UBI_Gen_Next\\PDF's")
    else:
        # Non-admin users see the selected path but cannot update it
        default_folder = "C:\\Users\\DLP-I516-29\\Documents\\Useful BI\\UseFulBI\\2023\\Data Engineering\\UBI_Gen_Next\\PDF's"
        st.text("Selected Path (Non-admin):")
        st.write(default_folder)

    
    files_in_folder = os.listdir(default_folder)
    total_documents = len(files_in_folder)
    st.write(f"Total Documents Available: {total_documents}")

    selected_documents = st.multiselect("Select documents to chat with:",["Select All"] +  files_in_folder)

    if "Select All" in selected_documents:
        selected_documents = files_in_folder
        st.markdown(f"**You've selected: All Documents ({len(selected_documents)} documents)**")
    else:
        st.markdown(f"**You've selected: {', '.join(selected_documents)} ({len(selected_documents)} documents)**")
        
    for file_item in selected_documents:
        file_extension = file_item.split('.')[-1].lower()
    
        if file_extension in ['csv','txt','pdf']:
            file_path = os.path.join(default_folder,file_item)

            text = ""
            if file_extension == 'pdf':
                pdf_reader = PdfReader(file_path)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            else:
                with open(file_path,'r',encoding='utf-8') as file_obj:
                    text = file_obj.read()
#    
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 10000,
                chunk_overlap = 200,
                length_function = len
                )
            chunks = text_splitter.split_text(text=text)

        #embeddings
    
            store_name = file_item[:-4]

            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl",'rb') as f:
                    VectorStore = pickle.load(f)
            #st.write('Embeddings loaded from disk')
            else:
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
                with open(f"{store_name}.pkl",'wb')as f:
                    pickle.dump(VectorStore,f)

            #Get User questions/query
            query = st.text_input(f"What's your query about the {file_item}?")

            if query:
                docs = VectorStore.similarity_search(query=query,k=3)
                llm = OpenAI(model_name='gpt-3.5-turbo-0613',temperature=0)
                chain = load_qa_chain(llm =llm,chain_type='stuff')
                with get_openai_callback()as cb:
                    response = chain.run(input_documents = docs,question = query)
                st.write(f"Results for {file_item}:")
                st.write(response)

    
            #st.write(docs)


                #st.write('Embeddings Computation loaded')

            
            
        #st.write(chunks)

        #st.write(text)

if __name__ == '__main__':
    main()
