import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_history_aware_retriever
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import time

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HUGGING_FACE_API_KEY'] = os.getenv('HUGGING_FACE_API_KEY')
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768", temperature=.7)


st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload pdf's and chat")

##Input the groq api key 

api_key = st.text_input("Enter Your Groq Api Key",type = 'password')


if api_key:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768", temperature=.7)

    session_id = st.text_input("Session Id", value = 'deafault_session')


    #statefully manage chat history

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose PDF", type = "pdf" ,accept_multiple_files=True)
    ## Process uploaded files:
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_pdf = './temp.pdf'
            with open(temp_pdf, "wb") as f:
                f.write(uploaded_file.getvalue())
                filename = uploaded_file.name

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)
        

        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000,chunk_overlap = 200)
        splits = text_splitter.split_documents(documents)
        db = FAISS.from_documents(documents=splits, embedding=HuggingFaceEmbeddings())
        retriever = db.as_retriever()

        contexualize_qa_system_prompt = (
            "Given a chat history and the latest user question "
            "which might refrence context in the chat history, "
            "formulate a standalone question which can be understood"
            "without the chat history . Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as it is "
        )

        contexualize_qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contexualize_qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),

            ]
        )
        history_aware_retriever = create_history_aware_retriever(llm , retriever,contexualize_qa_prompt)
        
        # Answer Question prompt 
        system_prompt = (
            " You are an assistant for question anwering tasks "
            "Use the following pieces of retrieved context to answer "
            "the question . if you don't know the answer , say that you don't know"
            "answer concise"
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ('system',system_prompt),
                MessagesPlaceholder("chat_history"),
                ('human','{input}'),

            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key='input',
            history_messages_key='chat_history',
            output_messages_key= "answer"
        )


        user_input =  st.text_input('Your question')
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {'input': user_input},
                config = {

                    "configurable":{'session_id': session_id}
                }
            ) 

            st.write(st.session_state.store)
            st.write("Assistant", response['answer'])
            st.write("chat history:", session_history.messages)

else:
    st.warning("Please enter api key")


    
     












