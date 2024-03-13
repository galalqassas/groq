import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
import os

load_dotenv()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Bot", page_icon="machine")
st.title("Galal Bot")


# get response from groq
def get_response(query):
    llm = ChatGroq(temperature=1.5,
     groq_api_key=os.getenv("groq_api_key"), # use your api key here
      model_name="llama2-70b-4096")
    memory = ConversationBufferWindowMemory(k=5)
    chain = ConversationChain(llm=llm, memory=memory, output_parser = StrOutputParser())

    return chain.invoke(query)['response']

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)


user_query = st.chat_input("How can I help you?")
if user_query != " " and user_query is not None:
    st.session_state.chat_history.append(HumanMessage(user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)
    with st.chat_message('AI'):
        response = get_response(user_query)
        st.markdown(response)
    st.session_state.chat_history.append(AIMessage(response))
