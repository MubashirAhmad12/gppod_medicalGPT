
import openai
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from utils_streamlit import SalesGPT
from utils_streamlit import conversation_stages
import random
openai.api_key  = st.secrets["OPENAI_API_KEY"]

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

verbose=True
llm = ChatOpenAI(temperature=0.3)
config = dict(
physician_name = "Ben",
company_name="InnoTech",
conversation_history=[],
conversation_stage = conversation_stages.get('1', "Patient Profile: Start the conversation by introducing yourself as a General Physician. Ask the patient about the Name, Age, Gender and Occupation. Do not move to the next stage until the patient provides Age, Gender and Occupation. If you do not find Age, Gender and Occupation of patient in the conversation history, Move the conversation to this stage and remain in this stage until Age, Gender and occupation information is provided"),
use_tools=False
)

def initialize_chain(human_input):
    if human_input!="":
        with st.spinner('Generating response...'):
            if st.session_state['count']==0:
                sales_agent = SalesGPT.from_llm(llm, verbose=False, **config)
                sales_agent.seed_agent() 
                st.session_state['sales_ag'] = sales_agent
                st.session_state['count'] = st.session_state['count'] +1
                # print(st.session_state['count'])
            else:
                sales_agent = st.session_state['sales_ag']
                # print(sales_agent)
            if human_input:
                sales_agent.human_step(human_input)
            # print(sales_agent.determine_conversation_stage())
            ai_response = sales_agent.step()
            ai_response =ai_response.replace("<END_OF_TURN>","").replace("<END_OF_CALL>","").replace("Ben: ","").replace('"',"")
            st.session_state['human'].append(human_input)
            st.session_state['ai'].append(ai_response)


if 'count' not in st.session_state:
    st.session_state['count'] = 0
if 'ai' not in st.session_state:
    st.session_state['ai'] = []
if 'human' not in st.session_state:
    st.session_state['human'] = []
if 'sales_ag' not in st.session_state:
    st.session_state['sales_ag'] = None
if 'user_msg' not in st.session_state:
    st.session_state['user_msg'] = None

def end_click():
    # st.session_state['prompts'] = [{"role": "system", "content": "You are a helpful assistant. Answer as concisely as possible with a little humor expression."}]
    st.session_state['count'] = 0
    st.session_state['ai'] = []
    st.session_state['human'] = []
    st.session_state['sales_ag'] = None

def get_text():
    """
    Get the user input text.

    Returns:
        (str): The text entered by the user
    """
  
    input_text = st.text_input(
        "You: ",
        # st.session_state['user'],
        key="user",
        placeholder="Your GP_Pod is here! Ask me anything ...",
        # label_visibility="hidden",
    )
    # print("hello",input_text)
    return input_text


    


st.title("Virtual General Physician👨‍⚕️🩺👩‍⚕️")
st.markdown("Get expert medical advice and diagnosis from the comfort with our Virtual General Physician service.")

user_input = get_text()
def clear_text():
    st.session_state["user"] = ""

col1, col2 = st.columns(2)
st.button("Clear text input", on_click=clear_text)


with col1:
    new_button = st.sidebar.button("New Chat", on_click=end_click, use_container_width=True)
with col2:
    end_button = st.sidebar.button("Clear History", on_click=end_click, use_container_width=True)

if user_input:
    initialize_chain(user_input)
if st.session_state['ai']:
    for i in range(len(st.session_state['ai'])-1, -1, -1):
        message(st.session_state['ai'][i], key=str(i))
        message(st.session_state['human'][i],
                is_user=True, key=str(i) + '_user')

