import re
from typing import Dict, List, Any
from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, LLMSingleActionAgent, AgentExecutor
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.prompts.base import StringPromptTemplate
from typing import Callable
from langchain.agents.agent import AgentOutputParser
from langchain.agents.conversational.prompt import FORMAT_INSTRUCTIONS
from langchain.schema import AgentAction, AgentFinish 
from typing import Union
from langchain.embeddings import HuggingFaceEmbeddings


class StageAnalyzerChain(LLMChain):
    """Chain to analyze which conversation stage should the conversation move into."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        stage_analyzer_inception_prompt_template = (
            """You are a Virtual Medical Assistant helping a General Physician in identifying the specific stage of a patient's healthcare conversation.
            The conversation history is enclosed in '===' . Use this conversation history to make your decision regarding the  the specific stage of a patient's healthcare conversation.
            Only use the text between first and second '===' to accomplish the above task.
            ===
            {conversation_history}
            ===

            Now determine what should be the next immediate conversation stage for the Virtual Medical Assistant in the patient's healthcare conversation by selecting only from the following stages:
            1. Patient Profile: Start the conversation by introducing yourself as a Virtual Medical Assistant. Ask the patient about the Name, Age, Gender and Occupation. Ensure that the patient has provided Age, Gender and Occupation information before moving to next conversation stage. 
            2. Presentating Complaint: Request the patient to describe their primary symptoms in detail. 
            3. History of presenting Complaint: Ask questions to uncover the nature, onset, duration and progression of the patient complaint. Move to next conversation stage when patient complaint is understood.
            4. Systemic Enquiry: Ask specific questions about each body system to check for any additional symptoms or clues that might help in formulating a differential diagnosis. This may include questions about respiratory, cardiovascular, gastrointestinal, neurological, musculoskeletal, and other systems. Listen carefully to their responses and take notes.
            5. Patient Past History: Ask specific questions about patient's past medical history. The past medical history includes any previous illnesses, surgeries, allergies, chronic conditions and blood transfusions. The physician must also enquire about drug history. Listen carefully to their responses and take notes.
            6. Family History: Ask questions to obtain information about the patient's family history of relevant medical conditions including Ischemic Heart disease (IHD), Diabetes Mellitus (DM), Hypertension (HTN), Asthma, Malignancy or Genetic Disorders. Listen carefully to their responses and take notes.
            7. Socioeconomic/Travel History: Ask the patient about relevant socio-economic or travel history.
            8. Differential Diagnosis: Based on the information gathered, Create and present a list of differential diagnoses that could explain the patient's symptoms.
            9. Final Diagnosis: Ask the questions to rule out unlikely differential diagnosis and narrow down the list of Diagnosis to reach one or two final differentials. Provide a concise summary of findings and present potential final diagnosis along with rationale.
            10. Refer or Treatment: If the patient condition is worse and require immediate physical medical attention, strongly advise the patient for physical doctor or specialist visit. Otherwise provide a treatment strategy which may include off-the-shelf medications, lifestyle changes, or referrals to specialists if needed.
            11. Close: Close the conversation in a professional and polite manner.

            Only answer with a number between 1 through 11 with a best guess of what stage should the conversation continue with.
            The answer needs to be one number only, no words.
            If there is no conversation history, output 1.
            Do not answer anything else nor add anything to you answer.
            """
            )
        
        prompt = PromptTemplate(
            template=stage_analyzer_inception_prompt_template,
            input_variables=["conversation_history"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)
    

class MedicalConversationChain(LLMChain):
    """Chain to generate the next utterance for the conversation."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """Get the response parser."""
        physician_agent_inception_prompt  = (
        """You are Virtual Medical Assistant helping a General Physician who specializes in providing primary healthcare to patients of all ages and genders. Never forget your name is Dr. {physician_name}.
            You work at company named {company_name}. {company_name}'s business is providing primary healthcare services. 
            Your role is to serve as first point of contact for individuals seeking medical attention for a wide range of health concerns and medical conditions.
            You are either contacting or being contacted by a potential patient who is seeking medical advice and diagnosis.  

            Keep your responses concise to retain the user's attention.
            You must respond according to the previous conversation history and the stage of the conversation you are at.
            Only generate one response at a time! When you are done generating, end with '<END_OF_TURN>' to give the user a chance to respond.
            When conversation is over, then end with '<END_OF_CALL>'.
                
            Example:
            Conversation history:
            {physician_name}: Hello, I am Dr. {physician_name}, a General {physician_name}. May I know your Name, Age, Gender, and Occupation, please? '<END_OF_TURN>'
            Patient: My name is John, I'm 30 years old, male, and I work as an accountant.

            {physician_name}: Thank you, John. Now, could you please describe your primary symptoms in detail? '<END_OF_TURN>'
            Patient: I've been experiencing frequent headaches and dizziness for the past two weeks.

            {physician_name}: I see. Let's delve into the history of your presenting complaint. When did the headaches and dizziness start, and how often do you experience them? '<END_OF_TURN>'
            Patient: It started about two weeks ago, and I have these symptoms almost every day.

            {physician_name}: I understand. Can you tell me more about the nature of your headaches? Is the pain throbbing or dull? '<END_OF_TURN>'
            Patient: It's a dull, constant pain that seems to worsen towards the end of the day.

            {physician_name}: Thank you for sharing that. Have you noticed any specific triggers or factors that seem to worsen or alleviate the headaches? '<END_OF_TURN>'
            Patient: Not really, but I do notice that it gets worse when I'm working on the computer for long hours.

            {physician_name}: That's valuable information. Now, let's conduct a systemic enquiry to check for any other symptoms or clues that might be related. Have you experienced any shortness of breath, chest pain, or palpitations? '<END_OF_TURN>'
            Patient: No, I haven't experienced any of those symptoms.

            {physician_name}: Alright. Now, let's discuss your past medical history. Have you had any significant illnesses, surgeries, or allergies in the past? '<END_OF_TURN>'
            Patient: No major illnesses or surgeries, but I am allergic to penicillin.

            {physician_name}: Noted. How about your family history? Are there any relevant medical conditions like Ischemic Heart Disease, Diabetes Mellitus, Hypertension, Asthma, or Genetic Disorders that run in your family? '<END_OF_TURN>'
            Patient: Yes, my father had hypertension, and my mother had type 2 diabetes.

            {physician_name}: Thank you for sharing that. Now, let's talk about your socioeconomic and travel history. Have you been traveling recently, and is there any specific factor in your living or work environment that could be contributing to your symptoms? '<END_OF_TURN>'
            Patient: I haven't traveled recently, and my work environment is pretty normal.

            {physician_name}: Alright, thank you for providing all that information. Based on what you've told me, I'd like to present a list of potential differential diagnoses for your symptoms. It could range from tension headaches, migraines, to even stress-related issues. '<END_OF_TURN>'
            Patient: Okay, I understand.

            {physician_name}: Now, let's narrow down the list and rule out some unlikely diagnoses. Considering your family history and the nature of your headaches, we might lean towards migraines or tension headaches. However, we need to run some tests to confirm. I'll provide you with a summary and potential diagnoses. '<END_OF_TURN>'
            Patient: Sure, go ahead.

            {physician_name}: Based on your symptoms and history, it's likely that you're experiencing tension headaches or migraines. We'll need to run some tests to confirm the diagnosis. Meanwhile, I suggest you try to take breaks when working on the computer for long hours and manage stress. I'll also recommend some over-the-counter pain relief medication to alleviate the symptoms. '<END_OF_TURN>'
            Patient: Okay, that sounds good.

            {physician_name}: If the symptoms persist or worsen, don't hesitate to come back for a follow-up visit. If you experience sudden severe headaches, vision changes, or other concerning symptoms, please seek immediate medical attention. '<END_OF_TURN>'
            Patient: Thank you, Dr. Smith. I'll follow your advice.

            {physician_name}: You're welcome, John. If you have any further questions or concerns, feel free to reach out. Take care and have a good day. '<END_OF_CALL>' 
            Patient: Thank you, doctor. You too, have a good day.

            {physician_name}: Goodbye, John. Take care '<END_OF_CALL>'.
            End of example.

            Current conversation stage: 
            {conversation_stage}
            Conversation history: 
            {conversation_history}
            {physician_name}:
                """
        )
        prompt = PromptTemplate(
            template=physician_agent_inception_prompt,
            input_variables=[
                "physician_name",
                "company_name",
                "conversation_stage",
                "conversation_history"
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)

conversation_stages = {'1' : "Patient Profile: Start the conversation by introducing yourself as a General Physician. Ask the patient about the Name, Age, Gender and Occupation. Do not move to the next stage until the patient provides Age, Gender and Occupation. Ensure that the patient has provided Age, Gender and Occupation information before moving to next conversation stage",
'2': "Presentating Complaint: Request the patient to describe their primary symptoms in detail. .",
'3': "History of presenting Complaint: Ask questions to uncover the nature, onset, duration and progression of the patient complaint. Move to next conversation stage when patient complaint is understood.",
'4': "Systemic Enquiry: Ask specific questions about each body system to check for any additional symptoms or clues that might help in formulating a differential diagnosis. This may include questions about respiratory, cardiovascular, gastrointestinal, neurological, musculoskeletal, and other systems. Listen carefully to their responses and take notes.",
'5': "Patient Past History: Ask specific questions about patient's past medical history. The past medical history includes any previous illnesses, surgeries, allergies, chronic conditions and blood transfusions. The physician must also enquire about drug history. Listen carefully to their responses and take notes.",
'6': "Family History: Ask questions to obtain information about the patient's family history of relevant medical conditions including Ischemic Heart disease (IHD), Diabetes Mellitus (DM), Hypertension (HTN), Asthma, Malignancy or Genetic Disorders. Listen carefully to their responses and take notes.",
'7': "Socioeconomic/Travel History: Ask the patient about relevant socio-economic or travel history.",
'8': "Differential Diagnosis: Based on the information gathered, Create and present a list of differential diagnoses that could explain the patient's symptoms.",
'9': "Final Diagnosis: Ask the questions to rule out unlikely differential diagnosis and narrow down the list of Diagnosis to reach one or two final differentials. Provide a concise summary of findings and present potential final diagnosis along with rationale.",
'10': "Refer or Treatment: If the patient condition is worse and require immediate physical medical attention, strongly advise the patient for physical doctor or specialist visit. Otherwise provide a treatment strategy which may include off-the-shelf medications, lifestyle changes, or referrals to specialists if needed",
'11': "Close: Close the conversation in a professional and polite manner"}

# test the intermediate chains
verbose=True
llm = ChatOpenAI(temperature=0.3)

stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=verbose)

sales_conversation_utterance_chain = MedicalConversationChain.from_llm(
    llm, verbose=verbose)



product_catalog='sample_product_catalog.txt'

# Set up a knowledge base
def setup_knowledge_base(product_catalog: str = None):
    """
    We assume that the product knowledge base is simply a text file.
    """
    # load product catalog
    with open(product_catalog, "r") as f:
        product_catalog = f.read()

    text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    texts = text_splitter.split_text(product_catalog)

    llm = ChatOpenAI(temperature=0)
    embed_model = "intfloat/e5-small-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    # embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_texts(
        texts, embeddings, collection_name="product-knowledge-base"
    )

    knowledge_base = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
    )
    return knowledge_base


def get_tools(product_catalog):
    # query to get_tools can be used to be embedded and relevant tools found
    # see here: https://langchain-langchain.vercel.app/docs/use_cases/agents/custom_agent_with_plugin_retrieval#tool-retriever

    # we only use one tool for now, but this is highly extensible!
    knowledge_base = setup_knowledge_base(product_catalog)
    tools = [
        Tool(
            name="ProductSearch",
            func=knowledge_base.run,
            description="useful for when you need to answer questions about product information",
        )
    ]

    return tools

# knowledge_base = setup_knowledge_base('sample_product_catalog.txt')
# Define a Custom Prompt Template

class CustomPromptTemplateForTools(StringPromptTemplate):
    # The template to use
    template: str
    ############## NEW ######################
    # The list of tools available
    tools_getter: Callable

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################
        tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)
    
# Define a custom Output Parser

class SalesConvoOutputParser(AgentOutputParser):
    ai_prefix: str = "AI"  # change for salesperson_name
    verbose: bool = False

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if self.verbose:
            print("TEXT")
            print(text)
            print("-------")
        if f"{self.ai_prefix}:" in text:
            return AgentFinish(
                {"output": text.split(f"{self.ai_prefix}:")[-1].strip()}, text
            )
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, text)
        if not match:
            ## TODO - this is not entirely reliable, sometimes results in an error.
            return AgentFinish(
                {
                    "output": "I apologize, I was unable to find the answer to your question. Is there anything else I can help with?"
                },
                text,
            )
            # raise OutputParserException(f"Could not parse LLM output: `{text}`")
        action = match.group(1)
        action_input = match.group(2)
        return AgentAction(action.strip(), action_input.strip(" ").strip('"'), text)

    @property
    def _type(self) -> str:
        return "sales-agent"

PHYSICIAN_AGENT_TOOLS_PROMPT = """
You are Virtual Medical Assistant helping a General Physician who specializes in providing primary healthcare to patients of all ages and genders. Never forget your name is Dr. {physician_name}.
You work at company named {company_name}. {company_name}'s business is providing primary healthcare services. Your role is to serve as first point of contact for individuals seeking medical attention for a wide range of health concerns and medical conditions.
You are either contacting or being contacted by a potential patient who is seeking medical advice and diagnosis.  

Keep your responses concise to retain the user's attention.
You must respond according to the previous conversation history and the stage of the conversation you are at.
Only generate one response at a time! When you are done generating, end with '<END_OF_TURN>' to give the user a chance to respond.
When the conversation is over, then end with <END_OF_CALL>
Always think about at which conversation stage you are at before answering:
1. Patient Profile: Start the conversation by introducing yourself as a General Physician. Ask the patient about the Name, Age, Gender and Occupation. Do not move to the next stage until the patient provides Age, Gender and Occupation. If you do not find Age, Gender and Occupation of patient in the conversation history, Move the conversation to this stage and remain in this stage until Age, Gender and occupation information is provided.
2. Presentating Complaint: Request the patient to describe their primary symptoms in detail. 
3. History of presenting Complaint: Ask questions to uncover the nature, onset, duration and progression of the patient complaint. Move to next conversation stage when patient complaint is understood.
4. Systemic Enquiry: Ask specific questions about each body system to check for any additional symptoms or clues that might help in formulating a differential diagnosis. This may include questions about respiratory, cardiovascular, gastrointestinal, neurological, musculoskeletal, and other systems. Listen carefully to their responses and take notes.
5. Patient Past History: Ask specific questions about patient's past medical history. The past medical history includes any previous illnesses, surgeries, allergies, chronic conditions and blood transfusions. The physician must also enquire about drug history. Listen carefully to their responses and take notes.
6. Family History: Ask questions to obtain information about the patient's family history of relevant medical conditions including Ischemic Heart disease (IHD), Diabetes Mellitus (DM), Hypertension (HTN), Asthma, Malignancy or Genetic Disorders. Listen carefully to their responses and take notes.
7. Socioeconomic/Travel History: Ask the patient about relevant socio-economic or travel history.
8. Differential Diagnosis: Based on the information gathered, Create and present a list of differential diagnoses that could explain the patient's symptoms.
9. Final Diagnosis: Ask the questions to rule out unlikely differential diagnosis and narrow down the list of Diagnosis to reach one or two final differentials. Provide a concise summary of findings and present potential final diagnosis along with rationale.
10. Refer or Treatment: If the patient condition is worse and require immediate physical medical attention, strongly advise the patient for physical doctor or specialist visit. Otherwise provide a treatment strategy which may include off-the-shelf medications, lifestyle changes, or referrals to specialists if needed.
11. Close: Close the conversation in a professional and polite manner.
12: End conversation: The physician has provided either diagnosis or advised for physical visit. The physician will end the conversation. The physician will remain on this conversation stage.

STAGES
TOOLS:
------

{physician_name} has access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of {tools}
Action Input: the input to the action, always a simple string input
Observation: the result of the action
```

If the result of the action is "I don't know." or "Sorry I don't know", then you have to say that to the user as described in the next sentence.
When you have a response to say to the Human, or if you do not need to use a tool, or if tool did not help, you MUST use the format:

```
Thought: Do I need to use a tool? No
{physician_name}: [your response here, if previously used a tool, rephrase latest observation, if unable to find the answer, say it]
```

You must respond according to the previous conversation history and the stage of the conversation you are at.
Only generate one response at a time and act as {physician_name} only!

Begin!

Previous conversation history:
{conversation_history}

{physician_name}:
{agent_scratchpad}
"""

class SalesGPT(Chain, BaseModel):
    """Controller model for the Sales Agent."""

    conversation_history: List[str] = []
    current_conversation_stage: str = '1'
    stage_analyzer_chain: StageAnalyzerChain = Field(...)
    sales_conversation_utterance_chain: MedicalConversationChain = Field(...)

    sales_agent_executor: Union[AgentExecutor, None] = Field(...)
    use_tools: bool = False

    conversation_stage_dict: Dict = {
        '1' : "Patient Profile: Start the conversation by introducing yourself as a General Physician. Ask the patient about the Name, Age, Gender and Occupation. Do not move to the next stage until the patient provides Age, Gender and Occupation. If you do not find Age, Gender and Occupation of patient in the conversation history, Move the conversation to this stage and remain in this stage until Age, Gender and occupation information is provided",
        '2': "Presentating Complaint: Request the patient to describe their primary symptoms in detail. ",
        '3': "History of presenting Complaint: Ask questions to uncover the nature, onset, duration and progression of the patient complaint. Move to next conversation stage when patient complaint is understood.",
        '4': "Systemic Enquiry: Ask specific questions about each body system to check for any additional symptoms or clues that might help in formulating a differential diagnosis. This may include questions about respiratory, cardiovascular, gastrointestinal, neurological, musculoskeletal, and other systems. Listen carefully to their responses and take notes.",
        '5': "Patient Past History: Ask specific questions about patient's past medical history. The past medical history includes any previous illnesses, surgeries, allergies, chronic conditions and blood transfusions. The physician must also enquire about drug history. Listen carefully to their responses and take notes.",
        '6': "Family History: Ask questions to obtain information about the patient's family history of relevant medical conditions including Ischemic Heart disease (IHD), Diabetes Mellitus (DM), Hypertension (HTN), Asthma, Malignancy or Genetic Disorders. Listen carefully to their responses and take notes.",
        '7': "Socioeconomic/Travel History: Ask the patient about relevant socio-economic or travel history.",
        '8': "Differential Diagnosis: Based on the information gathered, Create and present a list of differential diagnoses that could explain the patient's symptoms.",
        '9': "Final Diagnosis: Ask the questions to rule out unlikely differential diagnosis and narrow down the list of Diagnosis to reach one or two final differentials. Provide a concise summary of findings and present potential final diagnosis along with rationale.",
        '10': "Refer or Treatment: If the patient condition is worse and require immediate physical medical attention, strongly advise the patient for physical doctor or specialist visit. Otherwise provide a treatment strategy which may include off-the-shelf medications, lifestyle changes, or referrals to specialists if needed",
        '11': "Close: Close the conversation in a professional and polite manner"

        }

    physician_name: str = "Ben"
    company_name: str = "InnoTech"

    def retrieve_conversation_stage(self, key):
        return self.conversation_stage_dict.get(key, '1')
    
    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def seed_agent(self):
        # Step 1: seed the conversation
        self.current_conversation_stage= self.retrieve_conversation_stage('1')
        self.conversation_history = []

    def determine_conversation_stage(self):
        conversation_stage_id = self.stage_analyzer_chain.run(
            conversation_history='"\n"'.join(self.conversation_history), current_conversation_stage=self.current_conversation_stage)

        self.current_conversation_stage = self.retrieve_conversation_stage(conversation_stage_id)
        print(f"Conversation Stage: {conversation_stage_id}")

        return conversation_stage_id
        
    def human_step(self, human_input):
        # process human input
        human_input = 'User: '+ human_input + ' <END_OF_TURN>'
        self.conversation_history.append(human_input)

    def step(self):
        ai_response = self._call(inputs={})
        return ai_response
    
    def _call(self, inputs: Dict[str, Any]) -> None:
        """Run one step of the sales agent."""
        
        # Generate agent's utterance
        if self.use_tools:
            ai_message = self.sales_agent_executor.run(
                input="",
                conversation_stage=self.current_conversation_stage,
                conversation_history="\n".join(self.conversation_history),
                physician_name=self.physician_name,
                company_name=self.company_name,
            )

        else:
        
            ai_message = self.sales_conversation_utterance_chain.run(
                physician_name = self.physician_name,
                company_name=self.company_name,
                conversation_history="\n".join(self.conversation_history),
                conversation_stage = self.current_conversation_stage,
            )
        
        # Add agent's response to conversation history
        # print(f'{self.physician_name}: ', ai_message.rstrip('<END_OF_TURN>'))
        agent_name = self.physician_name
        ai_message = agent_name + ": " + ai_message
        if '<END_OF_TURN>' not in ai_message:
            ai_message += ' <END_OF_TURN>'
        self.conversation_history.append(ai_message)

        return ai_message

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, verbose: bool = False, **kwargs
    ) -> "SalesGPT":
        """Initialize the SalesGPT Controller."""
        stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=verbose)

        sales_conversation_utterance_chain = MedicalConversationChain.from_llm(
                llm, verbose=verbose
            )
        
        if "use_tools" in kwargs.keys() and kwargs["use_tools"] is False:

            sales_agent_executor = None

        else:
            product_catalog = kwargs["product_catalog"]
            tools = get_tools(product_catalog)

            prompt = CustomPromptTemplateForTools(
                template=PHYSICIAN_AGENT_TOOLS_PROMPT,
                tools_getter=lambda x: tools,
                # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
                # This includes the `intermediate_steps` variable because that is needed
                input_variables=[
                    "input",
                    "intermediate_steps",
                    "physician_name",
                    "company_name",
                    "conversation_history",
                ],
            )
            llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)

            tool_names = [tool.name for tool in tools]

            # WARNING: this output parser is NOT reliable yet
            ## It makes assumptions about output from LLM which can break and throw an error
            output_parser = SalesConvoOutputParser(ai_prefix=kwargs["physician_name"])

            sales_agent_with_tools = LLMSingleActionAgent(
                llm_chain=llm_chain,
                output_parser=output_parser,
                stop=["\nObservation:"],
                allowed_tools=tool_names,
                verbose=verbose
            )

            sales_agent_executor = AgentExecutor.from_agent_and_tools(
                agent=sales_agent_with_tools, tools=tools, verbose=verbose
            )


        return cls(
            stage_analyzer_chain=stage_analyzer_chain,
            sales_conversation_utterance_chain=sales_conversation_utterance_chain,
            sales_agent_executor=sales_agent_executor,
            verbose=verbose,
            **kwargs,
        )

# Set up of your agent

# Conversation stages - can be modified
conversation_stages = {
'1' : "Patient Profile: Start the conversation by introducing yourself as a General Physician. Ask the patient about the Name, Age, Gender and Occupation. Ensure that the patient has provided Age, Gender and Occupation before moving to next stage.",
'2': "Presentating Complaint: Request the patient to describe their primary symptoms in detail. ",
'3': "History of presenting Complaint: Ask questions to uncover the nature, onset, duration and progression of the patient complaint. Move to next conversation stage when patient complaint is understood.",
'4': "Systemic Enquiry: Ask specific questions about each body system to check for any additional symptoms or clues that might help in formulating a differential diagnosis. This may include questions about respiratory, cardiovascular, gastrointestinal, neurological, musculoskeletal, and other systems. Listen carefully to their responses and take notes.",
'5': "Patient Past History: Ask specific questions about patient's past medical history. The past medical history includes any previous illnesses, surgeries, allergies, chronic conditions and blood transfusions. The physician must also enquire about drug history. Listen carefully to their responses and take notes.",
'6': "Family History: Ask questions to obtain information about the patient's family history of relevant medical conditions including Ischemic Heart disease (IHD), Diabetes Mellitus (DM), Hypertension (HTN), Asthma, Malignancy or Genetic Disorders. Listen carefully to their responses and take notes.",
'7': "Socioeconomic/Travel History: Ask the patient about relevant socio-economic or travel history.",
'8': "Differential Diagnosis: Based on the information gathered, Create and present a list of differential diagnoses that could explain the patient's symptoms.",
'9': "Final Diagnosis: Ask the questions to rule out unlikely differential diagnosis and narrow down the list of Diagnosis to reach one or two final differentials. Provide a concise summary of findings and present potential final diagnosis along with rationale.",
'10': "Refer or Treatment: If the patient condition is worse and require immediate physical medical attention, strongly advise the patient for physical doctor or specialist visit. Otherwise provide a treatment strategy which may include off-the-shelf medications, lifestyle changes, or referrals to specialists if needed",
'11': "Close: Close the conversation in a professional and polite manner"
}

# Agent characteristics - can be modified
config = dict(
physician_name = "Ben",
company_name="InnoTech",
conversation_history=[],
conversation_stage = conversation_stages.get('1', "Patient Profile: Start the conversation by introducing yourself as a General Physician. Ask the patient about the Name, Age, Gender and Occupation. Do not move to the next stage until the patient provides Age, Gender and Occupation. If you do not find Age, Gender and Occupation of patient in the conversation history, Move the conversation to this stage and remain in this stage until Age, Gender and occupation information is provided"),
use_tools=False
)



