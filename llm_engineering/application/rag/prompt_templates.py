from langchain.prompts import PromptTemplate

from .base import PromptTemplateFactory

class SummarizeMLTranscriptionTemplate(PromptTemplateFactory):
    prompt: str = """The following is a transcription from a Machine Learning meeting, 
    the members are part of a study group focused on developing Machine Learning projects.
    Maybe some details about the conversation/speak are missing due there isn't access to the video/images meeting.
    Summarize the transcription extracting the relevant information; comments, ideas, explanations, projects descriptions,
    etc. Clean the transcription if needed. The goal is use the transcription summary to know:
    - What are they building?
    - What tools are they using?
    - What is the architecture of the solution?
    - What challenges they had? and How they solved them.
    Transcription text: {transcription}"""

    def create_template(self) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["transcription"]
        )


class QueryExpansionTemplate(PromptTemplateFactory):
    prompt: str = """You are an AI language model assistant. Your task is to generate {expand_to_n}
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions seperated by '{separator}'.
    Original question: {question}"""

    @property
    def separator(self) -> str:
        return "#next-question#"

    def create_template(self, expand_to_n: int) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["question"],
            partial_variables={
                "separator": self.separator,
                "expand_to_n": expand_to_n,
            },
        )


class SelfQueryTemplate(PromptTemplateFactory):
    prompt: str = """You are an AI language model assistant. Your task is to extract information from a user question.
    The required information that needs to be extracted is the user name or user id. 
    Your response should consists of only the extracted user name (e.g., John Doe) or id (e.g. 1345256), nothing else.
    If the user question does not contain any user name or id, you should return the following token: none.
    
    For example:
    QUESTION 1:
    My name is Paul Iusztin and I want a post about...
    RESPONSE 1:
    Paul Iusztin
    
    QUESTION 2:
    I want to write a post about...
    RESPONSE 2:
    none
    
    QUESTION 3:
    My user id is 1345256 and I want to write a post about...
    RESPONSE 3:
    1345256
    
    User question: {question}"""

    def create_template(self) -> PromptTemplate:
        return PromptTemplate(template=self.prompt, input_variables=["question"])
