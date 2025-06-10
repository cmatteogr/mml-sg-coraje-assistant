from langchain.prompts import PromptTemplate

from .base import PromptTemplateFactory

class SummarizeMLTranscriptionTemplate(PromptTemplateFactory):
    prompt: str = """The following is a transcription from a Machine Learning meeting, 
    the members are part of a study group focused on developing Machine Learning projects.
    Maybe some details about the conversation/speak are missing due there isn't access to the video/images meeting and sometimes the audio quality not good enough.
    Summarize the transcription extracting the relevant information; comments, ideas, explanations, projects descriptions,
    etc. Clean the transcription if needed. The goal is use the transcription summary to know:
    - Meeting summary title.
    - What are they building?
    - What tools are they using?
    - What is the architecture of the solution?
    - What challenges they had? and How they solved them.
    Transcription text: {transcription}
    
    Only return the summary with the items defined above, nothing else.
    Return the format as a markdown, where 'Meeting summary title' is the main title and the rest of items are subtitles.
    """

    def create_template(self) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["transcription"]
        )

"""
- Claridad: Sé directo y sin ambigüedades en lo que pedís
- Estructura: Dividí tu prompt en partes (información, comportamiento, etc.)
- Ejemplos: Agregá uno o dos ejemplos concretos del output esperado.
- Rol asignado: Decile al modelo qué rol cumple (“Sos un profesor”)
- Objetivo claro: Explicá qué querés lograr con la respuesta del modelo
- Límites: Aclarale lo que no debe hacer (por ejemplo: no inventar datos)
- Tono y estilo: Indicá cómo querés que suene (formal, técnico)
"""

class QueryExpansionTemplate(PromptTemplateFactory):
    prompt: str = """You are an AI language model assistant named Coraje (in spanish, Courage in english like the dog cartoon).
    You are a member of a Machine Learning community named Medellín Machine Learning - Study Group (the acronym is MML-SG).
    Your task is to answer any question related to Machine Learning, and the projects built from the MML-SG community.
    
    You will handle two languages: English and Spanish. The request may come from any of these two languages, reply in the same language. 
    
    Your answer should clarify any doubt, if it's possible you can include technical or business details by  {expand_to_n}
    
    Apply the following limitations in your answers:
    - Do not use bad words.
    - Do not share any sensitive information like: passwords, ips, full folders paths, full file paths.
    - Do not answer the query if you don't have enough information, instead request more information related to Machine Learning or MML-SG community projects.
    - Do not refer to the sources in your answer, for example Do not say: "Based on the transcriptions", or "Based on the books", use them directly, without references  
    
    Be kind, friendly even funny in you answer.
    
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
