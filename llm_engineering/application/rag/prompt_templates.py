from langchain.prompts import PromptTemplate

from .base import PromptTemplateFactory

class SummarizeMLTranscriptionTemplate(PromptTemplateFactory):
    prompt: str = """The following is a transcription from a Machine Learning meeting, 
    the members are part of a study group focused on developing Machine Learning projects from scratch.
    Maybe some details about the conversation/speak are missing because the content is a transcription of a virtual meeting, there isn't access to the video/images which are part of the meeting recording and sometimes the audio quality not good enough.
    Summarize in detail the transcription extracting the relevant information; comments, ideas, explanations, projects descriptions,
    etc. Clean the transcription if needed. The goal is use the transcription summary to know:
    - Title: Meeting summary title.
    - Summary: Meeting summary description.
    - Topics: What is the mein topic and subtopics.
    - Tools: What tools did they mention/use?
    - Resources: What resources (books, websites, etc.) did they use?
    - Challenges. If any, How they solved them?
    Transcription text: {transcription}
    
    Only return the summary with the items defined above, nothing else.
    Return the format as a markdown, where 'Title' is the main title and the rest of items are subtitles.
    """

    def create_template(self) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["transcription"]
        )


class SummarizeMLTranscriptionGetMetadataTemplate(PromptTemplateFactory):
    prompt: str = """The following is a transcription from a Machine Learning meeting, 
    the members are part of a study group focused on developing Machine Learning projects.
    Maybe some details about the conversation/speak are missing due there isn't access to the video/images meeting and sometimes the audio quality not good enough.
    Extract or infer the topics and tools mentioned in the meeting, usually they meet to discus a project they are working on.
    Extract the topics and tools in JSON format with two keys 'topics' and 'tools', both elements contains a list of strings all the items in lower case.
    The tools should be technical tools related to Machine Learning, Data science, Statistics, Programming, Algebra, Calculus or similar fields. 
    
    Example #1:
    {{
    "topics": ["supervised learning", "cars", "price prediction"],
    "tools": ["scikit-learn", "mlflow", "airflow", "tensorflow"]
    }}
    
    Example #2:
    {{
    "topics": ["unsupervised learning", "kmeans", "customer profiling"],
    "tools": ["scikit-learn", "data-profiling", "jupyter notebook"]
    }}
    
    Transcription text: {transcription}

    Only return the JSON object as a text with the items defined above, nothing else.
    """

    def create_template(self) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["transcription"]
        )


class SummarizeMLBookIndexTemplate(PromptTemplateFactory):
    prompt: str = """The following are the home pages of Machine Learning or related books, 
    Extract authors and book topics in JSON format with two keys 'authors' and 'topics', both elements contain a list of strings, all the elements are lowercase.
    
    Example #1:
    {{
    "authors": ["charles a. kamhoua", "chistopher d. kiekintveld", "fei fang"],
    "topics": ["game theory", "bayesian games", "decision making", "attack graph"]
    }}
    
    Example #2:
    {{
    "authors": ["simon j.d. prince"],
    "topics": ["supervised learning", "linear regression", "deep neural network", "backpropagation"]
    }}

    Machine Learning book home pages text: {ml_book_home_pages}

    Only return the JSON object as a text with the items defined above, nothing else.
    """

    def create_template(self) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["ml_book_home_pages"]
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
    prompt: str = """You are an AI language model assistant. Your task is to generate {expand_to_n}
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Return a JSON format with one key 'alternatives' which contain a list of strings (the alternative queries).
    
    Original question: {question}
    
    Only return the JSON object as a text with the items defined above, nothing else.
    """


    def create_template(self, expand_to_n: int) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["question"],
            partial_variables={
                "expand_to_n": expand_to_n,
            },
        )


class SelfQueryTemplate(PromptTemplateFactory):
    prompt: str = """You are an AI language model assistant. Your task is to extract information from a user question.
    The required information that needs to be extracted is the tools, topics or authors:
    
    * tools: technical tools related to Machine Learning, Data science, Statistics, Programming, Algebra, Calculus or similar fields.
    * topics: Machine Learning, Data science, Statistics, Programming, Algebra, Calculus or Business topics.
    * authors: Machine Learning book authors.
    
    Extract authors tools and topics in JSON format with two keys 'authors' 'tools', and 'topics', all of them contain a list of strings, all the elements are lowercase.
    If you don't find any information for tools, topics or authors in the user question, you should return an empty list for the corresponding key: []
    
    You can use these examples as reference:
    
    Example #1:
    {{
    "tools": ["scikit-learn", "mlflow", "airflow", "tensorflow"],
    "topics": ["game theory", "bayesian games", "decision making", "attack graph"],
    "authors": ["charles a. kamhoua", "chistopher d. kiekintveld", "fei fang"]
    }}
    
    Example #2:
    {{
    "tools": ["scikit-learn", "data-profiling", "jupyter notebook"],
    "topics": ["supervised learning", "linear regression", "deep neural network", "backpropagation"],
    "authors": []
    }}
    
    Example #3:
    {{
    "tools": [],
    "topics": [],
    "authors": ["simon j.d. prince"]
    }}
    
    User question: {question}
    
    Only return the JSON object as a text with the items defined above, nothing else.
    """

    def create_template(self) -> PromptTemplate:
        return PromptTemplate(template=self.prompt, input_variables=["question"])


class AIAssistantQueryTemplate(PromptTemplateFactory):
    prompt: str = """You are an AI assistant named Courage (like the dog cartoon) in spanish 'Coraje'. 
You are a member of Medellín Machine Learning - Study Group (MML-SG), it's a Machine Learning study group open and free.
Answer the questions related to projects, tools or topics discussed in the community using the provided context as the primary source of information for the content. .

{user_query_context}
    """

    def create_template(self) -> PromptTemplate:
        return PromptTemplate(template=self.prompt, input_variables=["user_query_context"])