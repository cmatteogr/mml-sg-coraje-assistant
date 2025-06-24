from abc import ABC
from .base import NoSQLBaseDocument
from .types import DataCategory


class Document(NoSQLBaseDocument, ABC):
    content: str
    platform: str

class TranscriptionDocument(Document):
    name: str
    filepath: str

    class Settings:
        name = DataCategory.MEETING_TRANSCRIPTION


class MLBookDocument(Document):
    filepath: str
    name: str
    author: str

    class Settings:
        name = DataCategory.ML_BOOK


class GithubCodeDocument(Document):
    filepath: str
    name: str
    project_path: str
    project_url: str
    sha: str
    repo: str

    class Settings:
        name = DataCategory.GITHUB_CODE

class MMLSGBaseDocument(Document):
    filepath: str
    name: str

    class Settings:
        name = DataCategory.MML_SG_BASE