from loguru import logger
from .base import BaseLocalCrawler
import os
import fitz
from llm_engineering.domain.documents import MLBookDocument
from pathlib import Path
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader



class MLBookCrawler(BaseLocalCrawler):
    model = MLBookDocument

    def extract(self, link: str, **kwargs) -> None:
        book_name = Path(os.path.basename(link)).stem
        logger.info(f"Starting scrapping data for ML Books: {link}")

        old_model = self.model.find(name=book_name)
        if old_model is not None:
            logger.info(f"ML Book already exists in the database: {link}")
            return

        # init destination and source paths
        destination_path = os.path.join(self.destination_folder, 'books_pdf', link)
        # check if source file exist
        if not os.path.exists(destination_path):
            raise FileNotFoundError(f"{destination_path} does not exist")

        ml_book_text = self._extract_ml_book_pdf(destination_path)
        logger.info(f"Found {len(ml_book_text)} ml books for: {link}")

        # save new book
        instance = self.model(platform="ml_books", content=ml_book_text, name=book_name, author='', filepath=destination_path)
        instance.save()

        logger.info(f"Finished scrapping data for ml book: {link}")

    def _extract_ml_book_pdf(self, filepath: str) -> str:
        """
        Extracts ML Book PDF.

        Args:
            filepath str: Document filepath

        Returns:
            str: Document text
        """
        u_pdf_ml_book = UnstructuredPDFLoader(filepath).load()
        with fitz.open(filepath) as pdf:
            text = ""
            for page in pdf:
                text += page.get_text()
        # return PDF text
        return text
