from loguru import logger
from .base import BaseLocalCrawler
import os
from llm_engineering.domain.documents import MLBookDocument
from pathlib import Path
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
from langchain_core.documents import Document


def documents_to_markdown(docs: list[Document]) -> str:
    """
    Transforms a list of documents from UnstructuredPDFLoader into a single
    Markdown string, mapping element categories to Markdown syntax.
    """
    markdown_segments = []
    # A simple way to handle nested titles (e.g., #, ##, ###)
    title_level = 1

    for doc in docs:
        category = doc.metadata.get('category')

        # Skip headers and footers
        if category in ["Header", "Footer"]:
            continue

        if category == "Title":
            # Use '#' for titles, increasing level for subsequent titles
            # This is a heuristic; you might need more sophisticated logic
            # for complex documents.
            segment = f"{'#' * title_level} {doc.page_content}"
            if title_level < 3: # Cap at ###
                title_level += 1
            # Reset title level if we encounter a big gap or new chapter text
            # (More advanced logic could be added here)

        elif category == "ListItem":
            segment = f"- {doc.page_content}"

        elif category == "CodeSnippet":
            # Assume python for now, can be made more generic
            segment = f"```python\n{doc.page_content}\n```"

        elif category == "Table":
            # Wrapping tables in code blocks is a robust way to preserve formatting
             segment = f"```\n{doc.page_content}\n```"

        else: # NarrativeText, Uncategorized, etc.
            segment = doc.page_content
            # Reset title level when we hit a normal paragraph
            title_level = 1

        markdown_segments.append(segment)

    # Join segments with double newlines for proper Markdown spacing
    return "\n\n".join(markdown_segments)


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
        u_pdf_ml_book = UnstructuredPDFLoader(filepath, mode="elements").load()
        # return PDF text
        markdown_content = documents_to_markdown(u_pdf_ml_book)
        return markdown_content


