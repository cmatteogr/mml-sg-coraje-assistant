from loguru import logger

from .base import BaseLocalCrawler
import os
from llm_engineering.domain.documents import TranscriptionDocument
from pathlib import Path


class TranscriptionCrawler(BaseLocalCrawler):
    model = TranscriptionDocument

    def extract(self, link: str, **kwargs) -> None:
        transcription_name = Path(os.path.basename(link)).stem
        logger.info(f"Starting scrapping data for Meeting Transcription: {link}")

        old_model = self.model.find(name=transcription_name)
        if old_model is not None:
            logger.info(f"Transcription already exists in the database: {link}")
            return

        # init destination and source paths
        destination_path = os.path.join(self.destination_folder, 'meetings_transcriptions_txt', link)
        # check if source file exist
        if not os.path.exists(destination_path):
            raise FileNotFoundError(f"{destination_path} does not exist")

        transcription = self._extract_transcription(destination_path)
        logger.info(f"Found {len(transcription)} meeting transcription for: {link}")
        # save new book
        instance = self.model(platform="transcription", content=transcription, name=transcription_name,
                              filepath=destination_path)
        instance.save()

        logger.info(f"Finished scrapping data for meeting transcription: {link}")

    def _extract_transcription(self, filepath: str) -> str:
        """
        Extracts transcription text.

        Args:
            filepath str: Document filepath

        Returns:
            str: Document text
        """
        with open(filepath, "r") as file:
            content = file.read()
        return content
