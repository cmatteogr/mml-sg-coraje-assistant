from loguru import logger
from .base import BaseLocalCrawler
import os
from llm_engineering.domain.documents import GithubCodeDocument
import base64
from github import Github, GithubException, ContentFile


class GithubCodeCrawler(BaseLocalCrawler):
    model = GithubCodeDocument

    def extract(self, link: str, **kwargs) -> None:
        logger.info(f"Starting scrapping data for Github code: {link}")

        # get github session
        GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
        github_session = Github(GITHUB_TOKEN)
        # download code
        repo = github_session.get_repo(link)
        repo_content = self._get_github_repository_data(repo)

        # for each document in the repo
        for element_name, element_content in repo_content.items():
            # get extension file, valid extension
            extension_file = os.path.splitext(element_name)[-1]
            if extension_file not in ['.py', '.ipynb', '.md', '.html', '.txt']:
                continue

            full_name = link + '/' + element_name

            old_model = self.model.find(name=full_name)
            if old_model is not None:
                logger.info(f"Github code already exists in the database: {full_name}")
                continue

            instance = self.model(platform="github",
                                  content=element_content['content'],
                                  filepath=element_content['path'],
                                  name=full_name,
                                  project_path= element_content['path'],
                                  project_url= element_content['url'],
                                  sha= element_content['sha'],
                                  repo=link)
            instance.save()

        logger.info(f"Finished scrapping data for Github code: {link}")

    def _get_github_repository_data(self, repo, current_path: str = "") -> list[dict]:
        """
        Recursively extracts the text content of all files in a given path of a repository.
        """
        files_content_map = {}
        # get repo content in current path
        contents = repo.get_contents(current_path)

        # list the items to process
        items_to_process = []
        if isinstance(contents, ContentFile.ContentFile):
            items_to_process.append(contents)
        elif isinstance(contents, list):
            items_to_process.extend(contents)

        while items_to_process:
            # get first item in the list until they are processed
            content_item = items_to_process.pop(0)

            if content_item.type == "dir":
                # if it's a directory, fetch its contents and add them to our processing list
                print(f"processing folder: {content_item.path}")
                try:
                    dir_contents = repo.get_contents(content_item.path)
                    if isinstance(dir_contents, list):  # Ensure it's a list of items
                        items_to_process.extend(dir_contents)
                    elif isinstance(dir_contents, ContentFile.ContentFile):  # Should not happen for a dir path
                        items_to_process.append(dir_contents)

                except GithubException as e:
                    logger.error(
                        f"Error accessing directory contents for '{content_item.path}' in '{repo.full_name}': {e}"
                    )
            elif content_item.type == "file":
                # if it's a file, try to decode its content
                print(f"processing file: {content_item.path}")
                # Ensure content is present and encoding is base64 (standard for file content via API)
                if content_item.encoding == "base64" and content_item.content:
                    try:
                        decoded_content = base64.b64decode(content_item.content).decode("utf-8")
                        file_content = {
                            "name": content_item.name,
                            "path": content_item.path,
                            "sha": content_item.sha,
                            "url": content_item.url,
                            "file_size": content_item.file_size,
                            "content": decoded_content
                        }
                        files_content_map[content_item.path] = file_content
                        print(f"Successfully decoded content for: {content_item.path}")
                    except UnicodeDecodeError:
                        logger.warning(
                            f"Could not decode file '{content_item.path}' in '{repo.full_name}' as UTF-8. Skipping.")
                    except Exception as e:  # Catch any other decoding/processing errors
                        logger.error(
                            f"Error processing file content for '{content_item.path}' in '{repo.full_name}': {e}")
                elif not content_item.content:
                    logger.debug(f"File '{content_item.path}' in '{repo.full_name}' has no content. Skipping.")
                elif content_item.encoding != "base64":
                    logger.debug(
                        f"File '{content_item.path}' in '{repo.full_name}' is not base64 encoded (encoding: {content_item.encoding}). Skipping direct content decoding."
                    )

        return files_content_map
