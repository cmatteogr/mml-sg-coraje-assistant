import re
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from llm_engineering.application.networks import EmbeddingModelSingleton
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

embedding_model = EmbeddingModelSingleton()


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=chunk_size, chunk_overlap=0)
    text_split_by_characters = character_splitter.split_text(text)

    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=chunk_overlap,
        tokens_per_chunk=embedding_model.max_input_length,
        model_name=embedding_model.model_id,
    )
    chunks_by_tokens = []
    for section in text_split_by_characters:
        chunks_by_tokens.extend(token_splitter.split_text(section))

    return chunks_by_tokens


def chunk_document(text: str, min_length: int, max_length: int) -> list[str]:
    """Alias for chunk_article()."""

    return chunk_article(text, min_length, max_length)


def chunk_article(text: str, min_length: int, max_length: int) -> list[str]:
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)

    extracts = []
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            if len(current_chunk) >= min_length:
                extracts.append(current_chunk.strip())
            current_chunk = sentence + " "

    if len(current_chunk) >= min_length:
        extracts.append(current_chunk.strip())

    return extracts
# default headers to split base
_headers_to_split_on_base = [
        ("#", "#"),
        ("##", "##"),
        ("###", "###"),
    ]

def chunk_header_chunk(text: str,
                       tokenizer_model:str="TheBloke/Nous-Hermes-2-Mixtral-8x7B-DPO-GPTQ",
                       headers_to_split_on=None,
                       max_tokens:int=200):
    """
    Chunking using Langchain text splitter
    :param text: text to split in chunks
    :param tokenizer_model: tokenizer model to use
    :param headers_to_split_on: headers to split on
    :param max_tokens: number of tokens to split
    :return: list of chunks
    """
    # init the tokenizer
    if headers_to_split_on is None:
        headers_to_split_on = _headers_to_split_on_base

    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    # NOTE: This method fix text to avoid remove headers in the same level when they are consecutive and text between them empty
    # this method add a dash between these headers to use MarkdownHeaderTextSplitter without this issue
    # text = fix_markdown_header_text_splitter_text(text)

    # split the markdown using the headers
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_splits = markdown_splitter.split_text(text)

    # define the longest headers token size
    chunks_metadata = [chunk.metadata for chunk in md_splits]
    longest_header_tokens = max([max([len(tokenizer.encode(c.get("#", ""))),
                                      len(tokenizer.encode(c.get("##", ""))),
                                      len(tokenizer.encode(c.get("###", "")))]) for c in chunks_metadata])

    chunk_splits_size = max_tokens - (longest_header_tokens * 3)
    custom_separators = ["\\.", "\n", " ", ""]
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer=tokenizer,
                                                                              chunk_size=chunk_splits_size,
                                                                              chunk_overlap=0,
                                                                              separators=custom_separators)
    # split longest splits
    md_splits = text_splitter.split_documents(md_splits)

    # concatenate the chunk based on threshold chunk size
    new_docs = []
    chunk = ""
    doc_index = 0
    breadcrumb = []
    while True:
        # check if doc index is greater than the markdowns splits list
        if doc_index >= len(md_splits):
            # append last text chunk to list and break the loop
            new_docs.append(chunk)
            break

        # get doc in current index
        doc = md_splits[doc_index]
        # remove first item in the metadata list, it should exist once per chunk
        breadcrumb_temp = [f"{k} {v}" for k,v in doc.metadata.items()]

        main_header = '\n'.join([item for item in breadcrumb_temp if item not in breadcrumb])

        if main_header != "":
            breadcrumb = breadcrumb_temp

        # add the rest of the headers
        # concatenate the headers and the page content
        new_docs_text = f"{main_header}\n\n{doc.page_content}\n\n"

        # check if new chunk length in smaller than the threshold
        sum_tokens = len(tokenizer.tokenize(chunk + new_docs_text))
        if sum_tokens > max_tokens:
            # append the current open chunk
            new_docs.append(chunk)
            # create a new chunk, set to blank
            chunk = '\n'.join(breadcrumb_temp)
            new_docs_text = f"\n\n{doc.page_content}\n\n"

        # add new chunk text
        chunk += new_docs_text

        # increase doc index
        doc_index += 1

    return new_docs

def chunk_overlap(text, chunk_size=1500, overlap=50):
    """
    Chunking using overlapping strategy
    :param text: text to split in chunks
    :param chunk_size: max chunk size
    :param overlap: overlap between chunks
    :return: list of chunks
    """
    # split by lines to process
    lines = text.split("\n")

    # Store chunks
    chunks = []
    current_chunk = []
    # for each line concatenate and overlap
    for line in lines:
        # If it's a heading, start a new chunk
        if re.match(r"^#{1,6} ", line):
            if current_chunk:
                # Store the previous chunk with buffer overlap
                chunk_text = "\n".join(current_chunk)
                chunks.append(chunk_text)
                # Keep overlap from last chunk
                buffer_chunk = chunk_text[-overlap:]
                # Reset chunk
                current_chunk = [buffer_chunk]

        # append line - chunk
        current_chunk.append(line)

        # check if the chunk exceeds the size limit
        if len("\n".join(current_chunk)) >= chunk_size:
            chunk_text = "\n".join(current_chunk)
            chunks.append(chunk_text)
            # keep last overlap portion
            buffer_chunk = chunk_text[-overlap:]
            # start a new chunk with the overlap
            current_chunk = [buffer_chunk]

    # append chunk
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    # return chunk
    return chunks