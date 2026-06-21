from langchain_text_splitters import (RecursiveCharacterTextSplitter)

def chunk_text(text: str):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    return splitter.split_text(text)


def chunk_page(text: str):
    if not text or not text.strip():
        return []

    return chunk_text(text)