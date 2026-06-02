from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_text(text)

    return chunks


if __name__ == "__main__":

    sample_text = """
    Infosys reported revenue growth in FY24.
    Operating margin stood at 20.7%.
    Attrition declined significantly.
    """ * 50

    chunks = chunk_text(sample_text)

    print(f"Total Chunks: {len(chunks)}")

    print("\nFIRST CHUNK:\n")
    print(chunks[0])

    print("\nSECOND CHUNK:\n")
    print(chunks[1])