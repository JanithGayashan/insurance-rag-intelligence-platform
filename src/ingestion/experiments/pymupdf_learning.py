import pymupdf

PDF_PATH = "../../../data/raw/allianz/motor/Allianz_Motor_Fle_ (Modular Product).pdf"

def experiment_open_pdf():
    document = pymupdf.open(PDF_PATH)

    print("Document:", document)
    print("Number of pages:", len(document))
    print("DataType Of document:", type(document))

    page_1 = document[0]
    print("DataType of page_1:", type(page_1))
    print("Content Of page_1:", page_1)

    text = page_1.get_text()
    print("DataType of text:", type(text))

    page_4 = document[23]
    print("Page number:", page_4.number)
    print("Page_4 width:", page_4.rect.width)
    print("Page_4 height:", page_4.rect.height)

    print("Page_1 width:", page_1.rect.width)
    print("Page_1 height:", page_1.rect.height)

    text_4 = page_4.get_text("text")
    print(text_4)

    blocks = page_4.get_text("blocks")
    print("Number of blocks:", len(blocks))
    print("DataType of blocks:", type(blocks))

    # for block in blocks:
    #     print(block)
    #     print("-" * 80)

    
    document.close()

def experiment_blocks_readable():
    document = pymupdf.open(PDF_PATH)

    page = document[23]

    blocks = page.get_text(
        "blocks",
        sort = True
    )

    for block in blocks:
        x0 = block[0]
        y0 = block[1]
        x1 = block[2]
        y1 = block[3]

        text = block[4]
        block_number = block[5]
        block_type = block[6]

        print(f"Block number: {block_number}")
        print(f"Block type: {block_type}")

        print(
            f"Bounding box: "
            f"({x0}, {y0}, {x1}, {y1})"
        )

        print("Text:")
        print(text)

        print("=" * 80)

    document.close()

def experiment_words():
    document = pymupdf.open(PDF_PATH)

    page = document[3]

    words = page.get_text(
        "words",
        sort = True
    )

    print("DataType of Words:", type(words))

    for word in words[:20]:
        print(word)

    document.close()

def experiment_dict():
    document = pymupdf.open(PDF_PATH)

    page = document[3]

    data = page.get_text("dict")

    print(data.keys())

    document.close()

def experiment_dict_blocks():
    document = pymupdf.open(PDF_PATH)

    page = document[3]

    data = page.get_text("dict")

    for block in data["blocks"]:
        print(block)
        print("=" * 80)

    document.close()



if __name__ == "__main__":
    experiment_open_pdf()
    # experiment_blocks_readable()
    # experiment_words()
    # experiment_dict()
    # experiment_dict_blocks()