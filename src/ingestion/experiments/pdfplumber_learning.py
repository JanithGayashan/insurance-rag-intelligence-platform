import pdfplumber

PDF_PATH = "../../../data/raw/allianz/motor/Allianz_Motor_Fle_ (Modular Product).pdf"

def experiment_open_pdf():
    with pdfplumber.open(PDF_PATH) as pdf:
        print("PDF Object:", pdf)
        print("Number of pages:", len(pdf.pages))

def experiment_page():
    with pdfplumber.open(PDF_PATH) as pdf:
        page = pdf.pages[3]

        print("Page object:", page)
        print("Width:", page.width)
        print("Height:", page.height)

def experiment_text():
    with pdfplumber.open(PDF_PATH) as pdf:
        page = pdf.pages[3]

        text = page.extract_text()
        print(text)

def experiment_word():
    with pdfplumber.open(PDF_PATH) as pdf:
        page = pdf.pages[3]

        words = page.extract_words()

        print("Number of words:", len(words))
        print("DataType of words:", type(words))

        for word in words[:20]:
            print(word)

def experiment_character():
    with pdfplumber.open(PDF_PATH) as pdf:
        page = pdf.pages[3]

        character = page.chars[:20]

        for char in character:
            print(char)

def experiment_graphics():
    with pdfplumber.open(PDF_PATH) as pdf:
        page = pdf.pages[23]

        print("Lines:", len(page.lines))
        print("Rectangles:", len(page.rects))

        print("\nFirst few lines:")

        for line in page.lines[:10]:
            print(line)

        print("\nFirst few rectangles:")

        for rect in page.rects[:10]:
            print(rect)

def experiment_find_tables():
    with pdfplumber.open(PDF_PATH) as pdf:
        page = pdf.pages[23]

        tables = page.find_tables()

        print("Number of tables:", len(tables))

        for index, table in enumerate(tables):
            print(f"\nTABLE {index}")
            print("Bounding box:", table.bbox)

            extracted = table.extract()

            for row in extracted:
                print(row)

def experiment_table_debug():
    with pdfplumber.open(PDF_PATH) as pdf:
        page = pdf.pages[23]

        image = page.to_image(
            resolution=150
        )

        image.debug_tablefinder()

        image.save(
            "pdfplumber_table_debug.png"
        )

if __name__ == "__main__":
    # experiment_open_pdf()
    # experiment_page()
    # experiment_text()
    # experiment_word()
    # experiment_character()
    # experiment_graphics()
    experiment_find_tables()
    # experiment_table_debug()