from docling.document_converter import DocumentConverter


PDF_PATH = "../../../data/raw/allianz/motor/Allianz_Motor_Fle_ (Modular Product).pdf"


def experiment_basic_conversion():

    converter = DocumentConverter()

    result = converter.convert(PDF_PATH)

    document = result.document

    print("Document:", document)
    print("Pages:", document.num_pages())

def experiment_text():

    converter = DocumentConverter()

    result = converter.convert(PDF_PATH)

    document = result.document

    text = document.export_to_text()

    print(text)

if __name__ == "__main__":
    # experiment_basic_conversion()
    experiment_text()