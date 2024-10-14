from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# function for text extraction

def text_extraction(file,typefile):
    text=''
    if typefile=='pdf':
        pdf=PdfReader(file)
        for page in pdf.pages:
            text+= page.extract_text()
        return(text)

    elif typefile in ('jpg','jpeg','png'):
        pass


# function to split text

def text_splitting(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    text_chunks=text_splitter.split_text(text)
    return(text_chunks)

