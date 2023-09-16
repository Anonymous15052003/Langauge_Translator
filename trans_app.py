import streamlit as st
from PIL import Image
import PyPDF2 as pdf
from PyPDF2 import PdfReader, PdfWriter
import pdfplumber
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import io
import pickle


image1 = Image.open('SIH2023-logo.png')

st.image(image1)

# image2 = Image.open('LOGO1.png')

# st.image(image2, caption='Hacksquard logo')

st.write(" # Language Translator")

# Upload a PDF file
def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to convert text to PDF
def text_to_pdf(text):
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    # Create a list of Paragraph objects
    content = []
    content.append(Paragraph(text, styles["Normal"]))

    # Build the PDF
    doc.build(content)
    pdf_value = pdf_buffer.getvalue()
    pdf_buffer.close()
    return pdf_value

pdf_file = st.file_uploader("upload kar ",type=["pdf"])

if pdf_file is not None:
    st.write(f"Extracting text from PDF file : {pdf_file.name}")
    extracted_text = read_pdf(pdf_file)

#transaltion


# Load the translation model and tokenizer from pickle files
with open("translation_model.pkl", "rb") as model_file:
    translation_model = pickle.load(model_file)

with open("tokenizer.pkl", "rb") as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Translate button
if st.button("Translate"):
    if extracted_text:
        model_inputs = tokenizer(extracted_text, return_tensors="pt")

    generated_tokens = translation_model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"]
    )

    extracted_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    # Display the translated text
    st.write(extracted_text[0])

# Convert text to PDF and provide a download button
if st.button("Generate PDF"):
    if extracted_text[0]:
        pdf_data = text_to_pdf(extracted_text[0])
        st.download_button(
            label="Download PDF",
            data=pdf_data,
            file_name="converted_text.pdf",
            key="download_button"
        )

# Display some content in the app
st.write("This is some content below the download button.")