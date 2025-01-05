## PDFDocumentMultiColumnTextProcessing

# What it does
Python scripts used for when processing/categorising large amounts of PDF documents. 
The Column Layout Detector script will attempt to separate/detect PDF documents that have a double-columned text layout from single-columned documents
The Double Column Text Extractor script will attempt to extract text from double (and single too) columned PDF doucments, and arrange these texts into sections and write them into a JSON file.

# Why PyMuPDF and PPStruct
The scripts' functionality are based on using many functionailities provided by the PyMuPDF-Fitz and PaddleOCR-PPStruct libraries.
PyMuPDF-Fitz have a very useful function to extract text block location, and so an aggregation of relative locations of all text across all pages into a "score" system very efficiently handles the separation between single-columned and double-columned PDF.
Using an OCR approach to extract text is significantly slower than extracting from the data present within the PDF directly, however this is done because a significant percentage PDF documents I was dealing with had bad data decay and corruption as well as being almost exclusively in Chinese, and so directly retrieving text information in the PDF resulted in too many unusable data. And so, an OCR image scanning approach was used through PaddleOCR to sidestep these issues.

PyMuPDF repo: https://github.com/pymupdf/PyMuPDF
PaddleOCR repo: https://github.com/PaddlePaddle/PaddleOCR
