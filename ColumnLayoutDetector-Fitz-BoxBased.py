"""
双栏区分

Author: Roger Zhang
Last Modified: 2024/05/17

Description:
Decides if PDF document layout is single-column or double-column using Paddle OCR

Update:
- 2024/04/25 Creation
- 2024/05/06 Paddle OCR custom implementation, various parametres added, parallel processing
- 2024/05/07 Changed from determining lines by characters to size occupying page
- 2024/05/13 Prepped for exe output
- 2024/05/17 Added timeout for functions to ensure program running 
- 2024/06/12 Simple fitz version, bypass paddleocr

Usage:
- Input one folder of PDF files
FOR BUILDING EXE:
- Add a version.text file to kmeans module in _internal
"""
# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import kmeans1d
from PIL import Image
import multiprocessing
import func_timeout
# import paddle.utils
import cv2
import fitz
fitz.TOOLS.mupdf_display_errors(False)
from VieTableRecognition.TableRecognitionDriver import Extract_Table_Meta_Info
# import pdfplumber
from tqdm import tqdm
from dataclasses import dataclass
from TimeRecorder import TimeRecorder
from tqdm.contrib.concurrent import process_map
import shutil
from itertools import islice
# import gc

@dataclass
class OCR_Config:
    # mininum percentage of page width to be considered a line
    minLineWidth: float
    # mininum confidence of text recognition to be considered
    minConfidence: float
    # Score threshold to decide if document is multi-column
    scoreThreshold: float
    # Threshold to determine if pages are too filled with tables
    tableThreshold: float
    # Max number of pages to take from document to go through OCR, limit for performance reasons
    maxNumPages: int
    # Worker threads
    workers: int
    # Chunksize/Batch size
    chunksize: int
    # Resize factor of images for OCR processing
    resizeFactor: float

def str2bool(v):
    return v.lower() in ("true", "yes", "t", "y", "1")

def remap(s, a1, a2, b1, b2):
    """
    Remaps a value 's'
    From scale 'a1' to 'a2'
    To scale 'b1' to 'b2'
    """
    return b1 + (s-a1)*(b2-b1)/(a2-a1)

def batched(iterable, chunk_size):
    iterator = iter(iterable)
    while chunk := tuple(islice(iterator, chunk_size)):
        yield chunk

def Get_Area_Of_Table(table_bounds):
    """
    Table bounds are in the form of (x0, top, x1, bottom)
    x0	Distance of left side of rectangle from left side of page.
    top	Distance of top of rectangle from top of page.
    x1	Distance of right side of rectangle from left side of page.
    bottom	Distance of bottom of the rectangle from top of page.

    Returns the area of the table
    """    
    tableWidth = table_bounds[2] - table_bounds[0]
    tableHeight = table_bounds[1] - table_bounds[3]

    return tableWidth*tableHeight

def Is_Within_Interval(coord, interval):

    if (not (interval[0][0] <= coord[0] <= interval[0][1])):
        return False
    if (not (interval[1][0] <= coord[1] <= interval[1][1])):
        return False

    return True

def Line_Scoring_Rule_1(TextPos, docWidth):
    """
    Returns score based text's X position relative to the document centre X position
    Rule: 
    Centre (1/2) position of PDF -> -1
    1/4 Position of PDF -> 1
    1/4 Position of PDF -> 1
    0 and 1 Position of PDF -> 0
    """
    
    score = 0

    if (TextPos < (docWidth/4)):
        # Text between 0 and 1/4
        score = remap(TextPos, 0, docWidth/4, 0, 1)
    elif (TextPos < (docWidth/2)):
        # Text between 1/4 and 1/2
        score = remap(TextPos, docWidth/4, docWidth/2, 1, -1)
    elif (TextPos < 3 * (docWidth/4)):
        # Text between 1/2 and 3/4 
        score = remap(TextPos, docWidth/2, 3 * (docWidth/4), -1, 1)
    elif (TextPos < (docWidth)):
        # Text between 3/4 and 1
        score = remap(TextPos, 3 * (docWidth/4), docWidth, 1, 0)
    else:
        # Error position, don't count it
        score = None
        #print ("Text position " + str(TextPos) + " greater than PDF width " + str(docWidth))
    
    return score

def Get_PDF_Files(input_path):
    filePaths = []
    brokenFiles = []
    if not os.path.exists(input_path):
            raise FileNotFoundError(input_path)
    if os.path.isdir(input_path):
        for root, dirs, files in tqdm(os.walk(input_path), desc="Reading files"):
            for file in files:
                if file.endswith(".pdf"):
                    if (is_pdf_corrupted_basic(os.path.join(root, file))):
                        brokenFiles.append(os.path.join(root, file))
                    else:
                        filePaths.append(os.path.join(root, file))
    return filePaths, brokenFiles

def binarize_img(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # conversion to grayscale image
        # use cv2 threshold binarization
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img

def alpha_to_color(img, alpha_color=(255, 255, 255)):
    if len(img.shape) == 3 and img.shape[2] == 4:
        B, G, R, A = cv2.split(img)
        alpha = A / 255

        R = (alpha_color[0] * (1 - alpha) + R * alpha).astype(np.uint8)
        G = (alpha_color[1] * (1 - alpha) + G * alpha).astype(np.uint8)
        B = (alpha_color[2] * (1 - alpha) + B * alpha).astype(np.uint8)

        img = cv2.merge((B, G, R))
    return img

def preprocess_image(_image, inv, bin, alpha_color=(255, 255, 255)):
    _image = alpha_to_color(_image, alpha_color)
    if inv:
        _image = cv2.bitwise_not(_image)
    if bin:
        _image = binarize_img(_image)
    return _image

def Process_PDF_To_Image(pdf_path, config: OCR_Config):
    """
    Converts a PDF to images
    """
    imgs = []
    
    with fitz.open(pdf_path) as pdf:
        for pg in range(0, pdf.page_count):
            page = pdf[pg]

            # for containedImage in containedImglist:
            #     inBuiltImagesBbox.append(page.get_image_bbox(containedImage))

            pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img = preprocess_image(img, False, True)

            imgs.append(img)
        shape = imgs[0].shape

    return imgs, shape

def Box_Size_Check(line, pdf_width, config: OCR_Config):
    """
    Returns true if line horizontal size occupies more than percentage of page width
    """
    lineWidth = abs(line[1][0] - line[0][0])
    if (lineWidth/pdf_width > config.minLineWidth):
        return True

    return False

def Get_Table_Positions(imgs:np.ndarray):
    """
    Returns list of lists of intervals representing table location
    [[x0,x1],[y0,y1]]
    """

    allTableIntervals = []

    for img in imgs:
        tables = Extract_Table_Meta_Info(img)
        tableIntervals = []

        if (len(tables) == 0):
            # No tables were found
            allTableIntervals.append(tableIntervals)
            continue

        for table in tables.keys():
            x0 = table[0]
            x1 = table[2]
            y0 = table[3]
            y1 = table[1]
            tableIntervals.append([[x0,x1],[y0,y1]])
        
        allTableIntervals.append(tableIntervals)

    return allTableIntervals

def Cluster_Check(lineCentreXCoords, PDF_width):
    """
    Given a list of X-coordinates, cluster them and check if they seem like a double column layout
    """

    lowerWidth = PDF_width * 0.4
    higherWidth = PDF_width * 0.6
    halfWidth = PDF_width/2

    # Separate lines into 2 clusters
    k = 2

    clusters, centroids = kmeans1d.cluster(lineCentreXCoords, k)

    # If both centroids are towards one side of the page, fail this set of lines

    if (centroids[0] < higherWidth and centroids[1] < higherWidth) or (centroids[0] > lowerWidth and centroids[1] > lowerWidth):
        return False

    return True

def PDF_To_Bboxes(pdf_path, config: OCR_Config):
    """
    Extract bbox information of all text on pages

    Output: [texts]
    texts: [BoundBoxInformation]
    BoundingBoxInformation: [[TopLeftX, TopLeftY],[TopRightX, TopRightY],[BottomRightX, BottomRightY],[BottomLeftX, BottomLeftY]]
    """

    bboxes = []

    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            pageBboxes = []
            blocks = page.get_textpage().extractBLOCKS()
            for block in blocks:
                x0, y0, x1, y1 = block[0], block[1], block[2], block[3]
                boundingBox = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                pageBboxes.append(boundingBox)
            bboxes.append(pageBboxes)

    return bboxes

def GetBoxCentre(BoundingBox):
    # Returns the centre of a rectangle box
    CentreX = (BoundingBox[0][0] + BoundingBox[1][0])/2
    CentreY = (BoundingBox[0][1] + BoundingBox[3][1])/2

    return (CentreX, CentreY)

def Score_Bbox_Results(texts, pdf_bounds, all_table_intervals, config: OCR_Config):
    """
    Analyses bbox results
    """

    scores = []
    PDF_width = 0

    if (len(texts) == 0):
        # If straight up no texts were detected
        return [-1]

    PDF_width = pdf_bounds[1]

    numTextsInTables = 0
    validTexts = 0
    lineCentreXCoords = []

    for pageNum, page in enumerate(texts):
        for line in page:
            if (Box_Size_Check(line, PDF_width, config)):
                # If text is big enough on the page
                validTexts += 1
                lineCentreX, lineCentreY = GetBoxCentre(line)
                if (len(all_table_intervals[pageNum]) != 0):
                    for table_interval in all_table_intervals[pageNum]:
                        if (Is_Within_Interval([lineCentreX,lineCentreY], table_interval)):
                            # Text is within a table, add to counter and continue
                            numTextsInTables += 1
                            continue

                lineCentreXCoords.append(lineCentreX)
                score = Line_Scoring_Rule_1(lineCentreX, PDF_width)
                if (score):
                    scores.append(score)

    if (validTexts < 10):
        # Not enough information/texts on pages
        return [-1]
    if (numTextsInTables/validTexts > config.tableThreshold):
        # If too many texts are in tables
        return [-1]
    if (not Cluster_Check(lineCentreXCoords, PDF_width)):
        # If the lines fail the cluster check
        return [-1]
    
    return scores

def Column_Layout_Detect(filepath, config):
    # Process pdf to image
    imgs, pdf_bounds = Process_PDF_To_Image(filepath, config)
    
    all_table_intervals = Get_Table_Positions(imgs)
        
    # Process image through OCR
    bboxOutput = PDF_To_Bboxes(filepath, config)

    # Find score
    scores = Score_Bbox_Results(bboxOutput, pdf_bounds, all_table_intervals, config)
    columnScore = np.mean(scores)

    del bboxOutput

    # Compare score to threshold
    if (columnScore > config.scoreThreshold):
        return ([filepath, True, columnScore])
    else:
        return ([filepath, False, columnScore])    

def Column_Layout_Detect_Wrapper(prepped_filepaths):
    # Hard code timeout values, ensure function keeps running
    maxWaitTime = 5
    filepath = prepped_filepaths[0]
    try:
        return func_timeout.func_timeout(maxWaitTime, Column_Layout_Detect, prepped_filepaths)
    except func_timeout.FunctionTimedOut:
        pass
    return ([filepath, False, -1])    

def Parallel_Process_Files(filepaths, config):
    """
    Perform layout detection for every PDF file in directory
    Output: [[filename, isDoubleColumn, finalScore],...]
    filename <- str
    isDoubleColumn <- bool
    finalScore <- float
    """

    results = []

    prepped_filepaths = [(x, config) for x in filepaths]

    results = process_map(Column_Layout_Detect_Wrapper, prepped_filepaths, max_workers=config.workers, chunksize=config.chunksize, total=len(prepped_filepaths), desc="Analysing files")

    return results

def is_scanned_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    # Check up to first 4 pages
    maxPageNum = min(doc.page_count, 5)
    
    for page in doc[:maxPageNum]:
        if page.get_text("text") != "":
            return False
    
    doc.close()

    return True

def pdf_is_encrypted(pdf_path):
    pdf = fitz.Document(pdf_path)
    return pdf.is_encrypted

def is_pdf_corrupted_basic(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        doc.close()
        del doc
    except:
        return True

    # if (len(doc.pages) == 0):
    #     return True

    return False

def get_size(file_path, unit='bytes'):
    file_size = os.path.getsize(file_path)
    exponents_map = {'bytes': 0, 'kb': 1, 'mb': 2, 'gb': 3}
    if unit not in exponents_map:
        raise ValueError("Must select from \
        ['bytes', 'kb', 'mb', 'gb']")
    else:
        size = file_size / 1024 ** exponents_map[unit]
        return round(size, 3)

def Analyse_Scanned_Files(filepaths, move, outputDir):
    scannedOutputDir = os.path.join(outputDir, "Scanned")

    os.makedirs(scannedOutputDir, exist_ok=True)

    newFilepaths = []

    for filepath in tqdm(filepaths, desc="Moving scanned files"):
        result = Move_Scanned_File_Wrapper(filepath, move, outputDir)
        if (result):
            newFilepaths.append(result)

    return newFilepaths

def Move_Scanned_File(filepath, move ,scannedOutputDir):
    if (pdf_is_encrypted(filepath)):
        return None

    if (is_scanned_pdf(filepath)):
        if (move):
            try:
                shutil.copy(filepath, scannedOutputDir)
            except:
                print("Failed to copy file " + filepath)
                pass
            return None
    else:
        return filepath

def Move_Scanned_File_Wrapper(filepath, move, outputDir):
    # Hard code timeout values, ensure function keeps running
    maxWaitTime = 5
    scannedOutputDir = os.path.join(outputDir, "Scanned")
    try:
        return func_timeout.func_timeout(maxWaitTime, Move_Scanned_File, (filepath, move, scannedOutputDir))
    except func_timeout.FunctionTimedOut:
        pass
    return None   


def Move_Files(results, moveDoube, moveOther, outputDir):
    otherFilesDir = os.path.join(outputDir, "Other")
    doubleColumnOutputDir = os.path.join(outputDir, "DoubleColumn")

    os.makedirs(otherFilesDir, exist_ok=True)
    os.makedirs(doubleColumnOutputDir, exist_ok=True)

    for filepath, isDoubleColumn, score in results:
        if (isDoubleColumn and moveDoube):
            try:
                shutil.copy(filepath, doubleColumnOutputDir)
            except:
                print("Failed to copy file " + filepath)
        elif (moveOther):
            try: 
                shutil.copy(filepath, otherFilesDir)
            except:
                print("Failed to copy file " + filepath)
    return

def Batch_Process_Files(filepaths, batchSize, moveScanned, moveDouble, moveOther, config:OCR_Config, outputDir):
    """
    Splits the filepaths into batches, and sends them to parallel processing
    """
    counter = 0
    print("Starting processing in batches of " + str(batchSize))

    for batch in batched(filepaths, batchSize):
        batchValidFiles = Analyse_Scanned_Files(batch, moveScanned, outputDir)

        results = Parallel_Process_Files(batchValidFiles, config)
        Move_Files(results, moveDouble, moveOther, outputDir)

        counter += len(batch)
        print("Processed " + str(counter) + " files")

    return

def main(args):
    print(args)
    input = args.input
    outputDir = args.outputDir

    config = OCR_Config(args.minLineWidth, 
                        args.minConfidence, 
                        args.scoreThreshold, 
                        args.tableThreshold,
                        args.maxPagesToScan, 
                        args.workers, 
                        args.chunksize, 
                        args.resizeFactor)

    timer = TimeRecorder()

    timer.start()

    # paddle.utils.run_check()

    # Testing specific files
    # testFile = r"C:\Everything\CNStateGrid\OCR Column Layout Detection\UnsortedData\TestData2\3149_50789419a6c54baf8335b5e290ea3d84_城市轨道交通工程投资控制和工程造价管理探讨.pdf"
    # # PDF_OCR_Scan_PDF(testFile, config)
    # print(Column_Layout_Detect([testFile, config]))

    filepaths, brokenFiles = Get_PDF_Files(input)

    startAt = args.startAt

    if (args.startAt <= 0):
        print("Analysing all files, total=" + str(len(filepaths)))
        Batch_Process_Files(filepaths, args.batchsize, args.moveScanned, args.moveDouble, args.moveOther, config, outputDir)
    elif (startAt < len(filepaths)):
        print("Starting at file " + str(startAt) + ", total=" + str(len(filepaths)- startAt))
        Batch_Process_Files(filepaths[args.startAt:], args.batchsize, args.moveScanned, args.moveDouble, args.moveOther, config, outputDir)
    else:
        print("Start at index invalid")

    #filepaths = Analyse_Scanned_Files(filepaths, False, outputDir)

    # results = Parallel_Process_Files(filepaths, config)

    # Move_Files(results, True, outputDir)

    # for entry in results:
    #     print(entry)

    # print("Broken PDF files: " +  str(brokenFiles))

    print("Done.", timer.record())
    print(timer.total())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Column Layout Detector")

    parser.add_argument(
        "--outputDir", "-O", default="ResultOutput", help="Output result directory"
    )

    parser.add_argument(
        "--minLineWidth", type=float, default=0.3, help="mininum number of characters to be considered a line of text"
    )
#
    parser.add_argument(
        "--minConfidence", type=float, default=0.9, help="mininum confidence of text recognition to be considered"
    )  

    parser.add_argument(
        "--scoreThreshold", type=float, default=0.3, help="Score threshold to decide if document is multi-column"
    )  

    parser.add_argument(
        "--tableThreshold", type=float, default=0.5, help="Threshold to determine if pages are too filled with tables"
    )  

    parser.add_argument(
        "--maxPagesToScan", type=int, default=2, help="Max number of pages to perform OCR on a single document"
    )

    parser.add_argument(
        "--workers", type=int, default=4, help="Number of workers"
    )

    parser.add_argument(
        "--batchsize", type=int, default=2000, help="batchsize"
    )

    parser.add_argument(
        "--chunksize", type=int, default=20, help="chunksize"
    )
   
    parser.add_argument(
        "--resizeFactor", default=1.5, help="Resize factor of images for OCR processing"
    )

    parser.add_argument(
        "--startAt", type=int, default=0, help="At which index of filepaths to start at, use if previously stopped for whatever reason and need to continue from there"
    )

    parser.add_argument(
        "--moveScanned", type=str2bool, default=True, help="Copy scanned files into results"
    )

    parser.add_argument(
        "--moveDouble", type=str2bool, default=True, help="Copy double columned files into results"
    )

    parser.add_argument(
        "--moveOther", type=str2bool, default=True, help="Copy other files into results"
    )
    
    parser.add_argument("input", help="Input directory")

    args = parser.parse_args()

    main(args)
