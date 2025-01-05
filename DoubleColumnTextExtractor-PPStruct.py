"""
双栏解析

Author: Roger Zhang
Last Modified: 2024/06/08

Description:
Generate JSON files from double column layout PDF files

Update:
- 2024/05/27 - Creation
- 2024/05/28 - Use GPU
- 2024/05/30 - To JSON conversion

使用:
- 在 cmd/console 里, 跑这个 script 时，添加文件夹 path。
增加/编辑：
- 这个 script 主要功能是控制 PDF->Image->PPStruct->JSON 流程, 如果需要改 JSON 输出格式和建 section 逻辑, 最好在 VieOCRBlockProcessing.py 里更改。 
"""
# -*- coding: utf-8 -*-
import fitz
fitz.TOOLS.mupdf_display_errors(False)
from PaddleOCR import PPStructure
# import pkg_resources
# from symspellpy.symspellpy import SymSpell
import os
import argparse
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
from TimeRecorder import TimeRecorder
from itertools import islice
import cv2
import json
import random
import re
import datetime
from PIL import Image
from VieOCRBlockProcessing import VieOCRBlockProcessor


@dataclass
class Config:
    # Number of files to be processed in one cycle
    fileBatchsize: int
    # mininum confidence of text recognition to be considered
    minConfidence: float

def batched(iterable, chunk_size):
    iterator = iter(iterable)
    while chunk := tuple(islice(iterator, chunk_size)):
        yield chunk

def Euclidean_Distance(coord1, coord2):
    """
    Returns the euclidean distance between two coordinates
    """
    point1 = np.array(coord1)
    point2 = np.array(coord2)
    
    # calculating Euclidean distance
    # using linalg.norm()
    dist = np.linalg.norm(point1 - point2)

    return dist

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

def Convert_PDF_To_Images(pdf_path):
    imgs = []
    with fitz.open(pdf_path) as pdf:
        for pg in range(0, pdf.page_count):
            page = pdf[pg]

            # for containedImage in containedImglist:
            #     inBuiltImagesBbox.append(page.get_image_bbox(containedImage))

            mat = fitz.Matrix(2, 2)
            pm = page.get_pixmap(matrix=mat, alpha=False)

            # if width or height > 2000 pixels, don't enlarge the image
            if pm.width > 2000 or pm.height > 2000:
                pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img = preprocess_image(img, False, True)

            imgs.append(img)
        shape = imgs[0].shape
        return imgs, shape, pdf.page_count

def Convert_All_PDF_Files_To_Images(filepaths):
    """
    Given list of PDF filepaths, convert each to images in a dictionary of
    {ConvertedFiles[filename] = [[shape, original_filepath], [img1, img2 ...]]}
    """
    ConvertedFiles = {}

    for file in filepaths:
        filename = os.path.basename(file)
        ConvertedFiles[filename] = []
        pdfImages, shape, pageCount = Convert_PDF_To_Images(file)
        ConvertedFiles[filename].append([shape, file])
        ConvertedFiles[filename].append(pdfImages)

    return ConvertedFiles
    """
    LineInfo is of type [lineLeftSideXPosition, lineCentreYPosition]
    """
    footerCentreXPos = footerLineCentre[0]

    if (lineInfo[1] < footerLineCentre[1]):
        #Line is not under footer
        return False
    else:
        if (footerCentreXPos < pageWidth * 0.4 and lineInfo[0] < pageWidth * 0.4):
            # Line belongs to a left side footer
            return True
        elif (footerCentreXPos > pageWidth * 0.6 and lineInfo[0] > pageWidth * 0.6):
            # Line belongs to a right side footer
            return True
        elif (pageWidth * 0.4 < footerCentreXPos < pageWidth * 0.6):
            # Footer is centre justified
            return True
    return False

def GetBoxCentre(BoundingBox):
    # Returns the centre of a rectangle box
    CentreX = (BoundingBox[0][0] + BoundingBox[1][0])/2
    CentreY = (BoundingBox[0][1] + BoundingBox[3][1])/2

    return (CentreX, CentreY)

def GetBoxHeight(BoundingBox):
    # Returns the height of a rectangle box
    height = abs(BoundingBox[0][1] - BoundingBox[3][1])

    return height  

def OCR_Scan_Images(images, table_engine, config: Config):
    """
    Pass images through OCR, output a result array
    Result is separated by pages
    Each page has lines

    """

    pages = []

    for image in images:
        result = table_engine(image)
        lines = []
        for line in result:
            line.pop('img')
            lines.append(line)
        pages.append(lines)

    return pages
                    
def Post_Process_Pages(fileResults):
    """
    Puts ocr results of a single file into reading order, then converts each labeled area into appropriate plain text
    """
    [shape, filepath], images, allPages = fileResults

    readyAllPages = []

    vieBlock = VieOCRBlockProcessor()

    for pageNum, page in enumerate(allPages):
        # First sort page by y-level
        page = sorted(page, key=lambda x: x['bbox'][1])

        # Then order by left, right and centre columns, and process each block
        LeftAndCentre = []
        Right = []
        # Boolean to track if next block is to be skipped, mostly because it will be already processed
        skipNextBlock = False

        # Iterate through each block in page
        for blockNum, block in enumerate(page):
            if (skipNextBlock):
                skipNextBlock = False
                continue

            # Check if block is table and/or table caption
            if (block['type'] == "table" or block['type'] == "table_caption"):
                # Look ahead for table or table caption
                if (blockNum+1 < len(page) and (page[blockNum+1]['type'] == "table" or page[blockNum+1]['type'] == "table_caption")):
                    convertedBlocks = vieBlock.Convert_OCR_Block_Tables_And_Caption(block, page[blockNum+1])
                    skipNextBlock = True
                else:
                    convertedBlocks = vieBlock.Convert_OCR_Block_Tables_And_Caption(block, None)
            else:         
                convertedBlocks = vieBlock.Convert_OCR_Block_Standard(block, pageNum)

            # Put into order based on column
            if ((block['bbox'][2] + block['bbox'][0])/2 < int(shape[1] * 0.55)):
                LeftAndCentre += convertedBlocks
            else:
                Right += convertedBlocks
        
        # Combine processed blocks into reading order
        page = LeftAndCentre + Right
        readyAllPages += page

    return readyAllPages

def Process_PDF_OCR_Results(ocrResults, outputDir):
    """
    Puts all ocr results into reading order, and create their respective folders, then converts each labeled area into appropriate plain text
    """
    readyToWrite = {}

    for filename in ocrResults.keys():
        # Create directory for file
        baseFilename = re.sub(".pdf", "", filename)
        fileOutputDir = os.path.join(outputDir, baseFilename)
        os.makedirs(fileOutputDir, exist_ok=True)
        
        ready = Post_Process_Pages(ocrResults[filename])
        readyToWrite[filename] = (ocrResults[filename][0][1], fileOutputDir, ready)

    # readyToWrite[filename] = (fileInputPath, fileOutputDirectory, processedBlocks)
    return readyToWrite

# OCR TO JSON
def PDF_Blocks_To_JSON(inputPath, outputDir, blocks, filename):
    
    fileDic = {}
    fileDic["datatype"] = filename
    fileDic['id'] = random.randint(0000000000, 9999999999)
    fileDic['title'] = '{}-{}.json'.format(filename, datetime.date.today())

    content_list = []
    
    currentSectionID = 1
    currentSectionTitle = ""
    currentSectionContent = ""

    hasReachedFirstTitle = False

    with fitz.open(inputPath) as pdf:
        # block = ["type", "content"] or block = ["image", index, pageNum, bbox]
        for block in blocks:
            # First title rules
            if (not hasReachedFirstTitle and block[0] != "title"):
                # Texts that are before first title, ignore
                continue
            elif (not hasReachedFirstTitle and block[0] == "title"):
                hasReachedFirstTitle = True
                currentSectionTitle = block[1]
                continue
            
            if (block[0] == "title" or block[0] == "body"):
                # Special ignore rules
                if (re.search(r"参考文献", block[1])):
                    # 参考文献之后不需要
                    finalBody = block[1].split("参考文献")[0]
                    currentSectionContent+=finalBody
                    break

                if (re.search(r"doi.{0,2}:", block[1])):
                    #doi 信息不需要
                    continue

            if (block[0] == "title"):
                # New section encountered, add previous section
                content_dic = {
                    'origin_title': currentSectionTitle,
                    'section_content': currentSectionContent,
                    'section_title': "",
                    'section_id': currentSectionID,
                    'rate': "1", 'title_level': 1
                }
                content_list.append(content_dic.copy())
                
                # Set new
                currentSectionTitle = block[1]
                currentSectionID += 1
                currentSectionContent = ""
            elif (block[0] == "body"):
                currentSectionContent+=block[1]
            elif (block[0] == "image"):
                # Crop and save image from original PDF
                imgIndex = block[1]
                pageObject = pdf[block[2]]
                bbox = block[3]
                mat = fitz.Matrix(2, 2)
                pm = pageObject.get_pixmap(matrix=mat, alpha=False)
                # if width or height > 2000 pixels, don't enlarge the image
                if pm.width > 2000 or pm.height > 2000:
                    pm = pageObject.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
                img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                croppedImage = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
                image2 = Image.fromarray(croppedImage) 
                imageFilename = re.sub(".pdf", "".join(("-", str(imgIndex), ".png")), filename)
                imageFilepath = os.path.join(outputDir, imageFilename)
                image2.save(imageFilepath)

                # Add words in content to indicate image and path
                indicator = "".join(("{", imageFilename, "}\n"))
                currentSectionContent+=indicator
            else:
                print("Unknown converted block type")

        # Add final section if any
        if (len(currentSectionTitle) > 0):
            content_dic = {
                    'origin_title': currentSectionTitle,
                    'section_content': currentSectionContent,
                    'section_title': "",
                    'section_id': currentSectionID,
                    'rate': "1", 'title_level': 1
                }
            content_list.append(content_dic.copy())
        
        fileDic['content'] = content_list
        newFilename = re.sub(".pdf", ".json", filename)
        newFilepath = os.path.join(outputDir, newFilename)
        with open(newFilepath, "w", encoding='utf-8') as file:
            json.dump(fileDic, file, ensure_ascii=False)

    return

def Process_PDF_Files(filepaths, table_engine, config:Config):
    ConvertedFiles = Convert_All_PDF_Files_To_Images(filepaths)

    allPages = []

    # Process each file
    for filename in ConvertedFiles.keys():
        ocrResults = OCR_Scan_Images(ConvertedFiles[filename][1], table_engine, config)
        ConvertedFiles[filename].append(ocrResults)

    return ConvertedFiles

def Write_Files(results):
    for filename in results.keys():
        PDF_Blocks_To_JSON(results[filename][0], results[filename][1], results[filename][2], filename)

def Batch_Process_PDF_Files(filepaths, config:Config, outputDir):
    """
    Splits the filepaths into batches, and sends them to processing
    """
    counter = 0
    print(" ".join(("Starting processing with batch setting", str(config.fileBatchsize))))

    # Setup OCR
    table_engine = PPStructure(show_log=False, use_gpu=True)

    for batch in batched(filepaths, config.fileBatchsize):
        ocrResults = Process_PDF_Files(batch, table_engine, config)
        readyToWrite = Process_PDF_OCR_Results(ocrResults, outputDir)
        Write_Files(readyToWrite)
        counter += len(batch)
        print("Processed " + str(counter) + " files")

    return

def is_pdf_corrupted_basic(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        doc.close()
        del doc
    except:
        return True
    
    return False

def main(args):
    print(args)
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    input = args.input
    outputDir = args.outputDir
    
    config = Config(50, 0.7)

    timer = TimeRecorder()

    timer.start()

    filepaths, brokenFiles = Get_PDF_Files(input)

    os.makedirs(outputDir, exist_ok=True)

    Batch_Process_PDF_Files(filepaths, config, outputDir)

    print("Done.", timer.record())
    print(timer.total())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Double Column to JSON")
    parser.add_argument(
        "--outputDir", "-O", default="ResultOutput", help="Output result directory"
    )
    parser.add_argument("input", type=str, help="Input directory")
    args = parser.parse_args()

    main(args)