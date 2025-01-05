"""
Author: Roger Zhang
Last Modified: 2024/06/06

Description:
Convert blocks of PPStructure output into appropriate content

Update:
- 2024/06/06 - Creation

Usage:
- Modify functions for each type of block
- All functions must have return format of: [[type, text], [type, text]...]
- where type = 'title' or 'body', which determines if this text is to be made a origin_title or put into section_content
- text = string, can be anything

PPStructure 会从 image 转换成 ["type", "res"] 格式的 list, 这个 .py 的功能是针对于各种 "type" 进行不同的转化
"title" -> origin_title
"body" -> section_content
"image" -> 保存一个新图片
"""

import re
from markdownify import markdownify as md

class VieOCRBlockProcessor(object):
    def __init__(self):
        self.imageIndex = 0

    def Convert_OCR_Block_Standard(self, block, pageNum):

        # Based on the type of block, send text to respective functions
        try:
            blockType = block['type']
        except:
            return ""
        
        if (not blockType):
            return ""
        
        if (blockType == "header"):
            return self.Convert_Header(block['res'])
        elif (blockType == "footer"):
            return self.Convert_Footer(block['res'])
        elif (blockType == "title"):
            return self.Convert_Title(block['res'])
        elif (blockType == "text"):
            return self.Convert_Text(block['res'])
        elif (blockType == "reference"):
            return self.Convert_Reference(block['res'])
        elif (blockType == "equation"):
            return self.Convert_Equation(block['res'])
        elif (blockType == "figure"):
            return self.Convert_Figure(block, pageNum)
        elif (blockType == "figure_caption"):
            return self.Convert_Figure_Caption(block['res'])
        else:
            print("Error! Unrecognised label: " + blockType)
            return [["body", ""]]

    def Convert_OCR_Block_Tables_And_Caption(self, block1, block2):
        # Converts table and table caption blocks into appropriate form

        title = "Table"
        body = ""

        block1Type = block1['type']
        if (block2):
            block2Type = block2['type']
            if (block1Type == "table_caption" and block2Type == "table"):
                # block1 is the title caption, block2 is a table
                title = self.Convert_Table_Caption(block1['res'])
                table = self.Convert_Table(block2['res']['html'])
                return [["title", title], ["body", table]]
            elif (block2Type == "table_caption" and block1Type == "table"):
                # block2 is the title caption, block1 is a table
                title = self.Convert_Table_Caption(block2['res'])
                table = self.Convert_Table(block1['res']['html'])
                return [["title", title], ["body", table]]
            elif (block1Type == "table" and block2Type == "table"):
                # Two tables
                table1 = self.Convert_Table(block1['res']['html'])
                table2 = self.Convert_Table(block2['res']['html'])
                return [["body", table1], ["body", table2]]  
            else:
                # Two captions, or something unknown, wtf? We just return both as body text
                for line in block1['res']:
                    body += line['text']
                for line in block2['res']:
                    body += line['text']
                return [["body", body]]
        else:
            if (block1Type == "table"):
                # No captions, make new section with default title
                table = self.Convert_Table(block1['res']['html'])
                return [["title", title], ["body", table]]
            else:
                # A table caption by itself, very strange, return it as a title
                title = self.Convert_Table_Caption(block1['res'])
                return [["title", title]]

    def Convert_Header(self, res):
        # We do not include headers
        return [["body", ""]]

    def Convert_Footer(self, res):
        # We do not include footers
        return [["body", ""]]

    def Convert_Title(self, res):
        # Make recognised titles as origin_titles, and mistaken parts into body
        title = ""
        body = ""
        for line in res:
            if (line["text"].count(",") >= 1 or line["text"].count("，") >= 1):
                # Commas appearing, possibly indicating author list
                body += line["text"]
            else:
                title += line['text']

        if (len(title) > 0):
            return [["title", title], ["body", body]]
        else:
            return [["body", body]]

    def Convert_Text(self, res):
        # Return the combined texts of a text block
        text = ""
        for line in res:
            text += line['text']
        return [["body", text]]

    def Convert_Reference(self, res):
        # We do not include references
        return [["body", ""]]

    def Convert_Figure(self, block, pageNum):
        try:
            bbox = block['bbox']
            y0 = int(bbox[1])
            y1 = int(bbox[3])
            x0 = int(bbox[0])
            x1 = int(bbox[2])

            self.imageIndex += 1
            return [["image", self.imageIndex, pageNum, (y0,y1,x0,x1)]]
        except:
            return [["body", "UNKNOWN FIGURE \n"]]

    def Convert_Figure_Caption(self, res):
        caption = ""
        for line in res:
            caption += line['text']
        return [["body", caption + "\n"]]

    def Convert_Table(self, html):
        # Converts the table HTML into plain text

        # Clear HTML and Body tags from string
        html = re.sub(r"<html>", "", html)
        html = re.sub(r"</html>", "", html)
        html = re.sub(r"<body>", "", html)
        html = re.sub(r"</body>", "", html)

        try:
            return md(html)
        except:
            print("Table conversion failed, returning raw html")
            return html

    def Convert_Table_Caption(self, res):
        title = ""
        for line in res:
            title += line['text']
        return title

    def Convert_Equation(self, res):
        # TODO
        return [["body", "\n UNKNOWN EQUATION \n"]]  