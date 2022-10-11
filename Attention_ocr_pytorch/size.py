import os
import os.path
from PIL import Image
import cv2
'''
filein: 输入图片文件夹
fileout: 输出图片文件夹
width: 输出图片宽度
height:输出图片高度
type:输出图片类型（png, gif, jpeg...）
'''
def ResizeImage(filein, fileout,  height, type):
    for image in os.listdir(filein):
        fullFile = os.path.join(filein, image) 
        img = Image.open(fullFile)
        Old_height, Old_width = img.height, img.width
       # 等比例缩放尺度。
        scale =Old_height/height   # 1
       # 获得相应等比例的图像宽度。
        width= 280#int(Old_width/scale)  # 2
        img = img.resize((width, height),Image.ANTIALIAS)
        name, extension = os.path.splitext(image)
        name= os.path.join(name+'.' +type)
        print(name)
        img = img.convert('RGB')
        img.save(fileout+os.sep+name)

if __name__ == "__main__":
    filein = r'data/val'
    fileout = r'data/val'
    height = 32
    type = 'jpg'
    ResizeImage(filein, fileout, height, type)      
