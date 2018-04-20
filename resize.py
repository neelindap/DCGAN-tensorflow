import PIL
from PIL import Image
import os


basewidth = 64
hsize = 64
dir = os.path.join(os.path.dirname(__file__),'data/Simpsons_new')
new_path = os.path.join(os.path.dirname(__file__),'data/Simpsons_new_64',)
i=0

for file in os.listdir(dir):
    filename = os.fsdecode(os.path.join(dir,file))

    img = Image.open(filename)
    #wpercent = (basewidth / float(img.size[0]))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    img.save(os.path.join(new_path,str(i)+".jpg"))
    i += 1