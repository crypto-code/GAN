from PIL import Image
import argparse
import os

def main(src, dst):
  im = Image.open(src)
  k=0
  if not os.path.exists(dst):
        os.mkdir(dst)
  for i in range(0,1024,128):
    for j in range(0,1024,128):
      imnew = im.crop((i,j,i+128,j+128))
      imnew =imnew.resize((512,512), Image.BICUBIC)
      imnew.save((r"%s\[%d].jpg" % (dst,k)), quality=100)
      k=k+1


if __name__=="__main__":
  parser = argparse.ArgumentParser(description='Crop Individual Images')
  parser.add_argument('--input',type=str,required=True,help='Input Image')
  parser.add_argument('--output',type=str,required=True,help='Output Directory')
  args = parser.parse_args()
  main(args.input, args.output)
