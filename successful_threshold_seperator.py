import numpy as np
import pandas as pd
from PIL import Image
from shutil import copyfile 

THRES_DIR = "thresholded_final/out_final/"
DATA_DIR = "final/"
OUT_DIR = "success_threshold_final/"
PERCENTAGE_WHITE = 1

def main():
    df = pd.read_csv("final.csv")
    failed_img_notes = open("failed_thres_img.txt", 'w+')
    for idx,row in df.iterrows():
        print row[0]
        thres_img = np.array(Image.open(THRES_DIR + row[0] + '_bin.bmp'))
        if (image_is_not_blank(thres_img)):
            copyfile(DATA_DIR + row[0] + ".jpg", OUT_DIR + row[0] + ".jpg")
        else:
            failed_img_notes.write(row[0] + ".jpg\n")

    failed_img_notes.close()

def image_is_not_blank(thres_img):
    flattened = thres_img.flatten()
    return np.sum(flattened) / float(flattened.size) * 100 > PERCENTAGE_WHITE

if __name__ == "__main__":
    main()
