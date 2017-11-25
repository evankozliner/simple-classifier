import pandas as pd
import subprocess
import os
import cv2

EXT = '.jpg'
DATA_DIR_THRES = 'thresholded_final/out_final/'
DATA_DIR = 'final/'
OUT_DIR = 'bound_box_final_2/'
#DATA_DIR = 'final/'

def main():
    gt = pd.read_csv("final.csv")
    # iterate over ground truth
    for idx, row in gt.iterrows():
        # use threshold fusion to build directory structure for transfer learning
        bounded_box_img = get_bounding_box(row['image_id'], threshold=False)
        cv2.imwrite(OUT_DIR + row['image_id'] + '.jpg',bounded_box_img)
        if idx % 5 == 0:
            print "{}% Complete".format(idx / float(len(gt)) * 100)

def get_bounding_box(image_id, threshold=True):
    img_path = DATA_DIR + image_id + EXT
    if threshold:
        threshold_fusion_cmd = "./fourier_0.8/threshold_fusion " + img_path
        thresholded_img_name = OUT_DIR + image_id + "_bin.bmp"
        subprocess.call(threshold_fusion_cmd, shell = True)
    else:
        thresholded_img_name = DATA_DIR_THRES + image_id + "_bin.bmp"
    threshold_img = cv2.imread(thresholded_img_name, 0)
    original_img = cv2.imread(img_path)
    contours = cv2.findContours(threshold_img, 
            cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_SIMPLE)[0][0]
    x,y,w,h = cv2.boundingRect(contours)

    cv2.rectangle(original_img,(x,y), (x+w,y+h),3)
    #cv2.imshow('img',original_img[y:y+h,x:x+w])
    #cv2.waitKey(0)
    
    return original_img[y:y+h, x:x+w]

if __name__ == "__main__":
    #box = get_bounding_box('isic_0000009', threshold=False)
    main()
