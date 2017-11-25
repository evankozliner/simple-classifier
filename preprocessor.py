import pandas as pd
import numpy as np
import PIL
import time
import math
from PIL import Image,ImageFile
from sklearn.decomposition import IncrementalPCA

# Median dimensions are 
# width: 1024, height: 768
# mean dimensions are
# width: 2061 height: 1410
#HEIGHT      = 768
#WIDTH       = 1024

# Median dimensions for the lesions after segmentation
HEIGHT = 510
WIDTH = 766

CHANNELS    = 3
DATA_DIR = 'bound_box_final_2/'
N_COMPS = 10
BATCH_SIZE=100

def timer(f):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        f(*args, **kwargs)
        t2 = time.time()
        print "Function took {} seconds. \n".format(str(t2-t1))
    return wrapper

def build_dataset(dataset, ignore_set):
    print "Building dataset: {}".format(dataset)
    data = pd.read_csv(dataset + '.csv')

    # Images not reliably complete
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Flattened image vector is of shape HEIGHT*WIDTH*CHANNELS
    # Add 1 dimension to vector for class
    data_matrix = np.zeros((data.shape[0], HEIGHT * WIDTH * CHANNELS + 1))

    for idx,row in data.iterrows():
        if row[0] + ".jpg" in ignore_set or not "isic" in row[0]:
            print "ignoring " + row[0] + ".jpg"
            continue
        img = Image.open(DATA_DIR + row[0] + '.jpg')
        resized_img = img.resize((HEIGHT, WIDTH), PIL.Image.ANTIALIAS)
        image_vector = np.array(resized_img).flatten()
        image_vector_with_class = np.concatenate(\
                (image_vector, [row['melanoma']]), axis=0)
        data_matrix[idx] = image_vector_with_class

        if idx % 50 == 0:
            print "{} images done...".format(idx)

    return data_matrix

def get_next_batch(data, i, BATCH_SIZE):
    if (i + 1) * BATCH_SIZE > data.shape[0]:
        return data[i*BATCH_SIZE:data.shape[0],:]
    return data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE,:]

def reduce_dimensionality(dataset):
    print "Reducing dimensionality for dataset: {}".format(dataset)
    print "Loading {}".format(dataset)
    data_with_class = np.load(dataset)
    print "Finished Loading {}".format(dataset)
    data_no_class, y = extract_features_and_class(data_with_class)

    # Will Seg-fault with regular PCA due to dataset size
    # Somewhat arbitrary batch size here. 
    pca = IncrementalPCA(n_components=N_COMPS)
    num_batches = int(math.ceil(y.shape[0] / float(BATCH_SIZE)))

    print "Beggining to fit dataset"
    for i in xrange(num_batches):
        batch = get_next_batch(data_no_class, i, BATCH_SIZE) 
        pca.partial_fit(batch)
        if i % 10 == 0:
            print "{}% complete.".format( float(i) / num_batches * 100)

    print "Beggining to fit transform dataset"
    reduced_data = None
    for i in xrange(num_batches):
        batch = get_next_batch(data_no_class, i, BATCH_SIZE) 
        transformed_chunk = pca.transform(batch)
        if reduced_data == None:
            reduced_data = transformed_chunk
        else:
            reduced_data = np.vstack((reduced_data, transformed_chunk))
        if i % 10 == 0:
            print "{}% complete.".format(float(i) / num_batches * 100)

    #reduced_data = pca.fit_transform(data_no_class)

    print "PCA complete for {} components. Explained variance: {}".\
            format(N_COMPS, np.sum(pca.explained_variance_ratio_))
    print reduced_data.shape
    print y.shape
    #reduced_data = pca.transform(data_no_class)
    reduced_data_with_class = np.hstack((reduced_data,y))
    return reduced_data_with_class

def extract_features_and_class(data_with_class):
    y = data_with_class[:,-1]
    # Reshape into column vector instead of row
    y_col = y.reshape(y.size,1)
    n_columns = data_with_class.shape[1] - 1
    data_no_class = data_with_class[:,0:n_columns]
    return data_no_class, y_col

def get_ignored_images():
    with open("failed_thres_img.txt") as f: 
        a = map(lambda x: x.split("\n")[0], f.readlines())
    return set(a)

@timer
def build_and_write_dataset(dataset):
    dataset_file = dataset + ".npy"

    ignore_set = get_ignored_images()

    dataset_matrix = build_dataset(dataset, ignore_set)
    print "Saving {}".format(dataset_file)
    np.save(dataset_file, dataset_matrix)

@timer
def build_and_write_reduced_dataset(dataset):
    # TODO format with num of 
    reduced_dataset_file = dataset + "_reduced.npy"
    reduced_matrix = reduce_dimensionality(dataset + '.npy')
    print "Saving {}".format(reduced_dataset_file)
    np.save(reduced_dataset_file, reduced_matrix)

def main(dataset):
    build_and_write_dataset(dataset)
    build_and_write_reduced_dataset(dataset)

if __name__ == "__main__":
    if BATCH_SIZE < N_COMPS:
        # See https://github.com/scikit-learn/scikit-learn/issues/6452
        raise ValueError("Number of components must be < \
                batch size.")
    main('final_isic_thres')

