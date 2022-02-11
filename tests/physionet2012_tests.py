import tensorflow as tf
import tensorflow_datasets as tfds

import medical_ts_datasets

def print_dataset_len(dataset_name):
    train_ds, train_ds_info = tfds.load(dataset_name,with_info=True, split=tfds.Split.TRAIN)     
    val_ds, val_ds_info = tfds.load(dataset_name,with_info=True, split=tfds.Split.VALIDATION) 
    test_ds, test_ds_info = tfds.load(dataset_name,with_info=True, split=tfds.Split.TEST) 

    print("train_size: {}".format(train_ds.cardinality()))
    print("val_size: {}".format(val_ds.cardinality()))
    print("test_size: {}".format(test_ds.cardinality()))

if __name__ == "__main__":
    print_dataset_len('physionet2012')
    print_dataset_len('physionet2012_subsampled')