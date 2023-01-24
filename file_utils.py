import os
import h5py
import numpy as np

path = '/scratch/midway2/dhayaa/tmp_test.h5'

with h5py.File(path, "w") as f:

    #Create all columns you need
    f.create_dataset('Mcal', data = [], chunks=(10**4,), maxshape = (None,))

    #Helper function. Just improves readability
    #Appends new_data array into existing dataset
    def add_data(dataset, new_data):

        dataset.resize(dataset.shape[0] + len(new_data), axis=0)
        dataset[-len(new_data):] = new_data

    #Open mcal files iteratively
    for i in range(3):

        #Get dataset
        mcal_fits = np.random.random(100_000)

        #Append to columns
        add_data(f['Mcal'], mcal_fits)

        print(f['Mcal'].shape, np.array(f['Mcal']))


if name == '__main__':

    pass
