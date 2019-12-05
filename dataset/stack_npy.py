'''
This script stacks all files and create a single file
'''
import numpy as np


def main():
    total = np.load('train_image1.npy')
    for i in range(2, 3):
        temp = np.load('train_image' + str(i) + '.npy')
        total = np.vstack((total, temp))  # (120, 256, 256, 3)
    print(total.shape)

    np.save('final_train.npy', total)

    total = np.load('mask_image1.npy')
    for i in range(2, 3):
        temp = np.load('mask_image' + str(i) + '.npy')
        total = np.vstack((total, temp))  # (120, 256, 256, 1)
    print(total.shape)

    np.save('final_mask.npy', total)


if __name__ == '__main__':
    main()
