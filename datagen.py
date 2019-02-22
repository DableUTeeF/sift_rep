import cv2
import os
import numpy as np


class SiftGenerator:
    def __init__(self, rootdir):
        self.listfile = []
        for subdir in os.listdir(rootdir):
            try:
                for file in os.listdir(os.path.join(rootdir, subdir)):
                    if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                        self.listfile.append(os.path.join(rootdir, subdir, file))

            except NotADirectoryError:
                if subdir.endswith('.jpg') or subdir.endswith('.jpeg') or subdir.endswith('.png'):
                    self.listfile.append(os.path.join(rootdir, subdir))

        self.sift = cv2.xfeatures2d.SIFT_create()

    def __getitem__(self, idx):
        x = np.zeros((128, 64, 64, 3), dtype='float32')
        y = np.zeros((128, 128), dtype='float32')
        img = cv2.imread(self.listfile[idx])
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(img_gray, None)
        j = 0
        for i in range(len(kp)):
            if j >= 128:
                break
            point = kp[i].pt
            if point[0] < 32 or point[1] < 32 or point[0] > img.shape[1]-32 or point[1] > img.shape[0]-32:
                continue

            xmin = int(point[0] - 32)
            xmax = int(point[0] + 32)
            ymin = int(point[1] - 32)
            ymax = int(point[1] + 32)
            x[j, :, :, :] = img[ymin:ymax, xmin:xmax, :]
            y[j, :] = des[j]
            j += 1
        x /= 127.5
        x -= 1
        y /= 255.
        return x, y

    def __len__(self):
        return len(self.listfile)
