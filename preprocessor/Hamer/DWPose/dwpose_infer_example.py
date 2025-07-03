import os

from annotator.dwpose import DWposeDetector

if __name__ == "__main__":
    pose = DWposeDetector()
    import cv2

    for filename in os.listdir('./sdxl_test_mesh'):
        oriImg = cv2.imread('./sdxl_test_mesh/' + filename)  # B,G,R order
        import matplotlib.pyplot as plt

        out = pose(oriImg)
        plt.imsave('./pose/'+filename, out)
