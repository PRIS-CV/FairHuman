import os

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

prompt_list = open("/nvfile-heatstorage/zangxh/intern/wangyx/prompt_test.txt", "r").readlines()
prompts = [prompt.strip() for prompt in prompt_list]

score = []

for i, prompt in enumerate(prompts): 
    for img_path in os.listdir("/nvfile-heatstorage/zangxh/intern/wangyx/train_repo/1000_test/sdxl/gen/control_new/"+str(i)):
                image = cv2.imread("/nvfile-heatstorage/zangxh/intern/wangyx/train_repo/1000_test/sdxl/gen/control_new/"+str(i)+"/"+img_path)
                image = image.copy()
                H, W, C = image.shape
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                if results.multi_handedness:
                    for handLms in results.multi_handedness:
                        for info in handLms.classification:
                            score.append(info.score)
print(len(score))
print(np.mean(score))