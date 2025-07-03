import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2


def detect_face_from_image(img_path):
    # 1、创建人脸检测器
    # 需要先下载预训练模型
    base_options = python.BaseOptions(model_asset_path='/nvfile-heatstorage/zangxh/intern/wangyx/evaluation/blaze_face_short_range.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    # 2、加载输入图片
    image = mp.Image.create_from_file(img_path)


    # 3、使用下载好的模型进行人脸检测
    detection_result = detector.detect(image)
    return detection_result

    # # 4、 可视化
    # image_copy = np.copy(image.numpy_view())
    # annotated_image = visualize(image_copy, detection_result)
    # rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    # # 在使用OpenCV的cv2.imshow函数显示图像时，它会默认将传入的图像数据解释为BGR格式
    # # 如果你传入的是RGB格式的图像数据，OpenCV会在显示时进行颜色通道的调整，使图像以BGR格式进行显示。
    # cv2.imshow('face detection', rgb_annotated_image)

    # # 输入esc结束捕获
    # if cv2.waitKey(0) == 27:
    #     cv2.destroyAllWindows()


if __name__ == '__main__':
    prompt_list = open("/nvfile-heatstorage/zangxh/intern/wangyx/prompt_test.txt", "r").readlines()
    prompts = [prompt.strip() for prompt in prompt_list]
   
    scores = []
    for i, prompt in enumerate(prompts): 
            for img_path in os.listdir("/nvfile-heatstorage/zangxh/intern/wangyx/train_repo/1000_test/sdxl/gen/control_new/"+str(i)):
                    detection_result = detect_face_from_image(img_path="/nvfile-heatstorage/zangxh/intern/wangyx/train_repo/1000_test/sdxl/gen/control_new/"+str(i)+"/"+img_path)
                    if detection_result is not None and len(detection_result.detections)>0:
                            for detection in detection_result.detections:
                                for info in detection.categories:
                                    score = info.score 
                                    scores.append(score)
    print(len(scores))
    print(np.mean(scores))
                     
