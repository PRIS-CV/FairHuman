import os
from yolo import YOLODetecotor
from PIL import Image, ImageDraw, ImageFilter
from tools import bbox_padding_by_ratio, bbox_padding, mask_dilate
image_dir="/nvfile-heatstorage/zangxh/intern/wangyx/dataset/BodyHands_new/image"
mask_hand_dir="/nvfile-heatstorage/zangxh/intern/wangyx/dataset/BodyHands_new/mask_hand"
mask_face_dir="/nvfile-heatstorage/zangxh/intern/wangyx/dataset/BodyHands_new/mask_face"
# Bulid Detector
model_dir = "./models"
device = "cuda:0"
detector_hand = YOLODetecotor(os.path.join(model_dir, "yolo/hand_yolov8n.pt"), 0.5, device)
detector_face = YOLODetecotor(os.path.join(model_dir, "yolo/face_yolov8m.pt"), 0.5, device)

def create_mask_from_bboxes(bboxes, shape):
    mask = Image.new("L", shape, "black")
    mask_draw = ImageDraw.Draw(mask)
    if len(bboxes)>0:
            for bbox in bboxes:
                mask_draw.rectangle(bbox, fill="white")
    else:
        mask = mask
    return mask

for img_path in os.listdir(image_dir):
    image=Image.open(image_dir+"/"+img_path)
    basename=img_path.split(".")[0]
    # if os.path.exists(mask_hand_dir+"/"+basename+".png"):
    #     continue 
    width, height = image.size
    yolo_detections_hand, _, _ = detector_hand(image)
    yolo_detections_face,_,_ = detector_face(image)
    mask_hand_dilate=Image.new("RGB", image.size, "black")
    mask_face_dilate=Image.new("RGB", image.size, "black")
    if yolo_detections_hand is not None and len(yolo_detections_hand)>0:
        ## bbox for hands
        # hand_bboxes=[]
        # for bbox in yolo_detections_hand:
        #     hand_bboxes.append(bbox_padding(bbox=bbox,value=16))
        hand_bboxes=yolo_detections_hand
        mask_hand=create_mask_from_bboxes(hand_bboxes,image.size)
        mask_hand_dilate=mask_dilate(mask_hand,value=16)
    if yolo_detections_face is not None and len(yolo_detections_face)>0:
        ## bbox for face
        # face_bboxes=[]
        # for bbox in yolo_detections_face:
        #     face_bboxes.append(bbox_padding(bbox=bbox,value=16))
        face_bboxes=yolo_detections_face
        mask_face=create_mask_from_bboxes(face_bboxes,image.size)
        mask_face_dilate=mask_dilate(mask_face,value=16)
    mask_hand_dilate.save(mask_hand_dir+"/"+basename+".png")
    mask_face_dilate.save(mask_face_dir+"/"+basename+".png")

