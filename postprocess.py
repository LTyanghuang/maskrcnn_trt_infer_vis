# python3
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

classes_labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus",
            "train", "truck", "boat", "traffic light", "fire hydrant",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock",
            "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

def vis_image(images, boxes, scores, labels, threshold, input_shape):
    src_img = cv2.imread(images)
    h, w, c = src_img.shape
   
    img_cv2 = cv2.resize(src_img, (input_shape[2], input_shape[3]))
    scores_list = np.squeeze(scores, axis=0)
    number_boxes_list = []
    for box_score in scores_list.tolist():
        if box_score != 0. and box_score > threshold:
            number_boxes_list.append(box_score)
    number_boxes = len(number_boxes_list)
    # print("number_boxes = ", number_boxes)

    boxes= np.squeeze(boxes, axis=0)
    labels= np.squeeze(labels, axis=0)
    
    boxes_list = boxes.tolist()[0 : number_boxes]
    labels_list = labels.tolist()[0 : number_boxes]

    print("boxes_list = ", boxes_list)
    # print("scores_list = ", scores_list)
    # print("labels_list = ", labels_list)

    for i in range(number_boxes):
        left = int(boxes_list[i][0])
        top = int(boxes_list[i][1])
        right = int(boxes_list[i][2])
        bottom = int(boxes_list[i][3])
        scores = scores_list[i]
        label = classes_labels[int(labels_list[i])]
        ptLeftTop = (left, top)
        ptRightBottom = (right, bottom)
        point_color = (0, 255, 0) # BGR
        thickness = 1 
        lineType = 4
        cv2.rectangle(img_cv2, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
        cv2.putText(
                img_cv2,
                label+' : '+str(scores),
                (left, top),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        print("label : {}, score : {}".format(label, scores))
    src_image_size = cv2.resize(img_cv2, (w, h))
    cv2.imwrite("res_od.jpg", src_image_size)

def plt_bboxes(img_path, classes, scores, bboxes, threshold, figsize=(10,10), linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    img = mpimg.imread(img_path)
    bboxes= np.squeeze(bboxes, axis=0)
    classes= np.squeeze(classes, axis=0)
    scores = np.squeeze(scores, axis=0)
    # 筛选出大于score threshold的box
    number_boxes_list = []
    for box_score in scores.tolist():
        if box_score != 0. and box_score > threshold:
            number_boxes_list.append(box_score)
    number_boxes = len(number_boxes_list)
    bboxes = bboxes[0 : number_boxes]
    classes = classes[0 : number_boxes]

    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            xmin = int(bboxes[i, 0]*width/384)
            ymin = int(bboxes[i, 1]*height/384)
            xmax = int(bboxes[i, 2]*width/384)
            ymax = int(bboxes[i, 3]*height/384)
            
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            class_name = str(cls_id)
            class_name = classes_labels[int(cls_id)]
            print("class_name = ", class_name)
            
            plt.gca().text(xmin, ymin - 2,
                           '{:s} | {:.3f}'.format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')
    plt.imshow(img)
    plt.savefig("res_od.jpg")


def py_nms(dets, thresh):
    """Pure Python NMS baseline."""
    #x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    #每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #order是按照score降序排序的
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        #计算当前概率最大矩形框与其他矩形框的相交框的坐标，会用到numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        #计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        #找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        #将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return keep
def get_boxes_and_scores(boxes, scores):
    # print("boxes = ", boxes)
    # print("scores = ", scores)
    scores = np.squeeze(scores, axis=0)
    scores_list = scores.tolist()
    scores_ok_list = []
    for i in scores_list:
        tmp_list = []
        tmp_list.append(i)
        scores_ok_list.append(tmp_list)
    # print("B_ok_list = ", scores_ok_list) 
    scores_ok_numpy = np.array(scores_ok_list)
    # print("B_ok_numpy = ", scores_ok_numpy)
    C = np.concatenate([boxes, scores_ok_numpy], axis=1)
    # print("C = ", C)
    return C
# test
if __name__ == "__main__":
    dets = np.array([[30, 20, 230, 200, 1], 
                     [50, 50, 260, 220, 0.9],
                     [210, 30, 420, 5, 0.8],
                     [430, 280, 460, 360, 0.7]])


    A = np.array([[298.69214,   184.99751,   361.89355,   203.16533],
                  [  0. ,     7.3610764 , 37.162582,  138.2139]])
    B = np.array([[0.90803236, 0.81604093]])
    
    print("A = ", A)
    print("B = ", B)
    B = np.squeeze(B, axis=0)
    B_list = B.tolist()
    B_ok_list = []
    for i in B_list:
        tmp_list = []
        tmp_list.append(i)
        B_ok_list.append(tmp_list)
    print("B_ok_list = ", B_ok_list) 
    B_ok_numpy = np.array(B_ok_list)
    print("B_ok_numpy = ", B_ok_numpy)
    C = np.concatenate([A, B_ok_numpy], axis=1)
    print("C = ", C)
    print('-'*50)
    print("dets = ", dets)
    thresh = 0.35
    keep_dets = py_nms(dets, thresh)
    print(keep_dets)
    print(dets[keep_dets])

