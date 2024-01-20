from defines import *


# 图像掩膜操作
# 去除其他环境影响，只保留叶子
def create_mask_for_plant(image):
    # bgr转化为hsv
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #绿色hsv大致区间
    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])
    # lower_hsv = np.array([25, 40, 40])
    # upper_hsv = np.array([80, 255, 255])

    # 二值化
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 形态学开操作 先腐蚀后膨胀
    # 同样是降噪 平滑 预处理 对于此题形态学比高斯模糊效果更加好
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def segment_plant(image):
    mask = create_mask_for_plant(image)
    # 利用掩膜进行图像混合
    # 求交集 、 掩膜提取图像
    output = cv2.bitwise_and(image, image, mask=mask)
    return output
