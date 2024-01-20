from defines import *

def get_color_feature(img):
    r, g, b = cv2.split(img)
    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)
    r_std = np.std(r)
    g_std = np.std(g)
    b_std = np.std(b)
    r_offset = (np.mean(np.abs((r - r_mean) ** 3))) ** (1. / 3)
    g_offset = (np.mean(np.abs((g - g_mean) ** 3))) ** (1. / 3)
    b_offset = (np.mean(np.abs((b - b_mean) ** 3))) ** (1. / 3)

    # # img= cv2.medianBlur(img, 3)  # 中值滤波
    # one = np.cumsum(cv2.calcHist([img], [1], None, [256], [0, 255],accumulate=True))
    # # one = np.std(one).tolist()
    # one = (one/(img.shape[0]*img.shape[1])).tolist()
    one = np.log1p([r_mean, g_mean, b_mean, r_std, g_std, b_std, r_offset, g_offset, b_offset]).tolist()
    return one