from defines import *
def get_LBP_feature():
    def get_lbp_feature(img):
        # 提取出绿色部分
        img = segment_plant(img)
        img = cv2.resize(img,(128,128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # LBP特征提取
        radius = 1  # LBP算法中范围半径的取值
        n_points = 8 * radius  # 领域像素点数

        img = ft.local_binary_pattern(gray, n_points, radius)
        max_bins = int(img.max() + 1)
        img_flat = img.reshape(img.shape[0], -1)
        img,_ = np.histogram(img_flat, density=True, bins=max_bins, range=(0, max_bins))
        return img.flatten()

    # 将数据dump保存  下次不需要重新读取
    if not os.path.exists('train_data.pkl'):
        train_data, test_data, image_type = load_data()
        pickle.dump(train_data, open("train_data.pkl", 'wb'))
        pickle.dump(test_data, open("test_data.pkl", 'wb'))
        pickle.dump(image_type, open("image_type.pkl", 'wb'))
    else:
        train_data = pickle.load(open("train_data.pkl", 'rb'))
        test_data = pickle.load(open("test_data.pkl", 'rb'))
        image_type = pickle.load(open("image_type.pkl", 'rb'))


    train_feature = []
    for class_label in train_data.keys():
        for image in train_data[class_label]:
            feature = get_lbp_feature(image).tolist()
            train_feature.append(feature)
    print("**************************")
    print(len(train_feature))

    # PCA降维
    pca = PCA(n_components=8)  # 自动选择特征个数  'mle'
    print("**************************")
    pca.fit(train_feature)
    print("**************************")
    train_feature = pca.transform(train_feature)

    test_feature = []
    for image in test_data:
        feature = get_lbp_feature(image).tolist()
        test_feature.append(feature)
    print(len(test_feature))
    test_feature = np.array(test_feature)

    test_feature = pca.transform(test_feature)

    return train_feature, test_feature