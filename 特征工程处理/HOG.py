from defines import *

def get_HOG_feature():
    def get_hog_feature(img):
        img = segment_plant(img)
        img = cv2.resize(img,(64,64))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feature=hog.compute(img, winStride=(32,32), padding=(0,0)).flatten()
        return feature

    # 将数据dump保存  下次不需要重新读取
    if not os.path.exists('train_data.pkl'):
        train_data,test_data,image_type=load_data()
        pickle.dump(train_data,open("train_data.pkl",'wb'))
        pickle.dump(test_data,open("test_data.pkl",'wb'))
        pickle.dump(image_type,open("image_type.pkl",'wb'))
    else:
        train_data=pickle.load(open("train_data.pkl",'rb'))
        test_data=pickle.load(open("test_data.pkl",'rb'))
        image_type=pickle.load(open("image_type.pkl",'rb'))


    #定义对象hog，同时输入定义的参数，剩下的默认即可
    winSize = (64,64)
    blockSize = (32,32)
    blockStride = (16,16)
    cellSize = (8,8)
    nbins = 9
    hog = cv2.HOGDescriptor( winSize,blockSize,blockStride,cellSize, nbins )

    train_feature=[]
    for class_label in train_data.keys():
        for image in train_data[class_label]:
            feature = get_hog_feature(image).tolist()
            train_feature.append(feature)
    print("**************************")
    print(len(train_feature))

    # PCA降维
    pca = PCA(n_components=16)  # 自动选择特征个数  'mle'
    print("**************************")
    pca.fit(train_feature)
    print("**************************")
    train_feature = pca.transform(train_feature)

    test_feature=[]
    for image in test_data:
            feature = get_hog_feature(image).tolist()
            test_feature.append(feature)
    print(len(test_feature))
    test_feature = np.array(test_feature)
    test_feature = pca.transform(test_feature)

    return train_feature,test_feature
