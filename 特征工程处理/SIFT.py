from defines import *
from color import get_color_feature


def build_vocabulary(image_paths, k, length):
    bow_kmeans_trainer = cv2.BOWKMeansTrainer(k)
    sift = cv2.SIFT_create()
    for i in range(length):
        for j in [0, 248, 610, 878, 1449, 1652, 2083, 2696, 2902, 3388, 3608, 4069]:
            img = cv2.imread(image_paths[j + i])
            gray = segment_plant(img)
            kp = sift.detect(gray, None)
            if len(kp) != 0:
                bow_kmeans_trainer.add(sift.compute(gray, sift.detect(gray, None))[1])
    vocab = bow_kmeans_trainer.cluster()
    return vocab


def get_image_paths(data_path, categories):
    train_image_paths = []
    test_image_paths = []

    train_labels = []
    test_labels = []
    print("获得训练集数据路径")
    for category in categories:
        image_paths = glob(os.path.join(data_path, 'train', category, '*.png'))
        path = os.path.join(data_path, 'train', category)
        files = os.listdir(path)

        for i in range(len(files)):
            train_image_paths.append(image_paths[i])
            train_labels.append(category)
        print(path, '读取完成！', '->', len(files))

    print("获得测试集数据路径")
    image_paths = glob(os.path.join(data_path, 'test', '*.png'))
    path = os.path.join(data_path, 'test')
    files = os.listdir(path)
    for i in range(len(files)):
        test_image_paths.append(image_paths[i])
    print(path, '读取完成！', '->', len(files))
    return train_image_paths, test_image_paths, train_labels


def get_train_feat(image_paths, vocab, k):
    flann_params = dict(algorithm=1, tree=5)
    flann = cv2.FlannBasedMatcher(flann_params, {})
    sift = cv2.SIFT_create()
    bow_img_descriptor_extractor = cv2.BOWImgDescriptorExtractor(sift, flann)
    bow_img_descriptor_extractor.setVocabulary(vocab)
    train = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        gray = segment_plant(img)
        one = bow_img_descriptor_extractor.compute(gray, sift.detect(gray, None))
        if one is None:
            one = np.array([[0 for i in range(k)]])

        one = one.tolist()[0]
        two = get_color_feature(gray)
        train.append(one + two)

    return train