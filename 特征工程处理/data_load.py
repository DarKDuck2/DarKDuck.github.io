from defines import *

def load_data():
    '''
    读取训练集和测试集
    {0: 'Black-grass', 1: 'Charlock', 2: 'Cleavers', 3: 'Common Chickweed', 4: 'Common wheat', 5: 'Fat Hen', 6: 'Loose Silky-bent',
    7: 'Maize', 8: 'Scentless Mayweed', 9: 'Shepherds Purse', 10: 'Small-flowered Cranesbill', 11: 'Sugar beet'}
    '''
    #整个训练集plant-seedlings-classification文件夹的路径
    DATA_FOLDER = "data"
    # train_folder保存为train文件夹的路径
    TRAIN_FOLDER = os.path.join(DATA_FOLDER,'train')
    # test_folder保存为test文件夹的路径
    TEST_FOLDER = os.path.join(DATA_FOLDER,'test')
    print(os.listdir(TRAIN_FOLDER))
    print(os.listdir(TEST_FOLDER)[:10])

    # 读取训练集
    train = {}
    image_type={}
    i=0
    #train{}为一个字典  train.key()为plant的标签 对应的train[label]为所有的训练的图片的numpy矩阵
    for plant_name in os.listdir(TRAIN_FOLDER):
        plant_path = os.path.join(TRAIN_FOLDER, plant_name)
        label = plant_name
        train[i] = []
        for image_path in glob(os.path.join(plant_path,'*png')):
            image = cv2.imread(image_path)
            train[i].append(image)
        print(plant_path,'读取完成！',label,'->',len(train[i]))
        image_type[i]=label
        i+=1
    print(image_type)

    # 读取测试集
    test_data=[]
    for image_path in glob(os.path.join(TEST_FOLDER,'*png')):
        image = cv2.imread(image_path)
        test_data.append(image)
    print('测试集长度：',len(test_data))
    print("读取完成！")
    return train,test_data,image_type