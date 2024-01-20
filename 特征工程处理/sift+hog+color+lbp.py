from defines import *
from SIFT import *
from HOG import *
from LBP import *
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
if __name__ == '__main__':
    data_path = "data"
    categories = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen',
                  'Loose Silky-bent',
                  'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
    CATE2ID = {v: k for k, v in enumerate(categories)}
    train_image_paths, test_image_paths, train_labels = get_image_paths(data_path, categories)
    trian_image_labels_id = [CATE2ID[x] for x in train_labels]
    print("开始建立词汇表===========================>")
    if not os.path.exists('vocab.pkl'):
        vacob = build_vocabulary(train_image_paths, 100, 200)
        pickle.dump(vacob, open('vocab.pkl', 'wb'))
    else:
        vacob = pickle.load(open('vocab.pkl', 'rb'))
    print(vacob.shape)
    print("词汇表建立完成===========================>")
    #HOG特征
    HOG_train, HOG_test = get_HOG_feature()
    #LBP特征
    LBP_train,LBP_test=get_LBP_feature()

    print("开始提取训练集特征===========================>")
    if not os.path.exists('train_sift+color.pkl'):
        train_X = get_train_feat(train_image_paths, vacob, 100)
        pickle.dump(train_X, open('train_sift+color.pkl', 'wb'))
    else:
        train_X = pickle.load(open('train_sift+color.pkl', 'rb'))

    train = []
    for i in range(len(train_X)):
        train.append(train_X[i] + HOG_train[i].tolist()+LBP_train[i].tolist())
        #train.append(train_X[i] + HOG_train[i].tolist())
    print("提取完成===========================>")
    train_y = trian_image_labels_id
    print("开始提取测试集特征===========================>")
    if not os.path.exists('test_sift+color.pkl'):
        test_X = get_train_feat(test_image_paths, vacob, 100)
        pickle.dump(test_X, open('test_sift+color.pkl', 'wb'))
    else:
        test_X = pickle.load(open('test_sift+color.pkl', 'rb'))
    test = []
    for i in range(len(test_X)):
        test.append(test_X[i] + HOG_test[i].tolist()+LBP_test[i].tolist())
        #test.append(test_X[i] + HOG_test[i].tolist())
    print("提取完成===========================>")
    print("开始训练===========================>")

    # 打乱顺序
    train, train_y = shuffle(train, train_y)
    # 划分
    #train, x_test, train_y, y_test = train_test_split(train, train_y, test_size=0.2)

    rf_model = RandomForestClassifier(n_estimators=400, max_features='sqrt', max_depth=40,
                                 bootstrap=False, oob_score=False, random_state=10)
    rbf_model = svm.SVC(kernel='rbf', probability=True, gamma='auto', C=700).fit(train, train_y)
    poly_model = svm.SVC(kernel='poly', probability=True, gamma='auto', C=800).fit(train, train_y)
    xg_model= XGBClassifier(learning_rate=0.13, max_depth=5, n_estimators=300, nthread=10,
                        use_label_encoder=False, eval_metric='mlogloss')
    voting_clf = VotingClassifier(estimators=[('rf', rf_model), ('svm_rbf', rbf_model),('xg',xg_model),('poly',poly_model)], voting='soft')
    voting_clf.fit(train, train_y)


    print("训练结束===========================>")
    #score1=model.score(train, train_y)
    score1=voting_clf.score(train,train_y)
    print("训练集准确率=====>>>>>>", score1)
    #score2= model.score(x_test, y_test)
    #score2= voting_clf.score(x_test, y_test)
    #print("验证集准确率=====>>>>>>",score2)
    print("开始预测===========================>")
    #preds = model.predict(test)
    preds = voting_clf.predict(test)
    print("预测结束===========================>")
    test = []
    for i in range(preds.shape[0]):
        test.append(categories[preds[i]])
    print(len(test))
    sample = pd.read_csv("submission_features.csv")
    submission = pd.DataFrame({'ID': sample['ID'], 'Category': test})
    #submission.to_csv(f'newsift+color+HOG+LBP+SVM{score1+score2}.csv', index=False)
    submission.to_csv(f'sift+color+HOG+LBP+SVM{score1}.csv', index=False)