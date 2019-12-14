# Final Term Project
## Bag of Words
1. 이미지를 sift하여 feature를 뽑는다
2. kmeans-clustering을 통해서 center를 찾는다. 이를 codeword라고 한다
3. codeword들로 구성된 codebook을 만든다
4. 이미지 하나당 codeword로 이루어진 히스토그램을 그린다
5. svm분류기로 학습을 한다

### sift (Scale Invariant Fature Transform) 
> 이미지의 크기가 달라지더라도 이미지의 특징적은 부분을 검출하는 기법

    sift = cv2.xfeatures2d.SIFT_create()
    #sift의 keypoint, descriptor을 계산하는 함수를 제공
    kp = [cv2.KeyPoint(x, y, Step_size) for y in range(0, img_gray.shape[0], Step_size) for x in range(0, img_gray.shape[1], Step_size)]
    #keypoint추출
    keypoint, des = sift.compute(img_gray, kp)
    #keypoints에서 descriptor를 계산한 후 keypoint와 descriptor를 리턴
        
### codebook
>  codebooksize는 400이 가장 성능이 좋았지만 노트북이 감당을 하지 못한다

    ! yes | pip3 install kmc2
    import kmc2
    from sklearn.cluster import MiniBatchKMeans
    import numpy as np
    codebooksize=200
    seeding = kmc2.kmc2(np.array(train_des).reshape(-1,128), codebooksize) 
    kmeans = MiniBatchKMeans(codebooksize, init=seeding).fit(np.array(train_des).reshape(-1,128))
    codebook = kmeans.cluster_centers_

### encoder
> spm이 아니라 bovw를 사용할 경우 encoding해서 svm에 넣어줘야한다

    def input_vector_encoder(feature, codebook):
    code, _ = vq.vq(feature, codebook)
    word_hist, bin_edges = np.histogram(code, bins=range(codebook.shape[0] + 1), normed=True)
    return word_hist
    #feature에는 sift한 이미지들을 하나씩 넣어준다
    
    
### svm

    C_range = 10.0 ** np.arange(-3, 3)
    gamma_range = 10.0 ** np.arange(-3, 3)
    param_grid = dict(gamma=gamma_range.tolist(), C=C_range.tolist())
    clf = GridSearchCV(svm.SVC(), param_grid, cv=5, n_jobs=-2)
    clf.fit(d_train, train_labels)
    y_fit=clf.predict(d_test)
