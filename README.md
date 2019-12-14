# Final Term Project
## Bag of Words
   1. 이미지를 sift하여 feature를 뽑는다  
   2. kmeans-clustering을 통해서 center를 찾는다. 이를 codeword라고 한다  
   3. codeword들로 구성된 codebook을 만든다  
   4. 이미지 하나당 codeword로 이루어진 히스토그램을 그린다  
   5. svm분류기로 학습을 한다  

## Spatial Pyramid Matching(SPM)
  spm은 이미지를 level별로 분할하여 각 분할 영역마다 히스토그램을 구하여 비교하는 방법이다  
   level-1인 경우에는 원본 사진 1장 그대로, level-2인 경우에는 1장을 4등분, level-3인 경우에는 16등분 하는 셈이다  
   이 때, 레벨이 올라갈 수록 즉, 분할을 많이 할 수록 높은 가중치를 준다.   
   level-1에서는 히스토그램이 한개 나올 것이다 h-1  
   level-2에서는 히스토그램이 4개 존재 h2-1,h2-2,h2-3,h2-4  
   level-3에서는 히스토그램이 16개 존재 h3-1,h3-2,h3-3,...,h3-16  
   이미지를 각각 학습시키고 나면 위와 같은 갯수의 히스토그램이 만들어진다  
   level-1: H-1  
   level-2: H2-1,H2-2,H2-3,H2-4  
   level-3: H3-1,H3-2,H3-3,...,H3-16  
   그러고 나면 h-1과 H-1을 비교하고, h2-1과 H2-1을 비교하는 방식으로 유사도를 측정하여   
   이 유사도를 종합하여 predict하는 방법이다  
 
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
