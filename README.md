# Final Term Project

### sift (Scale Invariant Fature Transform) 
> 이미지의 크기가 달라지더라도 이미지의 특징적은 부분을 검출하는 기법

    sift = cv2.xfeatures2d.SIFT_create()
    #sift의 keypoint, descriptor을 계산하는 함수를 제공
    kp = [cv2.KeyPoint(x, y, Step_size) for y in range(0, img_gray.shape[0], Step_size) for x in range(0, img_gray.shape[1], Step_size)]
    #keypoint추출
    keypoint, des = sift.compute(img_gray, kp)
    #keypoints에서 descriptor를 계산한 후 keypoint와 descriptor를 리턴
        
### codebook
    ! yes | pip3 install kmc2
    import kmc2
    from sklearn.cluster import MiniBatchKMeans
    import numpy as np
    codebooksize=200
    seeding = kmc2.kmc2(np.array(train_des).reshape(-1,128), codebooksize) 
    kmeans = MiniBatchKMeans(codebooksize, init=seeding).fit(np.array(train_des).reshape(-1,128))
    codebook = kmeans.cluster_centers_
