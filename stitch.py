import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import configparser
import os


# ================================================================== #
#                     选择特征提取器函数
# ================================================================== #
def detectAndDescribe(image, method=None):
    assert (
        method is not None
    ), "You need to define a feature detection method. Values are: 'sift', 'surf'"
    if method == "sift":
        descriptor = cv2.SIFT_create()
    elif method == "surf":
        descriptor = cv2.SURF_create()
    elif method == "brisk":
        descriptor = cv2.BRISK_create()
    elif method == "orb":
        descriptor = cv2.ORB_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    return (kps, features)


# ================================================================== #
#                     暴力检测函数
# ================================================================== #
def matchKeyPointsBF(featuresA, featuresB, method):
    bf = createMatcher(method, crossCheck=True)
    best_matches = bf.match(featuresA, featuresB)
    rawMatches = sorted(best_matches, key=lambda x: x.distance)
    print("Raw matches (Brute force):", len(rawMatches))
    return rawMatches


# ================================================================== #
#                     使用knn检测函数
# ================================================================== #
def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    matches = []
    for m, n in rawMatches:
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches


# ================================================================== #
#                     创建匹配器
# ================================================================== #
def createMatcher(method, crossCheck):
    if method == "sift" or method == "surf":
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == "orb" or method == "brisk":
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf


# ================================================================== #
#                     计算关键点的透视关系
# ================================================================== #
def getHomography(kpsA, kpsB, matches, reprojThresh):
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    if len(matches) > 4:
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        return (matches, H, status)
    else:
        return None


# ================================================================== #
#                     去除图像黑边
# ================================================================== #
def cutBlack(pic):
    rows, cols = np.where(pic[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    return pic[min_row:max_row, min_col:max_col, :]


# ================================================================== #
#                     调换
# ================================================================== #
def swap(a, b):
    return b, a


# ================================================================== #
#                     主要的函数
#                合并两张图（合并多张图基于此函数）
# ================================================================== #
def handle(path1, path2, isShow=False):
    feature_extractor = "sift"
    feature_matching = "knn"

    imageA = cv2.imread(path2)
    imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)
    imageA_gray = cv2.cvtColor(imageA, cv2.COLOR_RGB2GRAY)

    imageB = cv2.imread(path1)
    imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB)
    imageB_gray = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)

    if isShow:
        f = plt.figure(figsize=(10, 4))
        f.add_subplot(1, 2, 1)
        plt.title("imageB")
        plt.imshow(imageB)
        plt.xticks([]), plt.yticks([])
        f.add_subplot(1, 2, 2)
        plt.title("imageA")
        plt.imshow(imageA)
        plt.xticks([]), plt.yticks([])

    kpsA, featuresA = detectAndDescribe(imageA_gray, method=feature_extractor)
    kpsB, featuresB = detectAndDescribe(imageB_gray, method=feature_extractor)

    if isShow:
        fig, (ax1, ax2) = plt.subplots(
            nrows=1, ncols=2, figsize=(10, 4), constrained_layout=False
        )
        ax1.imshow(cv2.drawKeypoints(imageA_gray, kpsA, None, color=(0, 255, 0)))
        ax1.set_xlabel("(a)key point", fontsize=14)
        ax2.imshow(cv2.drawKeypoints(imageB_gray, kpsB, None, color=(0, 255, 0)))
        ax2.set_xlabel("(b)key point", fontsize=14)

    if feature_matching == "bf":
        matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
        img3 = cv2.drawMatches(
            imageA,
            kpsA,
            imageB,
            kpsB,
            matches[:100],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
    elif feature_matching == "knn":
        matches = matchKeyPointsKNN(
            featuresA, featuresB, ratio=0.75, method=feature_extractor
        )
        if len(matches) < 10:
            return None
        img3 = cv2.drawMatches(
            imageA,
            kpsA,
            imageB,
            kpsB,
            np.random.choice(matches, 100),
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

    if isShow:
        fig = plt.figure(figsize=(10, 4))
        plt.imshow(img3)
        plt.title("feature match")
        plt.axis("off")

    matchCount = len(matches)
    M = getHomography(kpsA, kpsB, matches, reprojThresh=4)
    if M is None:
        print("Error!")
    (matches, H, status) = M

    result = cv2.warpPerspective(
        imageA,
        H,
        (
            (imageA.shape[1] + imageB.shape[1]) * 2,
            (imageA.shape[0] + imageB.shape[0]) * 2,
        ),
    )
    resultAfterCut = cutBlack(result)

    if np.size(resultAfterCut) < np.size(imageA) * 0.95:
        print("图片位置不对,将自动调换")
        kpsA, kpsB = swap(kpsA, kpsB)
        imageA, imageB = swap(imageA, imageB)
        if feature_matching == "bf":
            matches = matchKeyPointsBF(featuresB, featuresA, method=feature_extractor)
        elif feature_matching == "knn":
            matches = matchKeyPointsKNN(
                featuresB, featuresA, ratio=0.75, method=feature_extractor
            )
            if len(matches) < 10:
                return None
        matchCount = len(matches)
        M = getHomography(kpsA, kpsB, matches, reprojThresh=4)
        if M is None:
            print("Error!")
        (matches, H, status) = M
        result = cv2.warpPerspective(
            imageA,
            H,
            (
                (imageA.shape[1] + imageB.shape[1]) * 2,
                (imageA.shape[0] + imageB.shape[0]) * 2,
            ),
        )

    result[0 : imageB.shape[0], 0 : imageB.shape[1]] = np.maximum(
        imageB, result[0 : imageB.shape[0], 0 : imageB.shape[1]]
    )
    result = cutBlack(result)
    return result, matchCount


# ================================================================== #
#                     合并多张图
# ================================================================== #
def handleMulti(*args, isShow=False):
    l = len(args)
    if isShow:
        row = math.ceil(l / 3)
        f = plt.figure(figsize=(10, 4))
        for i in range(l):
            f.add_subplot(row, 3, i + 1)
            plt.title(f"image({i+1})")
            plt.axis("off")
            plt.imshow(cv2.cvtColor(cv2.imread(args[i]), cv2.COLOR_BGR2RGB))
    assert l > 1
    isHandle = [0 for i in range(l - 1)]
    nowPic = args[0]
    args = args[1:]
    for j in range(l - 1):
        isHas = False
        matchCountList = []
        resultList = []
        indexList = []
        for i in range(l - 1):
            if isHandle[i] == 1:
                continue
            result, matchCount = handle(nowPic, args[i])
            if not result is None:
                matchCountList.append(matchCount)
                resultList.append(result)
                indexList.append(i)
                isHas = True
        if not isHas:
            return None
        else:
            index = matchCountList.index(max(matchCountList))
            nowPic = resultList[index]
            isHandle[indexList[index]] = 1
            print(f"合并第{indexList[index] + 2}个")
    return nowPic


# ================================================================== #
#                     主函数
# ================================================================== #
def stitch():
    config_file = "mixConfig.ini"
    config = configparser.ConfigParser()
    config.read(config_file)

    img1_path = config["stitch"]["image1"]
    img2_path = config["stitch"]["image2"]
    output_folder = config["stitch"]["output"]

    result, _ = handle(img1_path, img2_path, isShow=True)
    if not result is None:
        cv2.imshow("result", result[:, :, [2, 1, 0]])
        plt.show()
        cv2.waitKey(0)

        output_path = os.path.join(output_folder, "stitched_result.png")
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"结果图片已保存到 {output_path}")
    else:
        print("没有找到对应特征点,无法合并")
