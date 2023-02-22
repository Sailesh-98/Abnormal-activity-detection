import os
import traceback

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
from sklearn.metrics import precision_recall_fscore_support as score, accuracy_score

from app.abnormal_activity import create_mega_blocks as cmb
from app.abnormal_activity import motion_influence_generator as mig
from app.settings import MEDIA_ROOT


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


test1OriginalFrames = ['Normal', 'Normal', 'Abnormal', 'Normal', 'Abnormal', 'Normal', 'Normal', 'Abnormal', 'Normal', 'Normal', 'Normal', 'Normal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Normal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Normal', 'Normal', 'Normal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Abnormal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Abnormal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal']


def train_from_video(vid, video_name):
    '''
        calls all methods to train from the given video
        May return codewords or store them.
    '''
    print("Training From ", vid)
    try:
        MotionInfOfFrames, rows, cols = mig.getMotionInfuenceMap(vid)
        print("Motion Inf Map", len(MotionInfOfFrames))

        megaBlockMotInfVal = cmb.createMegaBlocks(MotionInfOfFrames, rows, cols)

        filename = MEDIA_ROOT + "output\\mega_block_motion_val_train_k5.npy".format(video_name)
        f_handle = file(filename, 'a')
        np.save(f_handle, megaBlockMotInfVal)
        f_handle.close()

        print('megaBlockMotInfVal ', np.amax(megaBlockMotInfVal))
        print('reject_outliers megaBlockMotInfVal ', np.amax(reject_outliers(megaBlockMotInfVal)))

        filename = MEDIA_ROOT + "output\\code_words_train_k5.npy".format(video_name)
        codewords = cmb.kmeans(megaBlockMotInfVal)
        f_handle = file(filename, 'a')
        np.save(f_handle, codewords)
        f_handle.close()

    except Exception as e:
        print(e)
        traceback.print_exc()
        return False
    return True


def square(a):
    return (a ** 2)


def diff(l):
    return (l[0] - l[1])


def showUnusualActivities(unusual, vid, noOfRows, noOfCols, n):
    global test1OriginalFrames
    unusualFrames = unusual.keys()
    unusualFrames.sort()

    cap = cv2.VideoCapture(vid)
    ret, frame = cap.read()
    rows, cols = frame.shape[0], frame.shape[1]
    rowLength = rows / (noOfRows / n)
    colLength = cols / (noOfCols / n)
    print("Block Size ", (rowLength, colLength))
    count = 0
    screen_res = 980, 520
    scale_width = screen_res[0] / 320
    scale_height = screen_res[1] / 240
    scale = min(scale_width, scale_height)
    window_width = int(320 * scale)
    window_height = int(240 * scale)

    unusualFramesArr = []
    unusualFramesNoDetected = ['Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal']
    while 1:
        print('count', count)
        ret, uFrame = cap.read()

        if ret is None or ret == False:
            break

        if count in unusualFrames:
            filename = 'unusual_frame_{}.jpg'.format(count)
            file_path = os.path.join(MEDIA_ROOT + 'output', filename)
            unusualFramesArr.append(filename)
            unusualFramesNoDetected.append("Abnormal")
            for blockNum in unusual[count]:
                print(blockNum)
                x1 = blockNum[1] * rowLength
                y1 = blockNum[0] * colLength
                x2 = (blockNum[1] + 1) * rowLength
                y2 = (blockNum[0] + 1) * colLength
                cv2.rectangle(uFrame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.imwrite(file_path, uFrame)
        else:
            unusualFramesNoDetected.append("Normal")

        count += 1
    # print("unusualFramesNoDetected", unusualFramesNoDetected)
    return unusualFramesArr, unusualFramesNoDetected


def constructMinDistMatrix(megaBlockMotInfVal, codewords, noOfRows, noOfCols, vid):
    global test1OriginalFrames
    threshold = 5.83682407063e-05

    n = 2
    minDistMatrix = np.zeros((len(megaBlockMotInfVal[0][0]), (noOfRows / n), (noOfCols / n)))
    for index, val in np.ndenumerate(megaBlockMotInfVal[..., 0]):
        eucledianDist = []
        for codeword in codewords[index[0]][index[1]]:
            temp = [list(megaBlockMotInfVal[index[0]][index[1]][index[2]]), list(codeword)]

            dist = np.linalg.norm(megaBlockMotInfVal[index[0]][index[1]][index[2]] - codeword)

            eucDist = (sum(map(square, map(diff, zip(*temp))))) ** 0.5

            eucledianDist.append(eucDist)

        minDistMatrix[index[2]][index[0]][index[1]] = min(eucledianDist)
    unusual = {}
    for i in range(len(minDistMatrix)):
        if (np.amax(minDistMatrix[i]) > threshold):
            unusual[i] = []
            for index, val in np.ndenumerate(minDistMatrix[i]):

                if (val > threshold):
                    unusual[i].append((index[0], index[1]))
    # print(unusual)
    unusualFramesArr, unusualFramesNoDetected = showUnusualActivities(unusual, vid, noOfRows, noOfCols, n)
    if len(test1OriginalFrames) < len(unusualFramesNoDetected):
        for i in range(len(unusualFramesNoDetected) - len(test1OriginalFrames)):
            test1OriginalFrames.append('Abnormal')
    return unusual, unusualFramesArr, unusualFramesNoDetected, test1OriginalFrames


def test_from_video(vid, video_name):
    unusualFramesCount = 0
    listOfUnusualFrames = []
    try:
        '''
                calls all methods to test the given video
        '''
        print = "Test video ", vid
        MotionInfOfFrames, rows, cols = mig.getMotionInfuenceMap(vid)

        megaBlockMotInfVal = cmb.createMegaBlocks(MotionInfOfFrames, rows, cols)

        np.save(MEDIA_ROOT + "output\\mega_block_motion_val_{}_k5.npy".format(video_name), megaBlockMotInfVal)

        codewords = np.load(MEDIA_ROOT + "output\\code_words_train_k5.npy")
        # print("codewords", codewords)
        unusualFramesCount, listOfUnusualFrames, unusualFramesNoDetected, originalFramesDetected = constructMinDistMatrix(megaBlockMotInfVal, codewords, rows, cols, vid)

        try:
            skplt.metrics.plot_confusion_matrix(originalFramesDetected, unusualFramesNoDetected, figsize=(12, 12))
            plt.savefig(MEDIA_ROOT + "output\\confusion_matrix.png", dpi=300)
            acc = accuracy_score(originalFramesDetected, unusualFramesNoDetected)

            precision, recall, fscore, support = score(originalFramesDetected, unusualFramesNoDetected)
            data = {'Accuracy': round(acc * 100, 2),
                    'Precision': round(precision[1] * 100, 2),
                    'Recall': round(recall[1] * 100, 2),
                    'FScore': round(fscore[1] * 100, 2)}
            keys = list(reversed(list(data.keys())))
            values = list(reversed(list(data.values())))
            fig, ax = plt.subplots(figsize=(10, 5))

            # creating the bar plot
            plt.bar(keys, values)

            plt.xlabel("Metrics")
            plt.ylabel("Value")
            plt.title("Identified Metrics")

            ax.xaxis.get_label().set_size(16)

            ax.yaxis.get_label().set_size(16)

            for index, data in enumerate(values):
                ax.text(x=index - 0.15, y=data / 2, s=data, fontdict=dict(fontsize=20))

            plt.savefig(MEDIA_ROOT + "output\\performance_graph.png", dpi=300)
            # plt.show()
        except Exception as e:
            print(e)
            traceback.print_exc()
    except Exception as e:
        print(e)
        traceback.print_exc()
        return False, unusualFramesCount, listOfUnusualFrames
    return True, unusualFramesCount, listOfUnusualFrames
