import numpy as np
import os
import struct

def cnn_features(path):
    listdir = sorted(os.listdir(path))
    feature = []
    for feat in listdir:
        with open(os.path.join(path, feat), 'rb') as f:
            map = f.read()
        feature.append(np.frombuffer(map, dtype=np.float32).reshape((14, 14, 512)))
    return feature

def get_GAP(feature):
    GAP = []
    for feat in feature:
        GAP.append(np.mean(feat, axis = (0, 1)))
    return np.array(GAP)

def get_std(feature):
    std = []
    for feat in feature:
        std.append(np.std(feat, axis = (0, 1)))
    return np.array(std)

def local_GMP(feature):
    LGmp = []
    for feat in feature: 

        feat1 = feat[:7, :7, :]
        feat2 = feat[:7, 7:, :]
        feat3 = feat[7:, :7, :]
        feat4 = feat[7:, 7:, :]
        GMP1 = np.max(feat1, axis = (0, 1))
        GMP2 = np.max(feat2, axis = (0, 1))
        GMP3 = np.max(feat3, axis = (0, 1))
        GMP4 = np.max(feat4, axis = (0, 1))

        all = GMP1 + GMP2 + GMP3 + GMP4
        LGmp.append(all)
    return np.array(LGmp)

def local_GAP(feature):
    LGAP = []
    for feat in feature: #2x2로 나누어서 각각 gap를 구함

        feat1 = feat[:7, :7, :]
        feat2 = feat[:7, 7:, :]
        feat3 = feat[7:, :7, :]
        feat4 = feat[7:, 7:, :]
        GAP1 = np.mean(feat1, axis = (0, 1))
        GAP2 = np.mean(feat2, axis = (0, 1))
        GAP3 = np.mean(feat3, axis = (0, 1))
        GAP4 = np.mean(feat4, axis = (0, 1))

        all = GAP1 + GAP2 + GAP3 + GAP4
        LGAP.append(all)

    return np.array(LGAP)

def main():
    path = './features/cnn/'
    feature = cnn_features(path)

    GAP = get_GAP(feature)
    std = get_std(feature)
    LGmp = local_GMP(feature)
    LGap = local_GAP(feature)

    final = np.concatenate((LGap,GAP,LGmp,std), axis = 1)

    with open(f'A3_2020314069.des', 'wb') as f:
        f.write(struct.pack('ii', 2000, 2048))
        f.write(final.astype('float32').tobytes())

if __name__ == '__main__':
    main()