import numpy as np


def loadModelFromFile(modelFilename):
    model = np.load(modelFilename)
    toRet = LargestOrderOnLevelModel(model)
    return toRet


class LargestOrderOnLevelModel():
    def __init__(self, model):
        if isinstance(model, str):
            self.model = np.load(model)
        else:
            self.model = model

    def getLargestOrderSize(self, x):
        x = np.atleast_1d(x)
        joint = self.model[0]
        coef = self.model[1]
        newJoint = np.append(joint, x.max())
        output = np.zeros(x.size)
        length = coef.size
        for i in range(0, length):
            inx = np.where((x > newJoint[i]) & (x <= newJoint[i + 1]))
            offset = 0
            for j in range(0, i):
                offset += (coef[j] - coef[j + 1]) * newJoint[j + 1]
            output[inx] = np.array(x)[inx] * coef[i] + offset
        return output
