import numpy as np
from scipy.optimize import leastsq
from proxent_config.reader import ConfigLoader

# given an array-like joint, an array-like bin nodes, a depth in percentage
# and two series of data which are totalQty and Lar1
# xdata and ydata must be of same size
MODEL_NAME = "largest_order"

class LargestOrderOnLevelModelGenerator:
    def __init__(self, config_file):
        self.model = None
        report_config = ConfigLoader(config_file).model_config(MODEL_NAME)
        self.joints = report_config["joints"]
        self.percentile = report_config["percentile_depth"]
        self.binNode = report_config["binNode"]

    def percentile_pick(self, array, num):
        new = np.sort(array)
        index = int(np.floor(len(new) * num))
        # print index
        return new[index - 1]

    def percentile_data(self, x, y, binNode, depth):
        newx = np.array([])
        newy = np.array([])
        # x = data.totalQty
        length = len(binNode)

        for i in range(1, length):
            look = np.where((x >= binNode[i - 1]) & (x < binNode[i]))
            if len(look[0]) == 0:
                continue
            # print look
            average = x[look].mean()
            percentile_y = self.percentile_pick(y[look], depth)
            newx = np.append(newx, average)
            newy = np.append(newy, percentile_y)
        return newx, newy


        # Joint should start from 0. e.g. 0, 100, 200, 300, 400, 500

    def fit(self, joint):
        def f(coef, x):
            new_joint = np.append(joint, max(x))
            output = np.zeros(len(x))
            length = len(coef)
            for i in range(0, length):
                inx = np.where((x > new_joint[i]) & (x <= new_joint[i + 1]))
                offset = 0
                for j in range(0, i):
                    offset += (coef[j] - coef[j + 1]) * new_joint[j + 1]
                output[inx] = np.array(x)[inx] * coef[i] + offset
            return output

        return f

    def customized_fit(self, joint, x, y):
        func = self.fit(joint)

        def errf(tpl, x, y):
            return func(tpl, x) - y

        initial = (1,) * len(joint)
        tplfinal, _ = leastsq(errf, initial[:], args=(x, y))
        return tplfinal

    def generateModel(self, xdata, ydata):
        xdata = np.array(xdata)
        ydata = np.array(ydata)
        binNode = np.array(self.binNode)
        binNode = np.append(binNode, max(xdata))
        joints = np.array(self.joints)
        # joint = np.append(joint, max(xdata)) Don't do this here.
        # accurate length of joint is needed in customized_fit

        newx, newy = self.percentile_data(xdata, ydata, binNode, self.percentile)
        coef = self.customized_fit(joints, newx, newy)
        self.model = np.array((joints, coef))
        return coef

    def saveModelToFile(self, filepath):
        np.save(filepath, self.model)
