import argparse
from proxent_largest_order.tracker import dataGenerator
from proxent_largest_order.tracker import dataLoader
from proxent_largest_order.model_generator import LargestOrderOnLevelModelGenerator



parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', action='store', dest="config_file")
parser.add_argument('-s', '--start', action="store", dest= "startDate", required=True)
parser.add_argument('-e', '--end', action="store", dest="endDate", required=True)
# group can be outrights, spreads, or butterflies
parser.add_argument('-g', '--group', action="store", dest="definition", required=True)


parser.add_argument('-out','--outputFile', action = "store", dest = "outputFile", default = 'modelFile.npy')
arguments = parser.parse_args()


# first track for largest order data
# returns the directory name in which all small csv files are stored
generatedDirPath = dataGenerator(arguments.config_file,
                                 arguments.definition,
                                 arguments.startDate,
                                 arguments.endDate)

xdata, ydata = dataLoader(generatedDirPath)

modelGenerator = LargestOrderOnLevelModelGenerator(arguments.config_file)

modelGenerator.generateModel(xdata, ydata)

modelGenerator.saveModelToFile(arguments.outputFile)
