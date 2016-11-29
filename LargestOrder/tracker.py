import pandas as pd
import glob
import alf.Utils.exchangeInfo as ExchangeInfo
from alf.instrument import Instrument
from os import mkdir, path
from proxent_trading_core.utils import \
    filterInstrumentsWithinExpiration, \
    getAssetInstrumentsByGroup

from proxent_config.reader import ConfigLoader
from proxent_top_tracker.tracker import ConsolidatedTopTracker

from alf_common.Constants.Sides import sides
from alf_common.Constants.UpdateActions import tradeActionsEnumList, \
    updateActions

MODEL_NAME = "largest_order"

# dataGenerator outputs csv files by creating a directory in current working directory
# and returns the name of this new folder. Name is the product category, e.g. outrights

def dataGenerator(config_file, definition, startDate, endDate):
    config_loader = ConfigLoader(config_file)
    asset = config_loader.global_config()["asset"]
    exchange = config_loader.global_config()["exchange"]
    tz = config_loader.global_config()["tz"]

    levelQtyMin = config_loader.model_config(MODEL_NAME)["levelQtyMin"]
    minTimeSpan = config_loader.model_config(MODEL_NAME)["minTimeSpan"]
    minPercentageChange = config_loader.model_config(MODEL_NAME)["minPercentageChange"]
    numLarTargets = config_loader.model_config(MODEL_NAME)["numLarTargets"]
    out_columns = config_loader.model_config(MODEL_NAME)["output_column_names"]


    instruments = getAssetInstrumentsByGroup(exchange, startDate, asset)
    secDefs, _, legs = ExchangeInfo.SecurityDefinitionFile(
        exchange, "%s 00:00:00" % (startDate)).getAllDefinitions()

    expirationBoundary = pd.Timestamp(startDate, tz=tz) + pd.DateOffset(years=5,
                                                                        months=2)
    filteredInstruments = filterInstrumentsWithinExpiration(instruments,
                                                            legs,
                                                            expirationBoundary)
    Symbols = filteredInstruments[definition].symbol.tolist()
    dateList = pd.bdate_range(start=startDate, end=endDate, tz=tz)

    def last_same_side(side, list):
        length = -1 * len(list)
        i = -1
        while i >= length:
            if list[i][4] == side:
                return list[i]
            i -= 1
        return None

    try:
        mkdir(definition)
    except OSError, e:
        if e.errno != 17:
            raise
        else:
            pass

    for date in dateList:
        for sym in Symbols:
            startTime = date - pd.DateOffset(hours=7)
            endTime = date + pd.DateOffset(hours=16)
            #symbol = exchange + ':' + sym
            symbol = "{}:{}".format(exchange, sym)

            inst = Instrument(symbol, startTime)
            inst.loadData(startTime, endTime, tz=tz,
                          includeTicks=True,
                          includeBooks=True,
                          includeImpliedBooks=False,
                          includeEvents=False,
                          suppressWarnings=True).convertPricesToDisplay()

            data = inst.mergeLoadedBooksWithTrades(bookType="Outright",
                                                   includeTicks=True)
            col = ['askPrice_1',
                   'askQty_1',
                   'askOrders_1',
                   'bidPrice_1',
                   'bidQty_1',
                   'bidOrders_1']
            data[col] = data[col].ffill()

            tracker = ConsolidatedTopTracker()

            output_data = []


            for i, info in enumerate(data.iterrows()):
                time = info[0]
                row = info[1]
                if row["level"] != 1 and row["level"] != 0:
                    continue
                tracker.process_update(info)

                if row["updateAction"] != updateActions["TradeSummary"]:
                    continue

                BidTop = True
                AskTop = True
                if tracker.get_status(1) == None:
                    BidTop = False
                if tracker.get_status(2) == None:
                    AskTop = False

                if BidTop == True and AskTop == True:
                    continue

                side = None
                if row["tradePrice"] == row["bidPrice_1"]:
                    if BidTop == True:
                        continue
                    else:
                        side = 1
                elif row["tradePrice"] == row["askPrice_1"]:
                    if AskTop == True:
                        continue
                    else:
                        side = 2
                else:
                    continue

                if row["tradeAggSide"] != sides["Unknown"] \
                    and row["tradeNumOrders"] < (numLarTargets + 2):
                    continue
                if row["tradeAggSide"] == sides["Unknown"] \
                    and row["tradeNumOrders"] < (numLarTargets + 1):
                    continue

                levelQty = row["bidQty_1"] if side == sides["Bid"] \
                                            else row["askQty_1"]

                if levelQty < levelQtyMin:
                    continue

                if len(output_data) > 0:
                    last = last_same_side(side, output_data)
                    if last != None and row["tradePrice"] == last[3]:
                        d = time - last[0]
                        if not (d.seconds > minTimeSpan
                                or abs(levelQty-last[5])/last[5]>minPercentageChange):
                            continue

                tradeTotalQty = 0
                j = i + 1
                while not (data.iloc[j]["updateAction"] == 8
                           and data.iloc[j]["tradePrice"] == row["tradePrice"]):
                    j += 1

                k = j
                while data.iloc[k]["updateAction"] == 8 \
                        and data.iloc[k]["tradePrice"] == row["tradePrice"]:
                    tradeTotalQty += data.iloc[k]["tradeQty"]
                    k += 1

                line = []
                multiple = None
                line.extend((time, symbol, row["tradePrice"], side))
                if side == 1:
                    line.extend((row["bidQty_1"], row["bidOrders_1"]))
                    multiple = row["bidQty_1"] / tradeTotalQty
                elif side == 2:
                    line.extend((row["askQty_1"], row["askOrders_1"]))
                    multiple = row["askQty_1"] / tradeTotalQty
                else:
                    print "SEVERE WARNING: WHAT ON EARTH"
                    exit(1)

                lar1 = data.iloc[j]["tradeQty"] * multiple
                lar2 = data.iloc[j + 1]["tradeQty"] * multiple
                lar3 = data.iloc[j + 2]["tradeQty"] * multiple
                if lar2 > lar1:
                    lar2 = None
                if lar2 == None:
                    if lar3 > lar1:
                        lar3 = None
                else:
                    if lar3 > lar2:
                        lar3 = None
                line.extend((lar1, lar2, lar3))
                output_data.append(line)

            output = pd.DataFrame(output_data, columns=out_columns)
            name = definition + '/' + str(date.month) + '-' + str(date.day) + '-' + sym + '.csv'
            output.to_csv(name)
        print str(date.month) + '-' + str(date.day) + ' finished'
    print 'entire scanning finished'
    return definition + '/'


def dataLoader(dir_path):
    filenames = glob.glob(path.join(dir_path, '*.csv'))
    df = pd.concat(pd.read_csv(f) for f in filenames)
    xdata = df["totalQty"]
    ydata = df["Lar1"]
    return xdata, ydata
