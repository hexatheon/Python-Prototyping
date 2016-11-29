import pandas as pd
import numpy as np
import cPickle as pickle
import os


from alf.instrument import Instrument
from alf.Utils.exchangeInfo import \
    ExchangeInfo, \
    SecurityDefinitionGroups, \
    SecurityDefinitionFile
from alf_common.Constants.UpdateActions import tradeActionsEnumList, \
    updateActions
from alf_common.Constants.Sides import sides
from proxent_utils.symbols import expiration_boundary

from proxent_trading_core.productMapper import ProductMapper
from proxent_trading_core.tvCalculator import TheoreticalValueFileLoader
from proxent_trading_core.utils import \
    filterInstrumentsWithinExpiration, \
    getAssetInstrumentsByGroup

from proxent_config.reader import ConfigLoader
from proxent_utils.logger import BaseLogger
from proxent_utils.file_system import makedirs

from proxent_top_tracker.tracker import ConsolidatedTopTracker, TopExistsEnum

MODEL_NAME = "top_and_fifo_fills"


class TheoreticalValueReference:
    def __init__(self,
                 symbols,
                 mapper,
                 tv_path,
                 tv_var_path):

        self.loader = TheoreticalValueFileLoader(tv_path, tv_var_path)
        self.spread_mapper = mapper
        self.symbols = symbols

    def calculate_spread_tv(self, outright_tvs, spread_symbol):
        left, right = self.spread_mapper(spread_symbol)
        left_tv = outright_tvs[left]
        right_tv = outright_tvs[right]

        spread_tv = left_tv - right_tv

        return spread_tv

    def get_tv(self, time, product):
        tv = self.loader.getTheoreticalValueAsOfTimestamp(time)[0]

        if product in self.symbols["outrights"]:
            return tv[product]

        elif product in self.symbols["spreads"]:
            return self.calculate_spread_tv(tv.values, product)

        elif product in self.symbols["butterflies"]:
            return None
            # return self.calculateBufly(tv.values, product)
        else:
            raise NotImplementedError("Product type not implemented")


class TopFillTracker:
    def __init__(self, config_file):
        config = ConfigLoader(config_file)
        self.bid_previous_top_quantity = None
        self.ask_previous_top_quantity = None
        self.fills = pd.DataFrame()

    def __add_fill__(self, time, trade_price, trade_quantity,
                     creation_time, side):
        fill = pd.DataFrame({
            'topCreationTime': creation_time.value,
            'side': side,
            'fillPrice': trade_price,
            'fillQty': trade_quantity},
            index=[time], columns=['topCreationTime', 'side',
                                   'fillPrice', 'fillQty'])

        self.fills = self.fills.append(fill)

    def save(self, path):
        self.fills.to_csv(path)

    def process(self, time, update_action, trade_price, trade_quantity,
                bid_exists, ask_exists, bid_status, ask_status):

        if bid_exists != TopExistsEnum.EXISTS:
            self.bid_previous_top_quantity = None
        else:
            if update_action == updateActions["New"]:
                self.bid_previous_top_quantity = bid_status["quantity"]
            else:
                if bid_status["quantity"] < self.bid_previous_top_quantity \
                        and not bid_status["potentially_cancelled"] \
                        and not bid_status["revised"]:
                    # if trade happens
                    self.__add_fill__(time, trade_price, trade_quantity,
                                      bid_status["creation_time"],
                                      bid_status["side"])
                    self.bid_previous_top_quantity = bid_status["quantity"]

        if ask_exists != TopExistsEnum.EXISTS:
            self.ask_previous_top_quantity = None
        else:
            if update_action == updateActions["New"]:
                self.ask_previous_top_quantity = ask_status["quantity"]
            else:
                if ask_status["quantity"] < self.ask_previous_top_quantity \
                        and ask_status["potentially_cancelled"] == False \
                        and ask_status["revised"] == False:
                    self.__add_fill__(time, trade_price, trade_quantity,
                                      ask_status["creation_time"],
                                      ask_status["side"])
                    self.ask_previous_top_quantity = ask_status["quantity"]


class FifoFillTracker:
    def __init__(self, config_file):
        config = ConfigLoader(config_file)
        self.min_fill_quantity = config.model_config(
            MODEL_NAME)["min_pro_rata_fill"]
        self.bid_target = None
        self.ask_target = None
        self.output = pd.DataFrame()

    def save(self, path):
        self.output.to_csv(path)

    def __search_fifo__(self, data):
        data.ffill(inplace=True)

        bid_no_top = np.where(
            ((data["tradeAggSide"] == sides["Ask"]) |
             ((data["tradeAggSide"] == sides["Unknown"]) &
              (data["tradePrice"] == data.bidPrice_1)))
            & (data["bidTop"] == TopExistsEnum.DOES_NOT_EXIST) &
            (data.updateAction == updateActions["TradeDetail"])
            & ((data.updateAction.shift(1) == updateActions["TradeSummary"]) |
               (data.updateAction.shift(1) ==
                updateActions["TradeDetailAgg"])))[0]

        bid_with_top = np.where(
            ((data["tradeAggSide"] == sides["Ask"]) |
             ((data["tradeAggSide"] == sides["Unknown"]) &
              (data["tradePrice"] == data.bidPrice_1)))
            & (data["bidTop"] == TopExistsEnum.EXISTS)
            & (data["updateAction"] == updateActions["TradeDetail"])
            & ((data["updateAction"].shift(1) ==
                updateActions["TradeSummary"]) |
               (data["updateAction"].shift(1) ==
                updateActions["TradeDetailAgg"]))
            & (data["updateAction"].shift(-1) ==
               updateActions["TradeDetail"]))[0]

        ask_no_top = np.where(
            ((data["tradeAggSide"] == sides["Bid"]) |
             ((data["tradeAggSide"] == sides["Unknown"])
              & (data["tradePrice"] == data.askPrice_1)))
            & (data["askTop"] == TopExistsEnum.DOES_NOT_EXIST)
            & (data["updateAction"] == updateActions["TradeDetail"])
            & ((data["updateAction"].shift(1) ==
                updateActions["TradeSummary"]) |
               (data["updateAction"].shift(1) ==
                updateActions["TradeDetailAgg"])))[0]

        ask_with_top = np.where(
            ((data["tradeAggSide"] == sides["Bid"]) |
             ((data["tradeAggSide"] == sides["Unknown"]) &
              (data["tradePrice"] == data.askPrice_1)))
            & (data["askTop"] == TopExistsEnum.EXISTS)
            & (data["updateAction"] == updateActions["TradeDetail"])
            & ((data["updateAction"].shift(1) ==
                updateActions["TradeSummary"]) |
               (data["updateAction"].shift(1) ==
                updateActions["TradeDetailAgg"]))
            & (data["updateAction"].shift(-1) ==
               updateActions["TradeDetail"]))[0]

        # bid / ask with top gives you the index with top. use the next index

        self.bid_target = np.concatenate((bid_no_top, (bid_with_top + 1)))
        self.ask_target = np.concatenate((ask_no_top, (ask_with_top + 1)))
        self.bid_target.sort()
        self.ask_target.sort()

    def __find_fifo__(self, data, target, side):
        for idx in target:
            prev = data.iloc[idx].tradeQty
            while data.iloc[idx]["updateAction"] == \
                    updateActions["TradeDetail"] \
                    and data.iloc[idx].tradeQty > self.min_fill_quantity:
                if data.iloc[idx].tradeQty <= prev:
                    # Order is correct based on Pro-rata book
                    pass
                else:
                    row = data.iloc[idx]
                    prev_qty = data.iloc[idx - 1].tradeQty if \
                        data.iloc[idx - 1]["updateAction"] == \
                        updateActions["TradeDetail"] else 0
                    next_qty = data.iloc[idx + 1].tradeQty if \
                        data.iloc[idx + 1]["updateAction"] == \
                        updateActions["TradeDetail"] else 0
                    fill_qty = row.tradeQty - (prev_qty + next_qty) / 2.0
                    self.__add_fill__(row.name, row["tradePrice"],
                                      fill_qty, side)
                prev = data.iloc[idx].tradeQty
                idx += 1
            while data.iloc[idx]["updateAction"] == \
                    updateActions["TradeDetail"] \
                    and data.iloc[idx].tradeQty == 2:
                idx += 1
            while data.iloc[idx]["updateAction"] == \
                    updateActions["TradeDetail"]:
                row = data.iloc[idx]
                self.__add_fill__(row.name, row["tradePrice"],
                                  row.tradeQty, side)
                idx += 1

    def __add_fill__(self, time, trade_price, trade_quantity, side):
        add = pd.DataFrame({'side': side, 'fillPrice': trade_price,
                            'fillQty': trade_quantity}, index=[time],
                           columns=['side', 'fillPrice', 'fillQty'])
        self.output = self.output.append(add)

    def process(self, data):
        self.__search_fifo__(data)
        self.__find_fifo__(data, self.bid_target, sides["Bid"])
        self.__find_fifo__(data, self.ask_target, sides["Ask"])


class CmeTopAndFifoFillsTracker(BaseLogger):
    def __init__(self, config_file):
        self.model_name = MODEL_NAME
        self.config_file = config_file
        self.config_loader = ConfigLoader(config_file)

        self.file_name = self.config_loader.model_config(
            self.model_name)["data_file_name"]
        self.global_config = self.config_loader.global_config()
        self.model_config = self.config_loader.model_config(
            "top_and_fifo_fills")

        self.tz = self.global_config["tz"]
        self.exchange = self.global_config["exchange"]
        self.asset = self.global_config["asset"]

        self.top_fills = pd.DataFrame()
        self.fifo_fills = pd.DataFrame()
        self.filtered_symbols = None
        self.spreads_mapper = None
        self.butterfly_mapper = None # to be developed

    def process(self, start_time, end_time):
        start = pd.Timestamp(start_time, tz=self.tz)
        end = pd.Timestamp(end_time, tz=self.tz)

        trade_date = ExchangeInfo.roundTimestampToTradeDate(self.exchange, end)

        instruments = getAssetInstrumentsByGroup(
            self.exchange, trade_date, self.asset)
        sec_defs, _, legs = SecurityDefinitionFile(
            self.exchange, "%s 00:00:00" % trade_date).getAllDefinitions()
        exp_boundary = trade_date + pd.DateOffset(years=5, months=2)
        filtered_instruments = filterInstrumentsWithinExpiration(
            instruments, legs, exp_boundary)
        self.spreads_mapper = ProductMapper(
            filtered_instruments, legs).getSpreadLegBuySellIdxs

        def_groups = SecurityDefinitionGroups(self.exchange, self.asset,
                                              timestamp=trade_date)
        exp_boundary = expiration_boundary(
            trade_date, self.model_config["num_outrights"])
        outrights = def_groups.getOutrights()
        outrights = outrights[
            outrights["maturityMonthYear"].astype(int) < exp_boundary]
        outrights.sort_values(
            ["securitySubType", "maturityMonthYear"], inplace=True)
        outrights = outrights['symbol'].tolist()

        calendar_types = self.model_config["calendar_categories"]
        butterfly_types = self.model_config["butterfly_categories"]

        spreads = []
        butterfly = []

        filtered_spreads = filtered_instruments['spreads']['symbol'].values
        for cal in calendar_types:
            calendars = def_groups.getCalendars("{}-month".format(cal))
            calendars.sort_values(["securitySubType", "maturityMonthYear"],
                                  inplace=True)
            calendars = calendars['symbol'].tolist()
            for spread in calendars:
                if spread in filtered_spreads:
                    spreads.append(spread)

        filtered_butterflies = \
            filtered_instruments['butterflies']['symbol'].values
        for bfly in butterfly_types:
            butterflies = def_groups.getButterflies("{}-month".format(bfly))
            butterflies.sort_values(
                ["securitySubType", "maturityMonthYear"], inplace=True)
            butterflies = butterflies['symbol'].tolist()
            for b in butterflies:
                if b in filtered_butterflies:
                    butterfly.append(b)

        self.filtered_symbols = {
            "outrights": outrights,
            "spreads": spreads,
            "butterflies": butterfly
        }

        top, fifo = self.__select_fills__(start, end)

        # the following four lines are for fast debugging.
        # It allows you to avoid repeatedly doing time-consuming select fills 
        #top.to_pickle(start.strftime('%Y-%m-%d') + ' top.cpkl')
        #fifo.to_pickle(start.strftime('%Y-%m-%d') + ' fifo.cpkl')

        #top = pd.read_pickle(start.strftime('%Y-%m-%d') + ' top.cpkl')
        #fifo = pd.read_pickle(start.strftime('%Y-%m-%d') + ' fifo.cpkl')



        # the rename and reset change the timestamp index to a new column, and
        # reset back to numerical index to avoid duplicated indexing which
        # screws merging

        top.index.rename('fillTime', inplace=True)
        fifo.index.rename('fillTime', inplace=True)

        top.reset_index(inplace=True)
        fifo.reset_index(inplace=True)

        top, fifo = self.__calculate_tv__(top, fifo)

        self.top_fills = top
        self.fifo_fills = fifo

    def __select_fills__(self, start, end):
        symbols = self.filtered_symbols["outrights"] \
        + self.filtered_symbols["spreads"]# + self.filtered_symbols["butterflies"]

        #symbols = ['GEU6-GEM7', 'GEZ6', 'GEH7']

        # ADD IN COLUMNS_NAMES OTHERWISE IN CASE EMPTY SHIT HAPPENS
        
        fifo_fills = pd.DataFrame(columns=['side', 'fillPrice',
                                           'fillQty'])
        top_fills = pd.DataFrame(columns=['topCreationTime', 'side',
                                          'fillPrice', 'fillQty'])

        for sym in symbols:
            #print sym
            self.logger.info("Scanning top and fifo fills for {}".format(sym))
            self.logger.info("Scanning for Top fills")
            symbol = "{}:{}".format(self.exchange, sym)
            inst = Instrument(symbol, start)
            inst.loadData(start, end, tz=self.tz,
                          includeTicks=True,
                          includeBooks=True,
                          includeImpliedBooks=False,
                          includeEvents=False,
                          suppressWarnings=True).convertPricesToDisplay()

            data = inst.mergeLoadedBooksWithTrades(bookType='Outright',
                                                   includeTicks=True)

            if sym not in self.filtered_symbols["outrights"]:
                data[["tradePrice", "bidPrice_1", "askPrice_1",
                      "price"]] /= 100.0

            del inst
            tracker = ConsolidatedTopTracker()
            top_fill_tracker = TopFillTracker(self.config_file)
            fifo_fill_tracker = FifoFillTracker(self.config_file)

            data['bidTop'] = TopExistsEnum.UNKNOWN
            data['askTop'] = TopExistsEnum.UNKNOWN

            bid_top_index = data.columns.get_loc('bidTop')
            ask_top_index = data.columns.get_loc('askTop')

            for idx, row in enumerate(data.iterrows()):
                timestamp = row[0]
                row_data = row[1]

                if row_data["updateAction"] not in tradeActionsEnumList:
                    top_exists_data = tracker.process_update(row)

                    bid_top_exists = tracker.bidTracker.top_exists
                    ask_top_exists = tracker.askTracker.top_exists

                    data.iat[idx, bid_top_index] = bid_top_exists
                    data.iat[idx, ask_top_index] = ask_top_exists
                else:
                    bid_top_exists = tracker.bidTracker.top_exists
                    ask_top_exists = tracker.askTracker.top_exists

                    data.iat[idx, bid_top_index] = bid_top_exists
                    data.iat[idx, ask_top_index] = ask_top_exists

                    top_exists_data = tracker.process_update(row)

                if row_data["level"] not in [0, 1]:
                    continue

                bid_status = tracker.get_status(sides["Bid"])
                ask_status = tracker.get_status(sides["Ask"])

                '''
                 note: The moment a top order completely trades out,
                       bid/askStatus will be None, since there are no tops now.
                       However, this last fill is a valid top fill.
                       This last fill info is returned directly by
                       tracker.process_update in top_exists_data
                       The following step modifies the top_exists_data to
                       replace the empty bid/askStatus
                '''

                if top_exists_data:
                    if top_exists_data["side"] == 1:
                        bid_status = top_exists_data
                    else:
                        ask_status = top_exists_data

                top_fill_tracker.process(
                    timestamp, row_data["updateAction"], row_data['tradePrice'],
                    row_data.tradeQty, bid_top_exists, ask_top_exists,
                    bid_status, ask_status)

            self.logger.info("Scanning for FIFO fills")

            fifo_fill_tracker.process(data)

            fifo_fill_tracker.output["symbol"] = sym
            fifo_fills = fifo_fills.append(fifo_fill_tracker.output)
            #fifo_fills["symbol"] = sym
            self.logger.info("Found {} FIFO Fills for {}".format(
                len(fifo_fills), sym))

            top_fill_tracker.fills["symbol"] = sym
            top_fills = top_fills.append(top_fill_tracker.fills)
            #top_fills["symbol"] = sym
            self.logger.info("Found {} Top Fills for {}".format(
                len(top_fills), sym))

        fifo_fills.sort_index(inplace=True)
        top_fills.sort_index(inplace=True)

        return top_fills, fifo_fills

    def __calculate_tv__(self, top, fifo):
        timeframes = self.model_config["pnl_timeframes"]
        tv_dir = self.model_config["tv_dir"]
        tv_path = os.path.join(tv_dir, self.model_config["tv_file"])
        tv_var_path = os.path.join(tv_dir, self.model_config["tv_var_file"])

        tv_calc = TheoreticalValueReference(
            self.filtered_symbols, self.spreads_mapper, tv_path, tv_var_path)

        tvderive = tv_calc.get_tv


        def get_tv(row):
            rv = {"TV: {}".format(x): np.nan for x in timeframes}
            for offset in timeframes:
                timestamp = row.fillTime + pd.tseries.frequencies.to_offset(
                    offset)
                tv_column = "TV: {}".format(offset)
                rv[tv_column] = tv_calc.get_tv(timestamp, row["symbol"])
            return pd.Series(rv)


        def calculate_pl(row):
            rv = {"PL: {}".format(x): np.nan for x in timeframes}
            for offset in timeframes:
                tv_column = "TV: {}".format(offset)
                pl_column = "PL: {}".format(offset)
                if row.side == 1:
                    rv[pl_column] = row[tv_column] - row['fillPrice']
                else:
                    rv[pl_column] = row['fillPrice'] - row[tv_column]
            return pd.Series(rv)

        top = top.join(top.apply(get_tv, axis = 1))
        top = top.join(top.apply(calculate_pl, axis = 1))

        fifo = fifo.join(fifo.apply(get_tv, axis = 1))
        fifo = fifo.join(fifo.apply(calculate_pl, axis = 1))

        # top_tv = top.apply(get_tv, axis=1)
        # top = top.join(top_tv)

        # top_pl = top.apply(calculate_pl, axis=1)
        # top = top.join(top_pl)

        # fifo_tv = fifo.apply(get_tv, axis=1)
        # fifo = fifo.join(fifo_tv)

        # fifo_pl = fifo.apply(calculate_pl, axis=1)
        # fifo = fifo.join(fifo_pl)

        return top, fifo
