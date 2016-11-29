import datetime
import pandas as pd
import numpy as np
import cPickle as pickle
# from pandas.tseries.offsets import Day
import os
import re

import matplotlib.pyplot as plt

from alf.Utils.exchangeInfo import ExchangeInfo, SecurityDefinitionGroups
from proxent_config.reader import ConfigLoader

from proxent_reports.base import AbstractReportDataWriter, \
    AbstractReportDataReader, get_file_path
from proxent_top_fifo.model import CmeTopAndFifoFillsTracker
from proxent_utils.symbols import expiration_boundary

REPORT_NAME = "cme_top_and_fifo_fills"


class CmeTopAndFifoFillsReportDataReader(AbstractReportDataReader):
    def __init__(self, config_file):
        super(CmeTopAndFifoFillsReportDataReader, self).__init__(config_file)
        self.report_name = REPORT_NAME
        self.file_name = self.config_loader.report_config(
            self.report_name)["data_file_name"]

    def __read_intraday__(self):
        trade_date = ExchangeInfo.roundTimestampToTradeDate(
            self.config_loader.global_config()["exchange"],
            pd.Timestamp(datetime.datetime.now().strftime("%Y-%m-%d"),
                         tz="UTC").tz_convert(
                         self.config_loader.global_config()["tz"]))

        rv_data = self.__read_archive__(trade_date)

        file_path = os.path.join(self.state_dir, REPORT_NAME, self.file_name)

        rv_state = None

        if os.path.exists(file_path):
            input_file = open(os.path.join(file_path), 'rb')
            rv_state = pickle.load(input_file)
            input_file.close()

        return rv_data, rv_state

    def __read_archive__(self, trade_date):
        file_path = get_file_path(trade_date,
                                  self.report_dir,
                                  self.report_name,
                                  self.file_name)

        rv = None
        if os.path.exists(file_path):
            input_file = open(os.path.join(file_path), 'rb')
            rv = pickle.load(input_file)
            input_file.close()

        return rv


class CmeTopAndFifoFillsReportDataWriter(AbstractReportDataWriter):
    def __init__(self, config_file, mode="archive"):
        super(CmeTopAndFifoFillsReportDataWriter, self).__init__(
            config_file, mode)

        self.report_name = REPORT_NAME

        self.file_name = self.config_loader.report_config(
            self.report_name)["data_file_name"]

        self.reader = CmeTopAndFifoFillsReportDataReader(self.config_file)

        self.data["top"] = pd.DataFrame()
        self.data["fifo"] = pd.DataFrame()

        if mode is "intraday":
            self.state = self.reader.load_state()

    def __save_archive__(self, trade_date):
        file_path = super(
            CmeTopAndFifoFillsReportDataWriter, self).__save_archive__(
            trade_date=trade_date)

        file_path = os.path.join(file_path, self.file_name)

        output = open(file_path, "wb")
        pickle.dump(self.data, output)
        output.close()

        self.logger.info("Saved the archived data to {}".format(file_path))

    def __save_intraday__(self, start_time, end_time):
        tz = self.config_loader.global_config()["tz"]
        start = pd.Timestamp(start_time, tz=tz)
        end = pd.Timestamp(end_time, tz=tz)

        file_path = super(
            CmeTopAndFifoFillsReportDataWriter, self).__save_intraday__(
            start_time=start, end_time=end)

        output = open(file_path, "wb")
        pickle.dump(self.state, output)
        output.close()

        self.logger.info("Saved the intraday data to {}".format(file_path))

    def __process_archive__(self, trade_date):
        # Warning: it seems that TV calculator can only start from 00:00
        # start_time = (trade_date - Day()).strftime("%Y-%m-%d 17:00:00")

        start_time = trade_date.strftime("%Y-%m-%d") + ' ' \
                     + self.config_loader.report_config(
                        self.report_name)["start_filter_time"]

        end_time = trade_date.strftime("%Y-%m-%d") + ' ' \
                   + self.config_loader.report_config(
                    self.report_name)["stop_filter_time"]
        self.__process_intraday__(start_time, end_time)

    def __process_intraday__(self, start_time, end_time, ignore_state=False):
        if ignore_state or self.mode == "archive" or len(self.state) == 0:
            self.state = {"top_state": pd.DataFrame(),
                          "fifo_state": pd.DataFrame()}

        tracker = CmeTopAndFifoFillsTracker(self.config_file)

        tracker.process(start_time, end_time)

        self.state["top_state"] = pd.concat((self.state["top_state"], 
                                            tracker.top_fills))
        self.state["fifo_state"] = pd.concat((self.state["fifo_state"], 
                                            tracker.fifo_fills))

        self.data["top"] = self.state["top_state"]
        self.data["fifo"] = self.state["fifo_state"]

    def get_report_data(self):
        return self.data


class CmeTopAndFifoFillsAnalysisReport(object):

    def __init__(self, config_file, trade_date):
        config = ConfigLoader(config_file)
        self.report_config = config.report_config(REPORT_NAME)
        self.global_config = config.global_config()
        self.plot_config = config.plot_config()
        color_theme = config.color_theme()
        self.primary_color = color_theme['primary']['0']
        self.secondary_color1 = color_theme['secondary-1']['0']
        self.secondary_color2 = color_theme['secondary-2']['0']

        reader = CmeTopAndFifoFillsReportDataReader(config_file)
        self.data = reader.read(trade_date=trade_date)

        self.all_top_pnl = self.data["top"]
        self.all_fifo_pnl = self.data["fifo"]

        reg = re.compile("PL")

        pnl_columns = [x for x in self.all_top_pnl.columns if reg.search(x)]

        for col in pnl_columns:
            self.all_top_pnl[col] = self.all_top_pnl[col] * 100.0 / \
                                    self.global_config["default_tick_size"]

        pnl_columns = [x for x in self.all_fifo_pnl.columns if reg.search(x)]
        for col in pnl_columns:
            self.all_fifo_pnl[col] = self.all_fifo_pnl[col] * 100.0 / \
                                     self.global_config["default_tick_size"]

        self.num_outrights = self.report_config["num_outrights"]
        self.pnl_timeframes = self.report_config["pnl_timeframes"]

        self.def_groups = SecurityDefinitionGroups(
            self.global_config["exchange"], self.global_config["asset"],
            trade_date)
        self.exp_boundary = expiration_boundary(
            dt=pd.Timestamp(trade_date, tz=self.global_config["tz"]),
            num_outrights=self.num_outrights)

        self.hist_bins = np.linspace(self.report_config["min_pnl_ticks"],
                                     self.report_config["max_pnl_ticks"],
                                     self.report_config["histogram_bins"])

        self.top_pnl = {}
        self.fifo_pnl = {}

        self.__prepare_pnl_data__()

    def __prepare_pnl_data__(self):
        outrights = self.def_groups.getOutrights()
        outrights = outrights[
            outrights["maturityMonthYear"].astype(int) <
            self.exp_boundary]["symbol"].values

        outrights_top_pnl = self.all_top_pnl[
            self.all_top_pnl["symbol"].isin(outrights)]
        outrights_fifo_pnl = self.all_fifo_pnl[
            self.all_fifo_pnl["symbol"].isin(outrights)]

        categories = self.report_config["calendar_categories"]

        calendars_top_pnl = pd.DataFrame()
        calendars_fifo_pnl = pd.DataFrame()

        for cat in categories:
            cals = self.def_groups.getCalendars("{}-month".format(cat))
            symbols = cals["symbol"].values

            pnl = self.all_top_pnl[self.all_top_pnl["symbol"].isin(symbols)]
            pnl["symbol"] = "{}-month".format(cat)
            calendars_top_pnl = pd.concat((calendars_top_pnl, pnl))

            pnl = self.all_fifo_pnl[self.all_fifo_pnl["symbol"].isin(symbols)]
            pnl["symbol"] = "{}-month".format(cat)
            calendars_fifo_pnl = pd.concat((calendars_fifo_pnl, pnl))

        categories = self.report_config["butterfly_categories"]

        butterfly_top_pnl = pd.DataFrame()
        butterfly_fifo_pnl = pd.DataFrame()

        for cat in categories:
            bfly = self.def_groups.getButterflies("{}-month".format(cat))
            symbols = bfly["symbol"].values

            pnl = self.all_top_pnl[self.all_top_pnl["symbol"].isin(symbols)]
            pnl["symbol"] = "{}-month".format(cat)
            butterfly_top_pnl = pd.concat((butterfly_top_pnl, pnl))

            pnl = self.all_fifo_pnl[self.all_fifo_pnl["symbol"].isin(symbols)]
            pnl["symbol"] = "{}-month".format(cat)
            butterfly_fifo_pnl = pd.concat((butterfly_fifo_pnl, pnl))

        self.top_pnl["outrights"] = outrights_top_pnl
        self.fifo_pnl["outrights"] = outrights_fifo_pnl
        self.top_pnl["calendars"] = calendars_top_pnl
        self.fifo_pnl["calendars"] = calendars_fifo_pnl
        self.top_pnl["butterflies"] = butterfly_top_pnl
        self.fifo_pnl["butterflies"] = butterfly_fifo_pnl

    def plot_distribution(self, category, weighted=False, latex=True):
        self.__plot_distribution__(category=category, weighted=weighted,
                                   latex=latex)

    def __plot_distribution__(self, category, weighted, latex=True):
        if latex:
            self.__plot_distribution_latex__(category=category,
                                             weighted=weighted)
        else:
            self.__plot_distribution_html__(category=category,
                                            weighted=weighted)

    def __plot_distribution_latex__(self, category, weighted):
        weighted_title = "Quantity Weighted " if weighted else ""
        if category is "outrights":
            fig, ax_arr = plt.subplots(len(self.pnl_timeframes), 2, sharex=True,
                                       sharey=True)
            top = self.top_pnl["outrights"]
            fifo = self.fifo_pnl["outrights"]
            self.__plot_comparison__(top=top, fifo=fifo, ax_arr=ax_arr,
                                     weighted=weighted)
            fig.suptitle("Outrights {}P/L Distributions (% of a Tick)".format(
                weighted_title), fontsize=self.plot_config["suptitle_size"])
            plt.gcf().set_size_inches(15, 15)
            plt.show()
        elif category is "calendars":
            cal_top = self.top_pnl["calendars"]
            cal_fifo = self.fifo_pnl["calendars"]
            for category in self.report_config["calendar_categories"]:
                fig, ax_arr = plt.subplots(len(self.pnl_timeframes), 2,
                                           sharex=True,
                                           sharey=True)
                category = "{}-month".format(category)
                top = cal_top[cal_top["symbol"] == category]
                fifo = cal_fifo[cal_fifo["symbol"] == category]
                self.__plot_comparison__(top=top, fifo=fifo, ax_arr=ax_arr,
                                         weighted=weighted)
                fig.suptitle("{} Calendars {}P/L Distributions (% of a Tick)"
                             "".format(category, weighted_title),
                             fontsize=self.plot_config["suptitle_size"])
                plt.gcf().set_size_inches(15, 15)
                plt.show()

        elif category is "butterflies":
            bfly_top = self.top_pnl["calendars"]
            bfly_fifo = self.fifo_pnl["calendars"]
            for category in self.report_config["calendar_categories"]:
                fig, ax_arr = plt.subplots(len(self.pnl_timeframes), 2,
                                           sharex=True,
                                           sharey=True)
                category = "{}-month".format(category)
                top = bfly_top[bfly_top["symbol"] == category]
                fifo = bfly_fifo[bfly_fifo["symbol"] == category]
                self.__plot_comparison__(top=top, fifo=fifo, ax_arr=ax_arr,
                                         weighted=weighted)
                fig.suptitle("{} Butterfly {}P/L Distributions (% of a Tick)"
                             "".format(category, weighted_title),
                             fontsize=self.plot_config["suptitle_size"])
                plt.gcf().set_size_inches(15, 15)
                plt.show()

    def __plot_distribution_html__(self, category, weighted=True):
        pass

    def __plot_comparison__(self, top, fifo, ax_arr, weighted):
        for idx, offset in enumerate(self.pnl_timeframes):
            col = "PL: {}".format(offset)
            if weighted:
                mean = np.average(top[col].values,
                                  weights=top["fillQty"].values)
            else:
                mean = np.mean(top[col])

            std = np.std(top[col].values)
            count = len(top)

            title = "Top Order Fills Pnl: {}\n" \
                    "Mean: {} | Std: {} | Count: {}".format(offset,
                                                            np.round(mean, 2),
                                                            np.round(std, 2),
                                                            count)
            weights = top["fillQty"].values if weighted else None
            self.__plot_histogram__(ax_arr[idx][0], top[col].values,
                                    title, mean, weights)

            if weighted:
                mean = np.average(fifo[col].values,
                                  weights=fifo["fillQty"].values)
            else:
                mean = np.mean(fifo[col])

            std = np.std(fifo[col].values)
            count = len(fifo)

            col = "PL: {}".format(offset)
            title = "FIFO Fills Pnl: {}\nMean: {} | Std: {} | Count: {}".format(
                offset, np.round(mean, 2), np.round(std, 2), count)
            weights = fifo["fillQty"].values if weighted else None
            self.__plot_histogram__(ax_arr[idx][1], fifo[col].values,
                                    title, mean, weights)

    def __plot_histogram__(self, axis, pnl, title, mean, weights=None):
        if weights is not None:
            axis.hist(pnl, bins=self.hist_bins, normed=True,
                      color=self.primary_color, weights=weights)
        else:
            axis.hist(pnl, bins=self.hist_bins, normed=True,
                      color=self.primary_color)
        axis.set_title(title, fontsize=self.plot_config["title_size"])
        axis.axvline(mean, color=self.secondary_color1, linestyle="dashed",
                     linewidth=2)
