import datetime
import pandas as pd
from pandas.tseries.frequencies import to_offset
import numpy as np
import cPickle as pickle
from pandas.tseries.offsets import Day
import os
import re

from alf.Utils.exchangeInfo import ExchangeInfo
from alf.packets import Packets
from alf.events import Events
from proxent_config.reader import ConfigLoader

from proxent_reports.base import AbstractReportDataWriter, \
    AbstractReportDataReader, get_file_path


REPORT_NAME = "cme_packets_statistics"


class CmePacketStatisticsReportDataReader(AbstractReportDataReader):
    def __init__(self, config_file):
        super(CmePacketStatisticsReportDataReader, self).__init__(config_file)
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


class CmePacketStatisticsReportDataWriter(AbstractReportDataWriter):

    def __init__(self, config_file, mode="archive"):
        super(CmePacketStatisticsReportDataWriter, self).__init__(config_file,
                                                                  mode)

        self.report_name = REPORT_NAME
        self.report_config = self.config_loader.report_config(
            self.report_name)
        self.file_name = self.report_config["data_file_name"]
        self.reader = CmePacketStatisticsReportDataReader(self.config_file)

        self.data["time_differences"] = pd.DataFrame()
        self.data["trade_packets"] = pd.DataFrame()
        self.data["book_packets"] = pd.DataFrame()
        self.data["rolling_summary"] = pd.DataFrame()

        if mode is "intraday":
            self.state = self.reader.load_state()

        if self.state is None:
            self.state = {"events_frequency": pd.DataFrame(),
                          "raw_packets_events": pd.DataFrame(),
                          "last_record": pd.DataFrame(),
                          "packets_frequency": pd.DataFrame()}

    def __save_archive__(self, trade_date):
        file_path = super(
            CmePacketStatisticsReportDataWriter, self).__save_archive__(
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
            CmePacketStatisticsReportDataWriter, self).__save_intraday__(
            start_time=start, end_time=end)

        output = open(file_path, "wb")
        pickle.dump(self.state, output)
        output.close()

        self.logger.info("Saved the intraday data to {}".format(file_path))

    def __process_archive__(self, trade_date):
        # start from 17:30 of prev day to avoid early time edge cases
        start_time = (trade_date - Day()).strftime("%Y-%m-%d 17:30:00")
        end_time = trade_date.strftime("%Y-%m-%d 16:00:00")
        self.__process_intraday__(start_time, end_time)

    def get_report_data(self):
        return self.data

    def __process_intraday__(self, start_time, end_time, ignore_state=False):
        if ignore_state or self.mode == "archive" or len(self.state) == 0:
            self.state = {"events_frequency": pd.DataFrame(),
                          "raw_packets_events": pd.DataFrame(),
                          "last_record": pd.DataFrame(),
                          "packets_frequency": pd.DataFrame()}
        gc = self.config_loader.global_config()

        start = pd.Timestamp(start_time, tz=gc["tz"])
        end = pd.Timestamp(end_time, tz=gc["tz"])

        self.logger.info("Loading the packets between {} and {}".format(
            start, end))

        packets_loader = Packets(gc["exchange"],  gc["channel_id"])
        packets_loader.loadData(start, end)
        packets = packets_loader.getPackets()
        packets = packets[["captureTime", "bookUpdates", "impBookUpdates",
                           "seqNum", "tradeDetails", "tradeSummaries"]]

        self.logger.info("Finished loading {} packets between {} and {}".format(
            len(packets), start, end))

        self.logger.info("Loading the events between {} and {}".format(
            start, end))

        events_loader = Events(gc["exchange"], gc["channel_id"])
        events_loader.loadData(start, end)
        events = events_loader.getEventSummaries()
        events = events[["eventId", "numPackets", "numStops", "startSeqNum"]]

        self.logger.info("Finished loading {} events between {} and {}".format(
            len(events), start, end))

        del packets_loader
        del events_loader

        # ignore two events with same startSeqNum. keep neither
        events = events.drop_duplicates('startSeqNum', keep=False)

        self.logger.info("Merged packets and events on starting sequence "
                         "number")

        pkts_evts_merged = pd.merge(packets, events, how="left",
                                    left_on="seqNum",
                                    right_on="startSeqNum")
        pkts_evts_merged = pkts_evts_merged.ffill()
        to_drop = np.where(pkts_evts_merged["seqNum"] >=
                           pkts_evts_merged["startSeqNum"] +
                           pkts_evts_merged["numPackets"])

        self.logger.info("Dropping {} records where the sequence number is "
                         "greater than maximum sequence number possible for the"
                         " corresponding event".format(len(to_drop)))

        pkts_evts_merged = pkts_evts_merged.drop(to_drop[0])
        pkts_evts_merged = pd.concat((self.state["last_record"],
                                      pkts_evts_merged))

        self.logger.info("Saving the last record from the raw packets and "
                         "events data")

        self.state["last_record"] = pkts_evts_merged.tail(n=1)

        pkts_evts_merged['timeDifference'] = pkts_evts_merged[
            "captureTime"].diff(1)

        pkts_evts_merged = pkts_evts_merged.drop(0)

        l = len(pkts_evts_merged)

        self.logger.info("Determining the packet type based pn trade details,"
                         "trade summaries, book updates, implied book updates,"
                         "and, stop order")

        pkts_evts_merged['packetType'] = np.packbits(np.transpose(
            [np.zeros(l, dtype=bool), np.zeros(l, dtype=bool),
             np.zeros(l, dtype=bool), np.zeros(l, dtype=bool),
             pkts_evts_merged["numStops"].astype(bool),
             pkts_evts_merged["impBookUpdates"].astype(bool),
             pkts_evts_merged["bookUpdates"].astype(bool),
             np.logical_or(pkts_evts_merged["tradeDetails"],
                           pkts_evts_merged["tradeSummaries"])]))

        pkts_evts_merged["pureTrade"] = np.nan
        pkts_evts_merged.loc[pkts_evts_merged["packetType"] == 1,
                             "pureTrade"] = 1

        pkts_evts_merged.index = pd.DatetimeIndex(
            pkts_evts_merged["captureTime"].values,
            tz="UTC").tz_convert(gc["tz"])

        pure_trades = pkts_evts_merged.groupby('eventId')["pureTrade"].sum()
        pure_trades = pure_trades.rename('totalPureTradePackets')
        pkts_evts_merged = pkts_evts_merged.join(
            pure_trades,on='eventId', how='left')

        pkts_evts_merged['recordType'] = np.nan

        pkts_evts_merged = pkts_evts_merged[
            ['captureTime', 'seqNum', 'tradeSummaries', 'eventId', 'numPackets',
             'startSeqNum', 'timeDifference', 'packetType',  "pureTrade",
             'totalPureTradePackets', 'recordType']]

        self.logger.info("Saving the raw packets and events DataFrame")

        self.state["raw_packets_events"] = pd.concat(
            (self.state["raw_packets_events"], pkts_evts_merged))

        trade_counts = pd.DataFrame(
            packets["tradeDetails"] + packets["tradeSummaries"],
            columns=["tradeCounts"])

        trade_counts = trade_counts[trade_counts["tradeCounts"] > 0]
        trade_counts = trade_counts.groupby("tradeCounts").size()
        trade_counts.columns = ["Frequency"]
        trade_counts_summary = pd.concat(
            (self.data["trade_packets"], trade_counts))

        self.logger.info("Saving the trade counts summary")

        trade_packets = trade_counts_summary.groupby(
            trade_counts_summary.index).sum()
        trade_packets = pd.DataFrame(trade_packets.ix[:, 0],
                                     index=trade_packets.index.values,
                                     columns=["totalUpdates"])

        print(trade_packets)

        self.data["trade_packets"] = trade_packets

        book_counts = pd.DataFrame(
            packets["bookUpdates"] + packets["impBookUpdates"],
            columns=["bookCounts"])
        book_counts = book_counts[book_counts["bookCounts"] > 0]
        book_counts = book_counts.groupby('bookCounts').size()
        book_counts.columns = "Frequency"
        book_counts_summary = pd.concat(
            (self.data["book_packets"], book_counts))

        self.logger.info("Saving the book counts summary")

        book_updates = book_counts_summary.groupby(
            book_counts_summary.index).sum()

        book_updates = pd.DataFrame(book_updates.ix[:, 0],
                                    columns=["totalUpdates"],
                                    index=book_updates.index.values)
        print(book_updates)

        self.data["book_packets"] = book_updates

        self.logger.info("Saving the packets frequency state")

        self.state["packets_frequency"] = pd.concat(
            (self.state["packets_frequency"],
             pd.DataFrame(packets.resample(
                 self.report_config["packets_resample_time"]).size())))

        self.logger.info("Saving the events frequency state")

        self.state["events_frequency"] = pd.concat(
            (self.state["events_frequency"],
             pd.DataFrame(events.resample(
                 self.report_config["events_resample_time"]).size())))

        del packets
        del events

        time_differences = self.__calculate_times__(
            self.state["raw_packets_events"])

        time_differences = time_differences[time_differences.timeDifference > 0]
        time_diff_grouped = time_differences.groupby('recordType')
        time_diff_summary = time_diff_grouped["timeDifference"].describe(
            percentiles=self.report_config["summary_percentiles"]).unstack(1)
        time_diff_summary.rename(columns={"50%": "median"})
        time_diff_summary.columns = \
            ["{}{}".format(x[:1].upper(), x[1:]) for x in
             time_diff_summary.columns]

        self.logger.info("Saving the time differences data")

        self.data["time_differences"] = time_diff_summary

        packets_rolling_count = to_offset(
            self.report_config["rolling_summary_time"]).nanos / to_offset(
            self.report_config["packets_resample_time"]).nanos

        events_rolling_count = to_offset(
            self.report_config["rolling_summary_time"]).nanos / to_offset(
            self.report_config["events_resample_time"]).nanos

        rolling_packets = self.state["packets_frequency"].rolling(
            packets_rolling_count).sum()
        rolling_events = self.state["events_frequency"].rolling(
            events_rolling_count).sum()

        percentiles = self.report_config["rolling_summary_percentiles"]

        packets_summary = rolling_packets.dropna().describe(
            percentiles=percentiles)
        events_summary = rolling_events.dropna().describe(
            percentiles=percentiles)

        rolling_summary = pd.DataFrame({"Packets": packets_summary.ix[:, 0],
                                        "Events": events_summary.ix[:, 0]},
                                       index=packets_summary.index)

        rolling_summary = rolling_summary.transpose()

        del rolling_summary["50%"]

        reg = re.compile("%")

        perc_columns = [x for x in rolling_summary.columns if reg.search(x)]

        rolling_summary = rolling_summary[np.append(["max"], perc_columns)]
        rolling_summary.columns = ["{}{}".format(x[:1].upper(), x[1:]) for x in
                                   rolling_summary.columns]

        self.logger.info("Saving the rolling frequency data")

        self.data["rolling_summary"] = rolling_summary

    def __calculate_times__(self, merged):
        self.logger.info(
            "Calculating time differences within individual events")

        self.logger.info("Calculating Trade to Trade time differences")

        trade_to_trade_time = merged[
            (merged["eventId"].diff(1) == 0)
            & (merged["packetType"].diff(1) == 0) & (merged["packetType"] == 1)]
        trade_to_trade_time = trade_to_trade_time.copy()
        trade_to_trade_time["recordType"] = 'W-Trade-Trade'

        self.logger.info("Calculating Trade to Book time differences")

        trade_to_book_time = merged[
            (merged["eventId"].diff(1) == 0) & (merged["packetType"] == 2)
            & (merged["packetType"].shift() == 1)]
        trade_to_book_time = trade_to_book_time.copy()
        trade_to_book_time["recordType"] = 'W-Trade-Book'

        self.logger.info("Calculating Multiple Trades to Book time differences")

        multi_trade_to_book_time = merged[
            (merged["packetType"] == 2) & (merged["eventId"].diff(1) == 0)
            & (merged["packetType"].shift() == 1)
            & (merged["eventId"].diff(2) == 0)
            & (merged["packetType"].shift(2) == 1)]
        multi_trade_to_book_time = multi_trade_to_book_time.copy()
        multi_trade_to_book_time["recordType"] = 'W-multiTrade-Book'

        self.logger.info("Calculating Single Trade to Book time differences")

        single_trade_to_book_time = merged[
            (merged["eventId"].diff(1) == 0) & (merged["packetType"] == 2)
            & (merged["packetType"].shift() == 1)
            & ((merged["packetType"].shift(2) != 1)
               | (merged["eventId"].diff(2) > 0))]
        single_trade_to_book_time = single_trade_to_book_time.copy()
        single_trade_to_book_time["recordType"] = 'W-singleTrade-Book'

        self.logger.info("Calculating Book to Book time differences")

        book_to_book_time = merged[
            (merged["eventId"].diff(1) == 0)
            & (merged["packetType"].diff(1) == 0) & (merged["packetType"] == 2)]
        book_to_book_time = book_to_book_time.copy()
        book_to_book_time["recordType"] = 'W-Book-Book'

        self.logger.info("Calculating Book to Implied Book time differences")

        book_to_implied_time = merged[
            (merged["eventId"].diff(1) == 0) & (merged["packetType"] == 4)
            & (merged["packetType"].shift() == 2)]
        book_to_implied_time = book_to_implied_time.copy()
        book_to_implied_time["recordType"] = 'W-Book-Imp'

        self.logger.info("Calculating Any Update to Stop time differences")

        any_to_stop_time = merged[
            (merged["eventId"].diff(1) == 0) & (merged["packetType"] > 8)
            & (merged["tradeSummaries"] > 0)]
        any_to_stop_time = any_to_stop_time.copy()
        any_to_stop_time["recordType"] = 'W-Any-Stop'

        self.logger.info("Calculating Non-stop to Stop time differences")

        non_stop_to_stop_time = merged[
            (merged["eventId"].diff(1) == 0) & (merged["packetType"] > 8)
            & (merged["tradeSummaries"] > 0)
            & ((merged["tradeSummaries"].shift() == 0)
               | (merged["eventId"].diff(2) == 1))]
        non_stop_to_stop_time = non_stop_to_stop_time.copy()
        non_stop_to_stop_time["recordType"] = 'W-nonStop-Stop'

        self.logger.info("Calculating time differences across events")

        self.logger.info(
            "Calculating Single Trade to Single Trade time differences")

        single_to_single_time = merged[
            (merged["eventId"].diff(1) == 1) & (merged["seqNum"].diff(1) == 1)
            & (merged["numPackets"] == 1) & (merged["numPackets"].shift() == 1)]
        single_to_single_time = single_to_single_time.copy()
        single_to_single_time["recordType"] = 'X-Single-Single'

        self.logger.info(
            "Calculating Single Trade to Multiple Trades time differences")

        single_to_multi_time = merged[
            (merged["eventId"].diff(1) == 1) & (merged["seqNum"].diff(1) == 1)
            & (merged["numPackets"] > 1) & (merged["numPackets"].shift() == 1)]
        single_to_multi_time = single_to_multi_time.copy()
        single_to_multi_time["recordType"] = 'X-Single-Multi'

        self.logger.info(
            "Calculating Multiple Trades to Single Trade time differences")

        multi_to_single_time = merged[
            (merged["eventId"].diff(1) == 1) & (merged["seqNum"].diff(1) == 1)
            & (merged["numPackets"] == 1) & (merged["numPackets"].shift() > 1)]
        multi_to_single_time = multi_to_single_time.copy()
        multi_to_single_time["recordType"] = 'X-Multi-Single'

        self.logger.info(
            "Calculating Multiple Trades to Multiple Trades time differences")

        multi_to_multi_time = merged[
            (merged["eventId"].diff(1) == 1) & (merged["seqNum"].diff(1) == 1)
            & (merged["numPackets"] > 1) & (merged["numPackets"].shift() > 1)]
        multi_to_multi_time = multi_to_multi_time.copy()
        multi_to_multi_time["recordType"] = 'X-Multi-Multi'

        self.logger.info(
            "Calculating Multiple Pure Trade to Multiple Pure Trade time "
            "differences")

        pure_multi_to_pure_multi_time = merged[
            (merged["eventId"].diff(1) == 1) & (merged["seqNum"].diff(1) == 1)
            & (merged["totalPureTradePackets"] > 1)
            & (merged["totalPureTradePackets"].shift() > 1)]
        pure_multi_to_pure_multi_time = pure_multi_to_pure_multi_time.copy()
        pure_multi_to_pure_multi_time["recordType"] = 'X-multiPT-multiPT'

        self.logger.info(
            "Calculating Multiple Pure Trade to Single Trade time differences")

        pure_multi_to_single_time = merged[
            (merged["eventId"].diff(1) == 1) & (merged["seqNum"].diff(1) == 1)
            & (merged["numPackets"] == 1)
            & (merged["totalPureTradePackets"].shift() > 1)]
        pure_multi_to_single_time = pure_multi_to_single_time.copy()
        pure_multi_to_single_time["recordType"] = 'X-multiPT-Single'

        self.logger.info(
            "Calculating Single Trade to Multiple Pure Trades time differences")

        single_to_pure_multi_time = merged[
            (merged["eventId"].diff(1) == 1) & (merged["seqNum"].diff(1) == 1)
            & (merged["totalPureTradePackets"] > 1)
            & (merged["numPackets"].shift() == 1)]
        single_to_pure_multi_time = single_to_pure_multi_time.copy()
        single_to_pure_multi_time["recordType"] = 'X-Single-multiPT'

        self.logger.info(
            "Calculating Multiple Pure Trade to Multiple Trade time "
            "differences")

        pure_multi_to_multi_time = merged[
            (merged["eventId"].diff(1) == 1) & (merged["seqNum"].diff(1) == 1)
            & (merged["numPackets"] > 1) &
            (merged["totalPureTradePackets"].shift() > 1)]
        pure_multi_to_multi_time = pure_multi_to_multi_time.copy()
        pure_multi_to_multi_time["recordType"] = 'X-multiPT-Multi'

        self.logger.info(
            "Calculating Multiple Trade to Multiple Pure Trade time "
            "differences")

        multi_to_pure_multi_time = merged[
            (merged["eventId"].diff(1) == 1) & (merged["seqNum"].diff(1) == 1)
            & (merged["totalPureTradePackets"] > 1)
            & (merged["numPackets"].shift() > 1)]
        multi_to_pure_multi_time = multi_to_pure_multi_time.copy()
        multi_to_pure_multi_time["recordType"] = 'X-Multi-multiPT'

        time_differences = pd.concat([trade_to_trade_time,
                                      trade_to_book_time,
                                      multi_trade_to_book_time,
                                      single_trade_to_book_time,
                                      book_to_book_time,
                                      book_to_implied_time,
                                      any_to_stop_time,
                                      non_stop_to_stop_time,
                                      single_to_single_time,
                                      single_to_multi_time,
                                      multi_to_single_time,
                                      multi_to_multi_time,
                                      pure_multi_to_pure_multi_time,
                                      pure_multi_to_single_time,
                                      single_to_pure_multi_time,
                                      pure_multi_to_multi_time,
                                      multi_to_pure_multi_time])

        time_differences = time_differences[['eventId', 'seqNum', 'startSeqNum',
                                             'numPackets', 'timeDifference',
                                             'recordType']]
        return time_differences


class CmePacketStatisticsReport(object):

    def __init__(self, config_file, start_date, end_date):
        config = ConfigLoader(config_file)
        self.report_config = config.report_config(REPORT_NAME)
        self.global_config = config.global_config()
        self.plot_config = config.plot_config()
        reader = CmePacketStatisticsReportDataReader(config_file)
        date_range = pd.bdate_range(start_date, end_date)
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        self.aggregated_time_differences = pd.DataFrame()
        self.aggregated_trade_packets = pd.DataFrame()
        self.aggregated_book_packets = pd.DataFrame()

        for trade_date in date_range:
            data = reader.read(trade_date=trade_date)
            td = data["time_differences"]
            tp = data["trade_packets"]
            bp = data["book_packets"]
            print(tp)
            print(bp)
            if len(td) > 0:
                td["TradeDate"] = trade_date
            if len(tp) > 0:
                tp["TradeDate"] = trade_date
            if len(bp) > 0:
                bp["TradeDate"] = trade_date
            self.aggregated_time_differences = pd.concat(
                (self.aggregated_time_differences, td))
            self.aggregated_trade_packets = pd.concat(
                (self.aggregated_trade_packets, tp))
            self.aggregated_book_packets = pd.concat(
                (self.aggregated_book_packets, bp))

        self.td_means = self.aggregated_time_differences.groupby(
            self.aggregated_time_differences.index).mean()
        self.tp_means = self.aggregated_trade_packets.groupby(
            self.aggregated_trade_packets.index).mean()
        self.bp_means = self.aggregated_book_packets.groupby(
            self.aggregated_book_packets.index).mean()

        self.rolling_summary = reader.read(trade_date=end_date)

    def display_time_differences(self, kind="today", latex=True):
        if latex:
            if kind is "today":
                self.__display_td_today_latex__()
            elif kind is "average":
                self.__display_td_avg_latex__()
        else:
            if kind is "today":
                self.__display_td_today_html__()
            elif kind is "average":
                self.__display_td_avg_html__()

    def __display_td_today_latex__(self):
        pass

    def __display_td_avg_latex__(self):
        pass

    def __display_td_today_html__(self):
        pass

    def __display_td_avg_html__(self):
        pass

    def display_frequencies(self, latex=True):
        if latex:
            self.__display_freq_latex__()
        else:
            self.__display_freq_html__()

    def __display_freq_latex__(self):
        pass

    def __display_freq_html__(self):
        pass

    def plot_frequency_averages(self, latex=True):
        if latex:
            self.__plot_freq_avg_latex__()
        else:
            self.__plot_freq_avg_html__()

    def __plot_freq_avg_latex__(self):
        pass

    def __plot_freq_avg_html__(self):
        pass

    def display_trade_packets(self, latex=True):
        if latex:
            self.__display_tp_latex__()
        else:
            self.__display_tp_html__()

    def __display_tp_latex__(self):
        pass

    def __display_tp_html__(self):
        pass

    def display_book_packets(self, latex=True):
        if latex:
            self.__display_bp_latex__()
        else:
            self.__display_bp_html__()

    def __display_bp_latex__(self):
        pass

    def __display_bp_html__(self):
        pass

    def plot_trade_packets(self, latex=True):
        if latex:
            self.__plot_tp_latex__()
        else:
            self.__plot_tp_html__()

    def __plot_tp_latex__(self):
        pass

    def __plot_tp_html__(self):
        pass

    def plot_book_packets(self, latex=True):
        if latex:
            self.__plot_bp_latex__()
        else:
            self.__plot_bp_html__()

    def __plot_bp_latex__(self):
        pass

    def __plot_bp_html__(self):
        pass



