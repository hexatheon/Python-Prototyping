import pandas as pd
import numpy as np

from alf_common.Constants.UpdateActions import \
    updateActions
from alf_common.Constants.Sides import sides
from proxent_utils.logger import BaseLogger


class TopExistsEnum:
    UNKNOWN = 0  # Only for the initial ambiguous updates
    EXISTS = 1  # Top Order Exists with some probability > 0
    DOES_NOT_EXIST = 2  # Top Order Does Not Exist


class AggressionState:
    LOCAL_MULTI = 1  # Aggression trade happens in local book with
    # multiple trade details
    IMP_MULTI = 2  # Aggression happens due to implied out trade with multiple
    # trade details
    LOCAL_REDUCE_TOP = 3  # Reduce the top quantity from the local book due to
    # an aggression in the local book
    IMP_REDUCE_TOP = 4  # Reduce the top quantity from the implied book due to
    # an aggression in the implied book
    NO_AGGRESSION = 0  # No aggression has happened. Top tracker is instantiated
    # with this state


class PrevBook:
    def __init__(self):
        self.cont = True  # true if there is no trade updates between books
        self.qty = None  # Quantity on the level
        self.numOrders = None  # Num of orders on the level


# Listens to tick and book updates
# Figures out the bid/ask tracker
# Processes the update through TopTracker api method update_digest

class ConsolidatedTopTracker:
    def __init__(self):
        self.bidTracker = TopTracker(sides['Bid'])
        self.askTracker = TopTracker(sides['Offer'])
        self.tradeActions = [updateActions['TradeSummary'],
                             updateActions['TradeDetail']]

    def process_tick_and_book(self, tick, book, timestamp):
        level = tick["level"]
        updateAction = tick["updateAction"]
        side = tick["side"]
        if updateAction == updateActions["TradeDetailAgg"] or \
           (updateAction not in self.tradeActions and level != 1):
            return []

        if updateAction in self.tradeActions:
            tradePrice = tick["price"]
            if tradePrice == self.askTracker.top_price:
                tracker = self.askTracker
            elif tradePrice == self.bidTracker.top_price:
                tracker = self.bidTracker
            else:
                return []
        else:
            tracker = self.bidTracker if side == sides['Bid'] else \
                self.askTracker

        return tracker.update_tick_and_book(
            updateAction, tick, book, timestamp)

    # Listens to ticks and book updates and processes them
    # argument info is the output of
    # inst.mergeLoadedBooksWithTrades(
    #     bookType="Outright", includeTicks=True).iterrows()
    def process_update(self, info):
        row = info[1]
        level = row.level

        # only level 1 and trade updates are relevant
        if level != 1 and level != 0:
            return []

        update_actions = row.updateAction
        side = row.side
        trade_price = row.tradePrice

        if update_actions == updateActions['TradeSummary'] \
                or update_actions == updateActions['TradeDetail']:
            if trade_price == self.askTracker.top_price:
                tracker = self.askTracker
            elif trade_price == self.bidTracker.top_price:
                tracker = self.bidTracker
            else:
                return []
        else:
            tracker = self.bidTracker if side == sides['Bid'] else \
                self.askTracker

        return tracker.update_digest(update_actions, info)

    def get_status(self, side):
        if side == sides['Bid']:
            return self.bidTracker.report_status()
        elif side == sides['Offer']:
            return self.askTracker.report_status()

    def get_simple_status_dict(self, side):
        if side == sides['Bid']:
            return self.bidTracker.report_simple_status_dict()
        elif side == sides['Offer']:
            return self.askTracker.report_simple_status_dict()


class TopTracker(BaseLogger):
    def __init__(self, side):
        self.top_exists = TopExistsEnum.UNKNOWN
        self.side = side
        self.previous_book = PrevBook()
        self._initialize_helper()

    def close_top(self, time):
        self.end_time = time
        self.top_exists = TopExistsEnum.DOES_NOT_EXIST
        rv = {
            "creation_time": self.create_time,
            "end_time": self.end_time,
            "side": self.side,
            "top_price": self.top_price,
            "initial_quantity": self.initial_quantity,
            "quantity": self.qty,
            "potentially_cancelled": self.potentially_cancelled,
            "revised": self.revised,
            "self_match_protection": np.nan
        }
        self._initialize_helper()
        return rv

    def _initialize_helper(self):
        self.create_time = None
        self.end_time = None
        self.potentially_cancelled = False
        self.initial_quantity = None
        self.qty = None
        self.top_price = None
        self.revised = False
        self.aggression_against_top = AggressionState.NO_AGGRESSION

    def report_status(self):
        """
        if top order is known to exist or Unknown
        report a tuple of the following items:
        1.time at which the top is created
        2.the side (bid/ask) which the top order belongs to
        3.the price at which the top order is created
        4.the initial qty of the top order
        5.the current estimated qty of the top order
        6.whether there have been cancellation of the exact remaining size
          of the top order.
        7.whether they have been order revision on the level (not necessarily
        the top order)

        """
        if self.top_exists == TopExistsEnum.DOES_NOT_EXIST:
            return None
        else:
            return {
                "creation_time": self.create_time,
                "side": self.side,
                "top_price": self.top_price,
                "initial_quantity": self.initial_quantity,
                "quantity": self.qty,
                "potentially_cancelled": self.potentially_cancelled,
                "revised": self.revised
            }

    def report_simple_status_dict(self):
        rv = {"status": self.top_exists}
        if self.top_exists == TopExistsEnum.DOES_NOT_EXIST:
            return rv
        else:
            rv["price"] = self.top_price
            rv["qty"] = self.qty
            rv["potentiallyCancelled"] = self.potentially_cancelled
            return rv

    def top_overshadow(self, time):
        if self.top_exists == TopExistsEnum.EXISTS:
            if self.qty > 0:
                self.qty = 0
            return self.close_top(time)

    def top_create(self, qty, price, time):
        self.create_time = time
        self.top_exists = TopExistsEnum.EXISTS
        self.qty = qty
        self.initial_quantity = qty
        self.top_price = price

    def price_changed_unexpectedly(self, price, time):
        if self.top_exists == TopExistsEnum.EXISTS and price != self.top_price:
            return True
        return False

    def check_cancels_and_revisions(self, currend_num_orders, current_quantity,
                                    previous_num_orders,
                                    previous_quantity, cont, time):
        ret = None
        if self.top_exists == TopExistsEnum.EXISTS:
            if currend_num_orders == previous_num_orders and cont:
                if currend_num_orders == 1:
                    if current_quantity - previous_quantity > 0:
                        self.qty = 0
                        ret = self.close_top(time)
                    else:
                        self.qty += current_quantity - previous_quantity
                self.revised = True
            elif (currend_num_orders == previous_num_orders - 1) and \
                            (previous_quantity - current_quantity) == self.qty:
                self.potentially_cancelled = True
        return ret

    def level_wipe_out(self, last_l1_qty, time):
        if self.top_exists == TopExistsEnum.EXISTS:
            diff = self.qty - last_l1_qty
            if diff == 0 or self.qty == 0:
                self.qty = 0
            else:
                # Note! here qty is set to the diff for easier debugging.
                # This may sometimes lead to unexpected negative values in
                # (only) the output table
                
                self.logger.warn("Level WIPE OUT: "
                                 "Difference between the current estimated"
                                 "top quantity and quantity on the level is {}"
                                 "".format(diff))
                self.qty = diff
            return self.close_top(time)

    def reduce_top(self, trade_quantity, time):
        if self.top_exists == TopExistsEnum.EXISTS:
            self.qty -= trade_quantity
            if self.qty <= 0:
                return self.close_top(time)

    def multi_end(self, trade_quantity, time):
        if self.top_exists == TopExistsEnum.EXISTS:
            diff = self.qty - trade_quantity
            if diff == 0:
                if self.potentially_cancelled == False:
                    self.qty = 0
            return self.close_top(time)

    def update_digest(self, update_action, info):
        # assumes info is the output of
        # inst.mergeLoadedBooksWithTrades(
        #     bookType="Outright", includeTicks=True).iterrows()

        """
        either return an empty list
        return a list of the following items:
        1. topCreationTime",
        2. topEndTime",
        3. side(bid/ask) to which the top order belongs
        4. price at which the top order is created
        5. initial Qty of the top order
        6. the final unaccounted qty of the top order
          (note that level_wipe_out returns the diff value)
        7. whether the top order has been potentially cancelled
        8. whether there have been order revision on the level
        (not necessarily the top)
        9. the possibility of self match protection (placeholder. Not developed)

        """
        time = info[0]
        row = info[1]
        size = row.qty
        price = row.price
        aggside = row.tradeAggSide
        trade_num_orders = row.tradeNumOrders
        tradeQty = row.tradeQty
        if self.side == sides['Bid']:
            qty = row.bidQty_1
            num_orders = row.bidOrders_1
        else:
            qty = row.askQty_1
            num_orders = row.askOrders_1

        return self._update(update_action, time, size, price, qty, num_orders,
                            aggside, trade_num_orders, tradeQty)

    def update_tick_and_book(self, update_action, tick, book, timestamp):
        if self.side == sides['Bid']:
            qty = book["bidQty_1"]
            num_orders = book["bidOrders_1"]
        else:
            qty = book["askQty_1"]
            num_orders = book["askOrders_1"]

        return self._update(update_action, timestamp,
                            tick["qty"], tick["price"], qty, num_orders,
                            tick["side"], tick["numOrders"], tick["qty"])

    def _update(self, update_action, time, size, price, qty, num_orders,
                aggside, trade_num_orders, trade_quantity):
        current_l1_qty = qty
        current_l1_orders = num_orders

        to_return = None

        if update_action == updateActions["New"]:
            to_return = self.top_overshadow(time)
            self.top_create(size, price, time)

        elif update_action == updateActions["Change"]:

            # Ideally this check_price_change should only return false.
            # All price changes should have already been accounted for by
            # other sub-routines
            if self.price_changed_unexpectedly(price, time):
                self.logger.warn("WARNING! price changed unexpectedly")
                return self.close_top(time)

            if self.previous_book.qty != None:
                last_l1_qty = self.previous_book.qty
                last_l1_orders = self.previous_book.numOrders
                to_return = self.check_cancels_and_revisions(current_l1_orders,
                                                    current_l1_qty,
                                                    last_l1_orders,
                                                    last_l1_qty,
                                                    self.previous_book.cont,
                                                    time)
                self.previous_book.cont = True

        elif update_action == updateActions["Delete"]:
            to_return = self.level_wipe_out(self.previous_book.qty, time)

        elif update_action == updateActions["TradeSummary"]:
            # Crucial: do not reduce the top here based merely on TradeSummary.
            # Implied-in screws everything

            if aggside == sides['Unknown']:  # Product was implied out
                if trade_num_orders == 1:
                    self.aggression_against_top = AggressionState.IMP_REDUCE_TOP
                elif trade_num_orders == 0:
                    print(" " * 60),
                    print("WARNING------------------------Butterfly here?")
                else:
                    self.aggression_against_top = AggressionState.IMP_MULTI

            else:
                if trade_num_orders > 2:
                    self.aggression_against_top = AggressionState.LOCAL_MULTI

                elif trade_num_orders == 2:
                    self.aggression_against_top = \
                        AggressionState.LOCAL_REDUCE_TOP
                else:

                    # Trade was implied-in trade. trade_num_orders is 1
                    pass

        elif update_action == updateActions["TradeDetail"] \
                and self.aggression_against_top != \
                        AggressionState.NO_AGGRESSION:

            if self.aggression_against_top == AggressionState.LOCAL_MULTI:
                to_return = self.multi_end(trade_quantity, time)

            elif self.aggression_against_top == \
                    AggressionState.LOCAL_REDUCE_TOP:
                to_return = self.reduce_top(trade_quantity, time)

            elif self.aggression_against_top == AggressionState.IMP_MULTI:
                to_return = self.multi_end(trade_quantity, time)

            elif self.aggression_against_top == AggressionState.IMP_REDUCE_TOP:
                to_return = self.reduce_top(trade_quantity, time)

            self.aggression_against_top = AggressionState.NO_AGGRESSION

        if pd.notnull(price): # means it is a book update
            self.previous_book.qty = current_l1_qty
            self.previous_book.numOrders = current_l1_orders
        else:
            self.previous_book.cont = False

        return to_return
