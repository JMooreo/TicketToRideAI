from typing import List

from src.actions.Action import Action
from src.actions.DrawDestinationsAction import DrawDestinationsAction
from src.actions.DrawRandomCardAction import DrawRandomCardAction
from src.actions.DrawVisibleCardAction import DrawVisibleCardAction
from src.actions.FinishSelectingDestinationsAction import FinishSelectingDestinationsAction
from src.actions.SelectDestinationAction import SelectDestinationAction


class InformationSet:
    @staticmethod
    # Hidden Information from the turn for opponents who wouldn't know the details of the actions performed.
    def for_opponents(turn_history: List[Action]) -> str:
        if len(turn_history) == 0:
            return ""

        # These compressions are specifically for the players who are NOT currently taking their turn

        # Destination Selection
        # Reduces 27,000 branches to 1
        if all([type(action) in [DrawDestinationsAction, SelectDestinationAction, FinishSelectingDestinationsAction]
                for action in turn_history]):
            return "hidden_dest_selection"

        # Drawing Cards
        # Reduces 81 branches to 36
        if all([type(action) in [DrawRandomCardAction, DrawVisibleCardAction] for action in turn_history]):
            sorted_actions = sorted(["draw_RANDOM" if isinstance(action, DrawRandomCardAction)
                                     else str(action) for action in turn_history])
            if len(sorted_actions) > 0:
                return "_AND_".join(sorted_actions)

        return " ".join(str(action) for action in turn_history)

    @staticmethod
    # Detailed Information from the turn, to compress the game tree.
    def for_current_player(turn_history: List[Action]):
        # Destination Selection
        # Reduces 27,000 branches to 4060
        if all([type(action) in [DrawDestinationsAction, SelectDestinationAction, FinishSelectingDestinationsAction]
                for action in turn_history]):
            info_set = "draw_dest"
            destinations = sorted([str(action.destination) for action in turn_history
                                   if isinstance(action, SelectDestinationAction)])
            if len(destinations) > 0:
                info_set += "_AND_select_"
                info_set += "_AND_".join(destinations)

            return info_set

        # Drawing Cards
        # Reduces 81 branches to 36.
        # Makes branches like draw_WILD_AND_draw_WILD possible.
        if all([type(action) in [DrawRandomCardAction, DrawVisibleCardAction] for action in turn_history]):
            sorted_actions = sorted([str(action) for action in turn_history])
            if len(sorted_actions) > 0:
                return "_AND_".join(sorted_actions)

        return str(turn_history[0])

    # TODO: "Current Player" Game Tree Branch Compression

    # There are 495 ways to start with 4 cards, and there are no decisions to be made.
    # We have to draw 4 cards at the start of the game.
    # All we care about is which combination of cards we start with.
    # It should make branches at nodes more consistent and accurate.

    # PAYMENT OPTIONS -- Reduction Factor: 132 once implemented.
        # TODO: let the bot decide what it wants to pay with, requires a new game state and an action space update.
        # Instead of deciding which card to pay with every step, just choose one of the valid options.
        # At most, 7 options
        # Best case, it reduces 924 options to 7
        # Worst case, it increases the branches from 1 to 2
