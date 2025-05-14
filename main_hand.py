import argparse
import signal
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils import load_csv_np


class HandDistributionPlot:
    def __init__(self, input):
        self.data = load_csv_np(input)

    def compute(self):
        header = self.data[0]
        rows = self.data[1:]

        # Find indices
        house_idx = header.tolist().index("Hogwarts House")
        hand_idx = header.tolist().index("Best Hand")

        # Extract relevant columns
        houses = rows[:, house_idx]
        hands = rows[:, hand_idx]

        # Clean data: remove missing or invalid entries
        valid_mask = (hands != '') & (houses != '')
        hands = hands[valid_mask]
        houses = houses[valid_mask]

        # Count occurrences
        df = pd.DataFrame({'House': houses, 'Hand': hands})
        counts = df.groupby(['House', 'Hand']).size().unstack(fill_value=0)

        # Create bar chart
        fig = go.Figure()

        for hand in ['Left', 'Right']:
            if hand in counts.columns:
                fig.add_trace(go.Bar(
                    x=counts.index,
                    y=counts[hand],
                    name=hand
                ))

        fig.update_layout(
            title="Distribution of Dominant Hand by House",
            xaxis_title="Hogwarts House",
            yaxis_title="Number of Students",
            barmode='group'
        )

        fig.show()


def optparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        '-i',
        action="store",
        dest="input",
        default="resources/dataset_train.csv",
        help="Set the input path file"
    )
    return parser.parse_args()


def signal_handler(sig, frame):
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    args = optparse()
    HandDistributionPlot(args.input).compute()
