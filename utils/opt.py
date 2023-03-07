import argparse
from pprint import pprint
import datetime
import os


# handles command line arguments

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.arg = None

    def _initial(self):
        curr_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # General options
        self.parser.add_argument('--datetime', type=str, default=curr_time, help='datatime now')
        self.parser.add_argument('--raw_data_path', type=str, default='data/EC3D/EC3D.pickle')
        self.parser.add_argument('--processed_path', type=str, default='data/EC3D/tmp_wo_val.pickle')
        self.parser.add_argument('--record', action='store_true', help='Record with Tensorboard, use --record to enable')
        self.parser.add_argument('--note', type=str, default='', help='A note about the run')

        # Model and Running Options
        self.parser.add_argument('--epoch', type=int, default=50, help='Number of epochs')
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
        self.parser.add_argument('--hidden', type=int, default=256, help='Number of hidden features')
        self.parser.add_argument('--p_dropout', type=float, default=0.5, help='Dropout probability, 1 for none')
        self.parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
        self.parser.add_argument('--lr_decay', type=int, default=5, help='every lr_decay epoch do lr decay')
        self.parser.add_argument('--gamma', type=float, default=0.2)
        self.parser.add_argument('--step_size', type=int, default=10, help='learning rate for the lr_scheduler')

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.arg), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.arg = self.parser.parse_args()
        self._print()
        return self.arg


if __name__ == '__main__':

    arg = Options().parse()
