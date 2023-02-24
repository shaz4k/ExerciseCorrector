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
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # General options
        self.parser.add_argument('--datetime', type=str, default=curr_time, help='datatime now')
        self.parser.add_argument('--raw_data_path', type=str, default='data/EC3D/EC3D.pickle')
        self.parser.add_argument('--processed_path', type=str, default='data/EC3D/tmp_wo_val.pickle')

        # Model and Running Options

        # Model and Running Options
        # parser.add_argument('--device', type=str, default=device, help='Device used for model training')
        self.parser.add_argument('--epoch', type=int, default=50, help='Number of epochs')
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
        self.parser.add_argument('--hidden', type=int, default=256, help='Number of hidden features')
        self.parser.add_argument('--p_dropout', type=float, default=0.5, help='Dropout probability, 1 for none')
        self.parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')

    def parse(self):
        self._initial()
        self.arg = self.parser.parse_args()

        return self.arg


if __name__ == '__main__':
    arg = Options().parse()
