from argparse import ArgumentParser

class TrainOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()
		
    def initialize(self):
        self.parser.add_argument('--domains', type=list, default=['Clipart','Product'], help='Path to experiment output directory') 	
        self.parser.add_argument('--img_size', default=256, type=int, help='Output size of generator')
        self.parser.add_argument('--batch', default=128, type=int, help='Number of Max iteration')
        self.parser.add_argument('--iter', default=10000, type=int, help='Number of Max iteration')
        self.parser.add_argument('--MEMO', type=str, default=None, help='Path to experiment output directory') 	

    def parse(self):
        opts = self.parser.parse_args()

        if opts.MEMO is None:
            opts.MEMO = f'Equal Encoder & Differ Decoder checking target Images reconstruction'  

        return opts