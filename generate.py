import torch
import argparse
import pyperclip
from network import CharRNN
from infer_utils import sample


parser = argparse.ArgumentParser(
    prog='SciFi Lorem',
    description='generate random *meaningful* sci-fi text'
)
parser.add_argument('-S', '--save_path', type=str, metavar='',
                    help='saves the text to a file with given name')
parser.add_argument('-s', '--size',
                    nargs='?', default=512, type=int, metavar='',
                    help='generates given no of characters (default 512)')
parser.add_argument('-p', '--prime',
                    nargs='?', default='The', type=str, metavar='',
                    help='sets the starter/prime text (default "The")')
parser.add_argument('-k', '--topk',
                    nargs='?', default=3, type=int, metavar='',
                    help='randomly choose one from top-k probable chars')
parser.add_argument('-v', '--verbose', nargs='?', const=1, type=bool,
                    metavar='', help='prints the text to console')
parser.add_argument('-c', '--copyclip', nargs='?', const=1, type=bool,
                    metavar='', help='copies the text to clipboard')


def main():
    args = parser.parse_args()
    if args.topk <= 1:
        print('-k/--topk should be a value greater than 1')
        exit(1)

    with open('models/char_rnn.pth', 'rb') as f:
        checkpoint = torch.load(f, map_location='cpu')

    model = CharRNN(
        tokens=checkpoint['tokens'],
        n_hidden=checkpoint['n_hidden'],
        n_layers=checkpoint['n_layers']
    )
    model.load_state_dict(checkpoint['model'])
    model.eval()

    text = sample(model, size=args.size, top_k=args.topk, prime=args.prime)

    if args.verbose:
        print(text + '\n')

    if args.save_path is not None:
        with open(args.save_path, 'w') as f:
            f.write(text)

        print(f'>>> generated text saved to {args.save_path}')

    if args.copyclip:
        pyperclip.copy(text)
        print('>>> generated text copied to clipboard')


if __name__ == '__main__':
    main()
