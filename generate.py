import torch
import argparse
import pyperclip
from network import CharRNN
from infer_utils import sample


parser = argparse.ArgumentParser(
    prog='Scifi Lorem',
    description='''
    generate random sci-fi text,
    at times they sound meaningful
    (maybe the AI wants to be a writer?)
    '''
)
parser.add_argument('-s', '--size', type=int, metavar='', required=True,
                    help='generates given no of chars;')
parser.add_argument('-p', '--prime', type=str, metavar='', required=True,
                    help='sets starter chars; (e.g., "The")')
parser.add_argument('-S', '--save_path', type=str, metavar='',
                    help='saves the chars to a file with given name;')
parser.add_argument('-P', '--print', type=bool, metavar='',
                    help='if "True", prints the text to console;')
parser.add_argument('-c', '--copy_to_clip', type=bool, metavar='',
                    help='if "True", copies the text to clipboard;')


def main():
    args = parser.parse_args()
    with open('models/char_rnn.pth', 'rb') as f:
        checkpoint = torch.load(f, map_location='cpu')

    model = CharRNN(
        tokens=checkpoint['tokens'],
        n_hidden=checkpoint['n_hidden'],
        n_layers=checkpoint['n_layers']
    )
    model.load_state_dict(checkpoint['model'])
    model.eval()

    text = sample(model, size=args.size, top_k=3, prime=args.prime)

    if args.print:
        print(text)

    if args.save_path is not None:
        with open(args.save_path, 'w') as f:
            f.write(text)

        print(f'>>> generated text saved to {args.save_path}')

    if args.copy_to_clip:
        pyperclip.copy(text)
        print('>>> generated text copied to clipboard')


if __name__ == '__main__':
    main()
