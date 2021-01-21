# SciFi Lorem

__What if I tell you, we can teach an AI to be a writer?__
> At first glance, this question sounds something straight out of a SciFi movie, but surprisingly it isn't.<br>
> This project is an approach to implement that, to put it another way, this project is an approach to "teach an AI how to write" one character at a time, using the state of the art __character level language models__ or sometimes also termed as __Char RNNs__.<br>

## Overview
> For training the model, the [SciFi Stories Text Corpus](https://www.kaggle.com/jannesklaas/scifi-stories-text-corpus) was used. The dataset consists of a large collection of Science Fiction stories with a total of around `150 Million` characters.<br>
> This model takes it as input and trains a Recurrent Neural Network that learns to predict the next character. The RNN can then be used to generate text, character by character, that will look like the original training data.<br>

- _Fun Fact: The generated text can be used as a [Lorem Ipsum](https://en.wikipedia.org/wiki/Lorem_ipsum) alternative._

## Setup
```bash
git clone "https://github.com/priyavrat-misra/scifi-lorem.git" && cd "scifi-lorem/"
pip install -r requirements.txt  # install dependencies
```

## Usage
```bash
python generate.py --help
```
```
usage: SciFi Lorem [-h] [-S] [-s ] [-p ] [-k ] [-v ] [-c ]

generate random *meaningful* sci-fi text

optional arguments:
  -h, --help            show this help message and exit
  -S , --save_path      saves the text to a file with given name
  -s [], --size []      generates given no of characters (default 512)
  -p [], --prime []     sets the starter/prime text (default "The")
  -k [], --topk []      randomly choose one from top-k probable chars
  -v [], --verbose []   prints the text to console
  -c [], --copyclip []  copies the text to clipboard
```

- Example
```bash
python generate.py -s=2048 -p="A long time ago in a galaxy far, far away..." -k=2 -S="./out.txt" -c -v
```
```
A long time ago in a galaxy far, far away... the story of the present story of a series of them and the 
stars are a serious structure of a computer to see them and what we have to do in the communication to 
their starting the star with the particular process. The secretary is a statement that will be able to 
contract them to the problem of a strange change. It was that the sun was a secret section and a chance to 
see that they were too strong. The second story would have the same thing and they would have to stop. It 
had the same strange chance. It was a pretty good sense of success and an answer. The streets were all 
about the same thing. The station had been a complex thing to see and then to be a series. To the surface 
of the concept of the station, the street was to be sure that the streets were too strange. The streets of 
the ship was to be seen at the same time. The second thing was a chance to see the street of the command 
and the ship was the same to him. He wasn't sure that his father was the only one who had been a computer 
and that he was still a computer to the planet that had been started and had been a state of strength and 
the star was almost as if the computer had been a chance. He had never seen the short stories. That way was 
a star on the station. He had to see him, the car showed that he would have to start a strange contraction. 
He was surprised at that at the time that he had been told the same and he was still a secretary of the 
ship which he had become. He was all too large, and his face was strong and hard. There was no such answer 
to the show of the ship. The stream was so strong that the state was a strong computer and the state of the 
ship. That was the way he had been a stranger than the star and the car was than the other story of an 
answer. That was all they wanted. The station was a strange star and the control room, the car was staring 
at the car as his shoulder was still still still and the statements of the contrast. He saw the commander 
that was all too late. He was a stranger, and his strength was the only word of the st

>>> generated text saved to ./out.txt
>>> generated text copied to clipboard
```

## Conclusions
- Even though the grammar appears to be messed up, the model did a good job learning the use of punctuation marks, spacing, upper case, lower case, etc correctly.
- And and and, it knows how to spell words!
