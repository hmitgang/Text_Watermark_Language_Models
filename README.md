### Harrison's Updates

In `reverse_watermark.py`, we extend the watermarking class. In `secret_watermark`, we add a secret to the hashing function. As seen in the Output, this adds confidentiality to the watermark. In other words, you need the secret to determine whether the text has been watermarked.

The `clear_watermark` is my attempt to remove the watermark from text. Notice that this class doesn't use the `binary_encoding_function` at all, and thus doesn't need to know the secret to remove the watermark. This class essentially does the same as the original model, but does not prioritize any specific binary encoding and thus more evenly separates the 1s and 0s.

The above modification, along with the attack, shows that the secret_watermark is able to provide _confidentiality_ but not _integrity_.


# :sweat_drops: [Watermarking Text Generated by Black-Box Language Models](https://arxiv.org/abs/2305.08883)

Official implementation of the watermark injection and detection algorithms presented in the [paper](https://arxiv.org/abs/2305.08883):

"Watermarking Text Generated by Black-Box Language Models" by _Xi Yang, Kejiang Chen, Weiming Zhang, Chang Liu, Yuang Qi, Jie Zhang, Han Fang, and Nenghai Yu_.  

## Requirements
- Python 3.9
- check requirements.txt
```sh
pip install -r requirements.txt
pip install git+https://github.com/JunnYu/WoBERT_pytorch.git  # Chinese word-level BERT model
python -m spacy download en_core_web_sm
```
- For Chinese, please download the [pre-trained Chinese word vectors](https://drive.google.com/file/d/1Zh9ZCEu8_eSQ-qkYVQufQDNKPC4mtEKR/view) and place it in the root directory.

## Repo contents

The watermark injection and detection modules are located in the `models` directory. `watermark_original.py` implements the iterative algorithms as described in the paper. `watermark_faster.py` introduces batch processing to speed up the watermark injection algorithm and the precise detection algorithm.

We provide two demonstrations, `demo_CLI.py` and `demo_gradio.py`, which correspond to command-line interaction and graphical interface interaction respectively.

## Demo Usage
> Click on the GIFs to enlarge them for a better experience.
### Graphical User Interface
```sh
$ python demo_gradio.py --language English --tau_word 0.8 --lamda 0.83
```
<p align="center">
  <img src="images/en_gradio.gif" />
</p>

```sh
$ python demo_gradio.py --language Chinese --tau_word 0.75 --lamda 0.83
```
<p align="center">
  <img src="images/cn_gradio.gif" />
</p>

### Command Line Interface
```sh
$ python demo_CLI.py --language English --tau_word 0.8 --lamda 0.83
```
<p align="center">
  <img src="images/eng_cli.gif" />
</p>

```sh
$ python demo_CLI.py --language Chinese --tau_word 0.75 --lamda 0.83
```

<p align="center">
  <img src="images/cn_cli.gif" />
</p>


