# deepspeech2-pytorch-adversarial-attack
PGD and FGSM algorithms are implemented to attack deepspeech2 model

## Get Start
Several dependencies required to be installed first. Please follow the instruction in [DeepSpeech 2 PyTorch](https://github.com/SeanNaren/deepspeech.pytorch) to build up the environments.</br>
It is recommended to setup your folders of [DeepSpeech 2 PyTorch](https://github.com/SeanNaren/deepspeech.pytorch) in the following structure.
```
ROOT_FOLDER/
├── this_repo/
├   ├──main.py
|   └──...
├──deepspeech.pytorch/
│   ├──models/
│   │   └──librispeech/
│   │       └──librispeech_pretrained_v2.pth
│   └──...
```
Then, you should download the DeepSpeech pretrained model from this [link](https://github.com/SeanNaren/deepspeech.pytorch/releases) provided by the [DeepSpeech 2 PyTorch](https://github.com/SeanNaren/deepspeech.pytorch)

## Introduction
Deep Speech 2 <sup>[1]</sup> is a modern ASR system, which enables end-to-end training as spectrogram is directly utilized to generate predicted sentence. In this work, PGD (Projected gradient descent) and FGSM (Fast Gradient Sign Method) algorithms are implemented to conduct adversarial attack against this ASR system.
1. Amodei, D., Ananthanarayanan, S., Anubhai, R., Bai, J., Battenberg, E., Case, C., ... & Zhu, Z. (2016, June). Deep speech 2: End-to-end speech recognition in english and mandarin. In International conference on machine learning (pp. 173-182).

## Usage
It is easy to perturb the original raw wave file to generate desired sentence with `main.py`.
```script
python3 main.py --input_wav your_wav.wav --output_wav to_save.wav --target_sentence HELLO_WORD
```
Actually, several parameters are available to make your adversarial attack better. `PGD` and `FGSM` modes are both provided with `epsilon`, `alpha`, and `PGD_iter` to adjusted for better results. For the details, please refer to `main.py`.

## Reference
The pytorch version STFT algorithm is from [this repo](https://github.com/pseeth/torch-stft).