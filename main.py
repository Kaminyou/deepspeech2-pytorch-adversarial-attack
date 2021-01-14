import sys, os
sys.path.append("../deepspeech.pytorch/")
from deepspeech_pytorch.configs.inference_config import TranscribeConfig
from deepspeech_pytorch.utils import load_decoder, load_model
from attack import Attacker
import torchaudio
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # I/O parameters
    parser.add_argument('--input_wav', type=str, help='input wav. file')
    parser.add_argument('--output_wav', type=str, default='None', help='output adversarial wav. file')
    parser.add_argument('--model_path', type=str, default='/home/user/folder/deepspeech.pytorch/models/librispeech/librispeech_pretrained_v2.pth', help='model pth path; please use absolute path')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    
    # attack parameters
    parser.add_argument('--target_sentence', type=str, default="HELLO WORLD", help='Please use uppercase')
    parser.add_argument('--mode', type=str, default="PGD", help='PGD or FGSM')
    parser.add_argument('--epsilon', type=float, default=0.25, help='epsilon')
    parser.add_argument('--alpha', type=float, default=1e-3, help='alpha')
    parser.add_argument('--PGD_iter', type=int, default=50, help='PGD iteration times')

    # plot parameters
    parser.add_argument('--plot_ori_spec', type=str, default="None", help='Path to save the original spectrogram')
    parser.add_argument('--plot_adv_spec', type=str, default="None", help='Path to save the adversarial spectrogram')
    args = parser.parse_args()

    cfg = TranscribeConfig
    model = load_model(device="cpu", model_path=args.model_path, use_half=False)
    decoder = load_decoder(labels=model.labels, cfg=cfg.lm)

    sound, sample_rate = torchaudio.load(args.input_wav)
    target_sentence = args.target_sentence.upper()
    if args.output_wav == "None":
        output_wav = None
    attacker = Attacker(model=model, sound=sound, target=target_sentence, decoder=decoder, device=args.device, save=output_wav)

    attacker.attack(epsilon = args.epsilon, alpha=args.alpha, attack_type=args.mode, PGD_round=args.PGD_iter)

    if args.plot_ori_spec != "None":
        attacker.get_ori_spec(args.plot_ori_spec)
    
    if args.plot_adv_spec != "None":
        attacker.get_adv_spec(args.plot_adv_spec )