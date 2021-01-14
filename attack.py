from stft import STFT, magphase
import torch.nn as nn
import torch
import Levenshtein
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

def target_sentence_to_label(sentence, labels="_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "):
    out = []
    for word in sentence:
        out.append(labels.index(word))
    return torch.IntTensor(out)

def torch_spectrogram(sound, torch_stft):
    real, imag = torch_stft(sound)
    mag, cos, sin = magphase(real, imag)
    mag = torch.log1p(mag)
    mean = mag.mean()
    std = mag.std()
    mag = mag - mean
    mag = mag / std
    mag = mag.permute(0,1,3,2)
    return mag


class Attacker:
    def __init__(self, model, sound, target, decoder, device="cpu", save=None):
        """
        model: deepspeech model
        sound: raw sound data [-1 to +1] (read from torchaudio.load)
        label: string
        """
        self.sound = sound
        self.target_string = target
        self.target = target
        self.__init_target()
        
        self.model = model
        self.model.to(device)
        self.model.train()
        self.decoder = decoder
        self.criterion = nn.CTCLoss()
        self.device = device
        self.torch_stft = STFT(n_fft=320 , hop_length=160, win_length=320 ,  window='hamming', center=True, pad_mode='reflect', freeze_parameters=True, device=self.device)
        self.save = save
    
    def get_ori_spec(self, save=None):
        spec = torch_spectrogram(self.sound.to(self.device), self.torch_stft)
        plt.imshow(spec.cpu().numpy()[0][0])
        if save:
            plt.savefig(save)
            plt.clf()
        else:
            plt.show()

    def get_adv_spec(self, save=None):
        spec = torch_spectrogram(self.perturbed_data.to(self.device), self.torch_stft)
        plt.imshow(spec.cpu().numpy()[0][0])
        if save:
            plt.savefig(save)
            plt.clf()
        else:
            plt.show()
    
    # prepare
    def __init_target(self):
        self.target = target_sentence_to_label(self.target)
        self.target = self.target.view(1,-1)
        self.target_lengths = torch.IntTensor([self.target.shape[1]]).view(1,-1)

    # FGSM
    def fgsm_attack(self, sound, epsilon, data_grad):
        
        # find direction of gradient
        sign_data_grad = data_grad.sign()
        
        # add noise "epilon * direction" to the ori sound
        perturbed_sound = sound - epsilon * sign_data_grad
        
        return perturbed_sound
    
    # PGD
    def pgd_attack(self, sound, ori_sound, eps, alpha, data_grad) :
        
        adv_sound = sound - alpha * data_grad.sign() # + -> - !!!
        eta = torch.clamp(adv_sound - ori_sound.data, min=-eps, max=eps)
        sound = ori_sound + eta

        return sound
    
    def attack(self, epsilon, alpha, attack_type = "FGSM", PGD_round=40):
        print("Start attack")
        
        data, target = self.sound.to(self.device), self.target.to(self.device)
        data_raw = data.clone().detach()
        
        # initial prediction
        spec = torch_spectrogram(data, self.torch_stft)
        input_sizes = torch.IntTensor([spec.size(3)]).int()
        out, output_sizes = self.model(spec, input_sizes)
        decoded_output, decoded_offsets = self.decoder.decode(out, output_sizes)
        original_output = decoded_output[0][0]
        print(f"Original prediction: {decoded_output[0][0]}")
        
        # ATTACK
        ############ ATTACK GENERATION ##############
        if attack_type == "FGSM":
            data.requires_grad = True
            
            spec = torch_spectrogram(data, self.torch_stft)
            input_sizes = torch.IntTensor([spec.size(3)]).int()
            out, output_sizes = self.model(spec, input_sizes)
            out = out.transpose(0, 1)  # TxNxH
            out = out.log_softmax(2)
            loss = self.criterion(out, self.target, output_sizes, self.target_lengths)
            
            self.model.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            perturbed_data = self.fgsm_attack(data, epsilon, data_grad)

        elif attack_type == "PGD":
            for i in range(PGD_round):
                print(f"PGD processing ...  {i+1} / {PGD_round}", end="\r")
                data.requires_grad = True
                
                spec = torch_spectrogram(data, self.torch_stft)
                input_sizes = torch.IntTensor([spec.size(3)]).int()
                out, output_sizes = self.model(spec, input_sizes)
                out = out.transpose(0, 1)  # TxNxH
                out = out.log_softmax(2)
                loss = self.criterion(out, self.target, output_sizes, self.target_lengths)
                
                self.model.zero_grad()
                loss.backward()
                data_grad = data.grad.data

                data = self.pgd_attack(data, data_raw, epsilon, alpha, data_grad).detach_()
            perturbed_data = data
        ############ ATTACK GENERATION ##############

        # prediction of adversarial sound
        spec = torch_spectrogram(perturbed_data, self.torch_stft)
        input_sizes = torch.IntTensor([spec.size(3)]).int()
        out, output_sizes = self.model(spec, input_sizes)
        decoded_output, decoded_offsets = self.decoder.decode(out, output_sizes)
        final_output = decoded_output[0][0]
        
        abs_ori = 20*np.log10(np.sqrt(np.mean(np.absolute(data_raw.cpu().numpy())**2)))
        abs_after = 20*np.log10(np.sqrt(np.mean(np.absolute(perturbed_data.cpu().numpy())**2)))
        db_difference = abs_after-abs_ori
        l_distance = Levenshtein.distance(self.target_string, final_output)
        print(f"Max Decibel Difference: {db_difference:.4f}")
        print(f"Adversarial prediction: {decoded_output[0][0]}")
        print(f"Levenshtein Distance {l_distance}")
        if self.save:
            torchaudio.save(self.save, src=perturbed_data.cpu(), sample_rate=16000)
        self.perturbed_data = perturbed_data
        return db_difference, l_distance, self.target_string, final_output