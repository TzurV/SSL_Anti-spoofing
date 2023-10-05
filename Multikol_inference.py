import logging
import numpy as np
import torch
from torch import Tensor
from torch import nn
from model import Model


logger = logging.getLogger()

class multikol_service:
    def __init__(self, settings):
        
        #GPU device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
        logger.info(f"Device: {self.device}")
        
        self.model = Model(None, self.device)
        nb_params = sum([param.view(-1).size()[0] for param in self.model.parameters()])
        self.model = nn.DataParallel(self.model).to(self.device)
        logger.info(f'loaded model size: {nb_params}')

        # limits audio duration
        self.cut = 64600

        # classification threshold
        self.threshold = settings['threshold']

        # load model
        self.model.load_state_dict(torch.load(settings['model_path'], map_location=self.device))
        self.model.eval()

    def pad(self, x):
        x_len = x.shape[0]
        if x_len >= self.cut:
            return x[:self.cut]
        # need to pad
        num_repeats = int(self.cut / x_len)+1
        padded_x = np.tile(x, (1, num_repeats))[:, :self.cut][0]
        return padded_x	
    
    def inference(self, audio_np):
        if type(audio_np) is not np.ndarray:
            raise Exception("type(audio_np) is not np.ndarray")
        
        X_pad = self.pad(audio_np)
        x_inp = Tensor(X_pad)
        x_inp = torch.unsqueeze(x_inp, dim=0)
        x_inp = torch.unsqueeze(x_inp, dim=2)

        batch_x = x_inp.to(self.device)
        batch_out = self.model(batch_x)
        
        score = (batch_out[:, 1]).data.cpu().numpy().ravel() 
        if score < self.threshold:
            hepothesis = "spoof"
        else:
            hepothesis = "bonafide"

        results = {"score": score[0], 
                   "hepothesis": hepothesis,
                   "original_length": audio_np.shape[0],
                   "padded_length": X_pad.shape[0]}
        logger.info(f"results: {results}")
        
        return results