import os
import torch
from train import TrainingApp

def get_bleu_score():
    app = TrainingApp()
    # app.cbs = app.cbs[3:5]
    _last_pth = 'runs/unofficial_single_gpu_run/model_ckpt_best.pt'
    assert os.path.exists(_last_pth)
    app.model.load_state_dict(torch.load(_last_pth))
    app.learner('before_fit')
    app.learner.one_epoch(is_train=False)



if __name__ == '__main__':
    get_bleu_score()
