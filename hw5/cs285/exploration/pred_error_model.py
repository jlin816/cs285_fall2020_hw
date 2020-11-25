from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch


class PredErrorModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        self.f_hat = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ob_dim,
                n_layers=self.n_layers,
                size=self.size)
        
        self.optimizer = self.optimizer_spec.constructor(
            self.f_hat.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )

        self.f_hat.to(ptu.device)

    def forward(self, ob_no, next_ob_no):
        """Gets the prediction error for ob_no
        Args:
            ob_no: shape (batch_size, self.ob_dim)
        Returns:
            error: shape (batch_size,)
        """
        # HINT: Remember to detach the output of self.f!
        # detach so that we don't update the random network
        error = torch.norm(self.f_hat(ob_no) - next_ob_no, dim=1)
        return error

    def forward_np(self, ob_no, next_ob_no):
        ob_no = ptu.from_numpy(ob_no)
        next_ob_no = ptu.from_numpy(next_ob_no)
        error = self(ob_no, next_ob_no).detach()
        return ptu.to_numpy(error)

    def update(self, ob_no, next_ob_no):
        # Update f_hat using ob_no
        # Hint: Take the mean prediction error across the batch
        ob_no = ptu.from_numpy(ob_no)
        next_ob_no = ptu.from_numpy(next_ob_no)
        loss = torch.mean(self.forward(ob_no, next_ob_no))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
