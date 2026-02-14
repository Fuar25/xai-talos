"""Simple data fitting example using TorchTrainer."""
import numpy as np
import talos as ta

# Data
N, D_in, D_out = 100, 10, 1
X, Y = np.random.rand(N, D_in), np.random.rand(N, D_out)
train_set, test_set = ta.Dataset(X, Y, 'Dummy').split(1, 1)
train_set.report(), test_set.report()

# Model
model = ta.model.torch_zoo.MLP(D_in, [32, 32], D_out)
model.summary(D_in)

# Optimization
trainer = ta.TorchTrainer(model, loss_fn='mse')
trainer.config.print_every = 20
trainer.config.validate_every = 100
trainer.train(train_set, max_iterations=500)
