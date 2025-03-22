import numpy as np
from collections import namedtuple
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# controls whether we concatenate (z, c, h), etc for features used for car.
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3 # extra hidden later
MODE_ZH = 4


class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        self.args = args
        self.input_size = 64
        self.hidden_size = 64
        rnn_num_mixture = 10
        z_size = 10
        self.rnn_size = 10
        rnn_r_pred, rnn_d_pred = 0, 0
        self.output_size = rnn_num_mixture * z_size * 3 + rnn_r_pred + rnn_d_pred

        self.rnn = nn.RNN(input_size=self.input_size , hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(self.rnn_size, self.output_size)
    
    def forward(self, x, hidden):
        x, hidden = self.rnn(x, hidden)
        x = self.fc(x)
        return x, hidden
    
    def init_state(self):
        return torch.zeros(1, self.rnn_size)


# @tf.function
# def rnn_sim(rnn, z, states, a, training=True): 
#   z = tf.reshape(tf.cast(z, dtype=tf.float32), (1, 1, rnn.args.z_size))
#   a = tf.reshape(tf.cast(a, dtype=tf.float32), (1, 1, rnn.args.a_width))
#   input_x = tf.concat((z, a), axis=2)
#   rnn_out, h, c = rnn.inference_base(input_x, initial_state=states, training=training) # set training True to use Dropout
#   rnn_state = [h, c]
#   rnn_out = tf.reshape(rnn_out, [-1, rnn.args.rnn_size])
#   out = rnn.out_net(rnn_out)
#   mdnrnn_params, r, d_logits = rnn.parse_rnn_out(out)
#   mdnrnn_params = tf.reshape(mdnrnn_params, [-1, 3*rnn.args.rnn_num_mixture])
#   mu, logstd, logpi = tf.split(mdnrnn_params, num_or_size_splits=3, axis=1)

#   logpi = logpi / rnn.args.rnn_temperature # temperature
#   logpi = logpi - tf.reduce_logsumexp(input_tensor=logpi, axis=1, keepdims=True) # normalize

#   d_dist = tfd.Binomial(total_count=1, logits=d_logits)
#   d = tf.squeeze(d_dist.sample()) == 1.0
#   cat = tfd.Categorical(logits=logpi)
#   component_splits = [1] * rnn.args.rnn_num_mixture
#   mus = tf.split(mu, num_or_size_splits=component_splits, axis=1)

#   # temperature
#   sigs = tf.split(tf.exp(logstd) * tf.sqrt(rnn.args.rnn_temperature), component_splits, axis=1) 

#   coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale in zip(mus, sigs)]
#   mixture = tfd.Mixture(cat=cat, components=coll)
#   z = tf.reshape(mixture.sample(), shape=(-1, rnn.args.z_size))
  
#   if rnn.args.rnn_r_pred == 0:
#     r = 1.0 # For Doom Reward is always 1.0 if the agent is alive
#   return rnn_state, z, r, d
