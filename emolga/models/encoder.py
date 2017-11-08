import theano
import logging
import copy
import math

from emolga.layers.core import Dropout, Dense, Dense2, Identity
from emolga.layers.recurrent import *
from emolga.layers.embeddings import *
from emolga.layers.attention import *
from emolga.models.core import Model
from nltk.stem.porter import *

logger = logging.getLogger(__name__)
RNN    = GRU             # change it here for other RNN models.
err    = 1e-9

class Encoder(Model):
    """
    Recurrent Neural Network-based Encoder
    It is used to compute the context vector.
    """

    def __init__(self,
                 config, rng, prefix='enc',
                 mode='Evaluation',  # it's RNN
                 embed=None,
                 use_context=False):
        super(Encoder, self).__init__()
        self.config = config
        self.rng = rng
        self.prefix = prefix
        self.mode = mode
        self.name = prefix
        self.use_context = use_context

        self.return_embed = False
        self.return_sequence = False

        """
        Create all elements of the Encoder's Computational graph
        """
        # create Embedding layers
        logger.info("{}_create embedding layers.".format(self.prefix))
        if embed:
            self.Embed = embed
        else:
            self.Embed = Embedding(
                self.config['enc_voc_size'], # 50000
                self.config['enc_embedd_dim'], # 150
                name="{}_embed".format(self.prefix))
            self._add(self.Embed)

        if self.use_context:
            self.Initializer = Dense(
                config['enc_contxt_dim'],
                config['enc_hidden_dim'],
                activation='tanh',
                name="{}_init".format(self.prefix)
            )
            self._add(self.Initializer)

        """
        Encoder Core
        """
        # create RNN cells
        if not self.config['bidirectional']:
            logger.info("{}_create RNN cells.".format(self.prefix))
            self.RNN = RNN(
                self.config['enc_embedd_dim'],
                self.config['enc_hidden_dim'],
                None if not use_context
                else self.config['enc_contxt_dim'],
                name="{}_cell".format(self.prefix)
            )
            self._add(self.RNN)
        else:
            logger.info("{}_create forward RNN cells.".format(self.prefix))
            self.forwardRNN = RNN(
                self.config['enc_embedd_dim'],
                self.config['enc_hidden_dim'],
                None if not use_context
                else self.config['enc_contxt_dim'],
                name="{}_fw_cell".format(self.prefix)
            )
            self._add(self.forwardRNN)

            logger.info("{}_create backward RNN cells.".format(self.prefix))
            self.backwardRNN = RNN(
                self.config['enc_embedd_dim'],
                self.config['enc_hidden_dim'],
                None if not use_context
                else self.config['enc_contxt_dim'],
                name="{}_bw_cell".format(self.prefix)
            )
            self._add(self.backwardRNN)

        logger.info("create encoder ok.")

    def build_encoder(self, source, context=None, return_embed=False,
                      return_sequence=False,
                      return_gates=False,
                      clean_mask=False):
        """
        Build the Encoder Computational Graph

        For the copynet default configurations (with attention)
            return_embed=True,
            return_sequence=True,
            return_gates=True,
            clean_mask=False
        Input:
            source : source text, a list of indexes, shape=[nb_sample, max_len]
            context: None
        Return:
            For Attention model:
                return_sequence=True: to return the embedding at each time, not just the end state
                return_embed=True:
                    X_out:  a list of vectors [nb_sample, max_len, 2*enc_hidden_dim], encoding of each time state (concatenate both forward and backward RNN)
                    X:      embedding of text X [nb_sample, max_len, enc_embedd_dim]
                    X_mask: mask, an array showing which elements in X are not 0 [nb_sample, max_len]
                    X_tail: encoding of ending of X, seems not make sense for bidirectional model (head+tail) [nb_sample, 2*enc_hidden_dim]

        nb_sample:  number of samples, defined by batch size
        max_len:    max length of sentence (lengths of input are same after padding)
        """
        # clean_mask means we set the hidden states of masked places as 0.
        # sometimes it will help the program to solve something
        # note that this option only works when return_sequence.
        # we recommend to leave at least one mask in the end of encoded sequence.

        # Initial state
        Init_h = None
        if self.use_context:
            Init_h = self.Initializer(context)

        # word embedding
        if not self.config['bidirectional']:
            X, X_mask = self.Embed(source, True)
            if return_gates:
                X_out, Z, R = self.RNN(X, X_mask, C=context, init_h=Init_h,
                                       return_sequence=return_sequence,
                                       return_gates=True)
            else:
                X_out     = self.RNN(X, X_mask, C=context, init_h=Init_h,
                                     return_sequence=return_sequence,
                                     return_gates=False)
            if return_sequence:
                X_tail    = X_out[:, -1]

                if clean_mask:
                    X_out     = X_out * X_mask[:, :, None]
            else:
                X_tail    = X_out
        else:
            source2 = source[:, ::-1]
            '''
            Get the embedding of inputs
                shape(X)=[nb_sample, max_len, emb_dim]
                shape(X_mask)=[nb_sample, max_len]
            '''
            X,  X_mask  = self.Embed(source , mask_zero=True)
            X2, X2_mask = self.Embed(source2, mask_zero=True)

            '''
            Get the output after RNN
                return_sequence=True
            '''
            if not return_gates:
                '''
                X_out: hidden state of all times, shape=(nb_samples, max_sent_len, input_emb_dim)
                '''
                X_out1 = self.backwardRNN(X, X_mask,  C=context, init_h=Init_h, return_sequence=return_sequence)
                X_out2 = self.forwardRNN(X2, X2_mask, C=context, init_h=Init_h, return_sequence=return_sequence)
            else:
                '''
                X_out: hidden state of all times, shape=(nb_samples, max_sent_len, input_emb_dim)
                Z:     update gate value, shape=(n_samples, 1)
                R:     reset gate value, shape=(n_samples, 1)
                '''
                X_out1, Z1, R1  = self.backwardRNN(X, X_mask,  C=context, init_h=Init_h,
                                                   return_sequence=return_sequence,
                                                   return_gates=True)
                X_out2, Z2, R2  = self.forwardRNN(X2, X2_mask, C=context, init_h=Init_h,
                                                  return_sequence=return_sequence,
                                                  return_gates=True)
                Z = T.concatenate([Z1, Z2[:, ::-1, :]], axis=2)
                R = T.concatenate([R1, R2[:, ::-1, :]], axis=2)

            if not return_sequence:
                X_out  = T.concatenate([X_out1, X_out2], axis=1)
                X_tail = X_out
            else:
                X_out  = T.concatenate([X_out1, X_out2[:, ::-1, :]], axis=2)
                X_tail = T.concatenate([X_out1[:, -1], X_out2[:, -1]], axis=1)

                if clean_mask:
                    X_out     = X_out * X_mask[:, :, None]

        X_mask  = T.cast(X_mask, dtype='float32')
        if not return_gates:
            if return_embed:
                return X_out, X, X_mask, X_tail
            return X_out
        else:
            if return_embed:
                return X_out, X, X_mask, X_tail, Z, R
            return X_out, Z, R

    def compile_encoder(self, with_context=False, return_embed=False, return_sequence=False):
        source  = T.imatrix()
        self.return_embed = return_embed
        self.return_sequence = return_sequence
        if with_context:
            context = T.matrix()

            self.encode = theano.function([source, context],
                                          self.build_encoder(source, context,
                                                             return_embed=return_embed,
                                                             return_sequence=return_sequence))
            self.gtenc  = theano.function([source, context],
                                          self.build_encoder(source, context,
                                                             return_embed=return_embed,
                                                             return_sequence=return_sequence,
                                                             return_gates=True))
        else:
            """
            return
                X_out:  a list of vectors [nb_sample, max_len, 2*enc_hidden_dim], encoding of each time state (concatenate both forward and backward RNN)
                X:      embedding of text X [nb_sample, max_len, enc_embedd_dim]
                X_mask: mask, an array showing which elements in X are not 0 [nb_sample, max_len]
                X_tail: encoding of end of X, seems not make sense for bidirectional model (head+tail) [nb_sample, 2*enc_hidden_dim]
            """
            self.encode = theano.function([source],
                                          self.build_encoder(source, None,
                                                             return_embed=return_embed,
                                                             return_sequence=return_sequence))

            """
            return
                Z:  value of update gate, shape=(nb_sample, 1)
                R:  value of update gate, shape=(nb_sample, 1)
            """
            self.gtenc  = theano.function([source],
                                          self.build_encoder(source, None,
                                                             return_embed=return_embed,
                                                             return_sequence=return_sequence,
                                                             return_gates=True))
