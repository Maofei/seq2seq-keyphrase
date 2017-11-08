__author__ = 'jiataogu, memray'
import theano
import logging
import copy
import math

import emolga.basic.objectives as objectives
import emolga.basic.optimizers as optimizers

from theano.compile.nanguardmode import NanGuardMode
from emolga.utils.generic_utils import visualize_
from emolga.layers.core import Dropout, Dense, Dense2, Identity
from emolga.layers.recurrent import *
from emolga.layers.ntm_minibatch import Controller
from emolga.layers.embeddings import *
from emolga.layers.attention import *
from emolga.models.core import Model
from emolga.models.encoder import Encoder
from emolga.models.decoder import Decoder
from nltk.stem.porter import *

logger = logging.getLogger(__name__)
RNN    = GRU             # change it here for other RNN models.
err    = 1e-9

class FnnDecoder(Model):
    def __init__(self, config, rng, prefix='fnndec'):
        """
        mode = RNN: use a RNN Decoder
        """
        super(FnnDecoder, self).__init__()
        self.config = config
        self.rng = rng
        self.prefix = prefix
        self.name = prefix

        """
        Create Dense Predictor.
        """

        self.Tr = Dense(self.config['dec_contxt_dim'],
                             self.config['dec_hidden_dim'],
                             activation='maxout2',
                             name='{}_Tr'.format(prefix))
        self._add(self.Tr)

        self.Pr = Dense(self.config['dec_hidden_dim'] / 2,
                             self.config['dec_voc_size'],
                             activation='softmax',
                             name='{}_Pr'.format(prefix))
        self._add(self.Pr)
        logger.info("FF decoder ok.")

    @staticmethod
    def _grab_prob(probs, X):
        assert probs.ndim == 3

        batch_size = probs.shape[0]
        max_len = probs.shape[1]
        vocab_size = probs.shape[2]

        probs = probs.reshape((batch_size * max_len, vocab_size))
        return probs[T.arange(batch_size * max_len), X.flatten(1)].reshape(X.shape)  # advanced indexing

    def build_decoder(self, target, context):
        """
        Build the Decoder Computational Graph
        """
        prob_dist = self.Pr(self.Tr(context[:, None, :]))
        log_prob  = T.sum(T.log(self._grab_prob(prob_dist, target) + err), axis=1)
        return log_prob

    def build_sampler(self):
        context   = T.matrix()
        prob_dist = self.Pr(self.Tr(context))
        next_sample = self.rng.multinomial(pvals=prob_dist).argmax(1)
        self.sample_next = theano.function([context], [prob_dist, next_sample], name='sample_next_{}'.format(self.prefix))
        logger.info('done')

    def get_sample(self, context, argmax=True):

        prob, sample = self.sample_next(context)
        if argmax:
            return prob[0].argmax()
        else:
            return sample[0]


########################################################################################################################
# Encoder-Decoder Models ::::
#
class RNNLM(Model):
    """
    RNN-LM, with context vector = 0.
    It is very similar with the implementation of VAE.
    """
    def __init__(self,
                 config, n_rng, rng,
                 mode='Evaluation'):
        super(RNNLM, self).__init__()

        self.config = config
        self.n_rng  = n_rng  # numpy random stream
        self.rng    = rng  # Theano random stream
        self.mode   = mode
        self.name   = 'rnnlm'

    def build_(self):
        logger.info("build the RNN-decoder")
        self.decoder = Decoder(self.config, self.rng, prefix='dec', mode=self.mode)

        # registration:
        self._add(self.decoder)

        # objectives and optimizers
        self.optimizer = optimizers.get('adadelta')

        # saved the initial memories
        if self.config['mode'] == 'NTM':
            self.memory    = initializations.get('glorot_uniform')(
                    (self.config['dec_memory_dim'], self.config['dec_memory_wdth']))

        logger.info("create the RECURRENT language model. ok")

    def compile_(self, mode='train', contrastive=False):
        # compile the computational graph.
        # INFO: the parameters.
        # mode: 'train'/ 'display'/ 'policy' / 'all'

        ps = 'params: {\n'
        for p in self.params:
            ps += '{0}: {1}\n'.format(p.name, p.eval().shape)
        ps += '}.'
        logger.info(ps)

        param_num = np.sum([np.prod(p.shape.eval()) for p in self.params])
        logger.info("total number of the parameters of the model: {}".format(param_num))

        if mode == 'train' or mode == 'all':
            if not contrastive:
                self.compile_train()
            else:
                self.compile_train_CE()

        if mode == 'display' or mode == 'all':
            self.compile_sample()

        if mode == 'inference' or mode == 'all':
            self.compile_inference()

    def compile_train(self):

        # questions (theano variables)
        inputs  = T.imatrix()  # padded input word sequence (for training)
        if self.config['mode']   == 'RNN':
            context = alloc_zeros_matrix(inputs.shape[0], self.config['dec_contxt_dim'])
        elif self.config['mode'] == 'NTM':
            context = T.repeat(self.memory[None, :, :], inputs.shape[0], axis=0)
        else:
            raise NotImplementedError

        # decoding.
        target  = inputs
        logPxz, logPPL = self.decoder.build_decoder(target, context)

        # reconstruction loss
        loss_rec = T.mean(-logPxz)
        loss_ppl = T.exp(T.mean(-logPPL))

        L1       = T.sum([T.sum(abs(w)) for w in self.params])
        loss     = loss_rec

        updates = self.optimizer.get_updates(self.params, loss)

        logger.info("compiling the compuational graph ::training function::")
        train_inputs = [inputs]

        self.train_ = theano.function(train_inputs,
                                      [loss_rec, loss_ppl],
                                      updates=updates,
                                      name='train_fun'
                                      )
        logger.info("pre-training functions compile done.")

        # add monitoring:
        self.monitor['context'] = context
        self._monitoring()

        # compiling monitoring
        self.compile_monitoring(train_inputs)

    @abstractmethod
    def compile_train_CE(self):
        pass

    def compile_sample(self):
        # context vectors (as)
        self.decoder.build_sampler()
        logger.info("display functions compile done.")

    @abstractmethod
    def compile_inference(self):
        pass

    def default_context(self):
        if self.config['mode'] == 'RNN':
            return np.zeros(shape=(1, self.config['dec_contxt_dim']), dtype=theano.config.floatX)
        elif self.config['mode'] == 'NTM':
            memory = self.memory.get_value()
            memory = memory.reshape((1, memory.shape[0], memory.shape[1]))
            return memory

    def generate_(self, context=None, max_len=None, mode='display'):
        """
        :param action: action vector to guide the question.
                       If None, use a Gaussian to simulate the action.
        :return: question sentence in natural language.
        """
        # assert self.config['sample_stoch'], 'RNNLM sampling must be stochastic'
        # assert not self.config['sample_argmax'], 'RNNLM sampling cannot use argmax'

        if context is None:
            context = self.default_context()

        args = dict(k=self.config['sample_beam'],
                    maxlen=self.config['max_len'] if not max_len else max_len,
                    stochastic=self.config['sample_stoch'] if mode == 'display' else None,
                    argmax=self.config['sample_argmax'] if mode == 'display' else None)

        sample, score = self.decoder.get_sample(context, **args)
        if not args['stochastic']:
            score = score / np.array([len(s) for s in sample])
            sample = sample[score.argmin()]
            score = score.min()
        else:
            score /= float(len(sample))

        return sample, np.exp(score)


class AutoEncoder(RNNLM):
    """
    Regular Auto-Encoder: RNN Encoder/Decoder
    """

    def __init__(self,
                 config, n_rng, rng,
                 mode='Evaluation'):
        super(RNNLM, self).__init__()

        self.config = config
        self.n_rng  = n_rng  # numpy random stream
        self.rng    = rng  # Theano random stream
        self.mode   = mode
        self.name = 'vae'

    def build_(self):
        logger.info("build the RNN auto-encoder")
        self.encoder = Encoder(self.config, self.rng, prefix='enc')
        if self.config['shared_embed']:
            self.decoder = Decoder(self.config, self.rng, prefix='dec', embed=self.encoder.Embed)
        else:
            self.decoder = Decoder(self.config, self.rng, prefix='dec')

        """
        Build the Transformation
        """
        if self.config['nonlinear_A']:
            self.action_trans = Dense(
                self.config['enc_hidden_dim'],
                self.config['action_dim'],
                activation='tanh',
                name='action_transform'
            )
        else:
            assert self.config['enc_hidden_dim'] == self.config['action_dim'], \
                    'hidden dimension must match action dimension'
            self.action_trans = Identity(name='action_transform')

        if self.config['nonlinear_B']:
            self.context_trans = Dense(
                self.config['action_dim'],
                self.config['dec_contxt_dim'],
                activation='tanh',
                name='context_transform'
            )
        else:
            assert self.config['dec_contxt_dim'] == self.config['action_dim'], \
                    'action dimension must match context dimension'
            self.context_trans = Identity(name='context_transform')

        # registration
        self._add(self.action_trans)
        self._add(self.context_trans)
        self._add(self.encoder)
        self._add(self.decoder)

        # objectives and optimizers
        self.optimizer = optimizers.get(self.config['optimizer'], kwargs={'lr': self.config['lr']})

        logger.info("create Helmholtz RECURRENT neural network. ok")

    def compile_train(self, mode='train'):
        # questions (theano variables)
        inputs  = T.imatrix()  # padded input word sequence (for training)
        context = alloc_zeros_matrix(inputs.shape[0], self.config['dec_contxt_dim'])
        assert context.ndim == 2

        # decoding.
        target  = inputs
        logPxz, logPPL = self.decoder.build_decoder(target, context)

        # reconstruction loss
        loss_rec = T.mean(-logPxz)
        loss_ppl = T.exp(T.mean(-logPPL))

        L1       = T.sum([T.sum(abs(w)) for w in self.params])
        loss     = loss_rec

        updates = self.optimizer.get_updates(self.params, loss)

        logger.info("compiling the compuational graph ::training function::")
        train_inputs = [inputs]

        self.train_ = theano.function(train_inputs,
                                      [loss_rec, loss_ppl],
                                      updates=updates,
                                      name='train_fun')
        logger.info("pre-training functions compile done.")

        if mode == 'display' or mode == 'all':
            """
            build the sampler function here <:::>
            """
            # context vectors (as)
            self.decoder.build_sampler()
            logger.info("display functions compile done.")

        # add monitoring:
        self._monitoring()

        # compiling monitoring
        self.compile_monitoring(train_inputs)
