import theano
import logging
import copy
import math

from emolga.layers.core import Dropout, Dense, Dense2, Identity
from emolga.layers.recurrent import *
from emolga.layers.ntm_minibatch import Controller
from emolga.layers.embeddings import *
from emolga.layers.attention import *
from emolga.models.core import Model

from nltk.stem.porter import *

logger = logging.getLogger(__name__)
RNN    = GRU             # change it here for other RNN models.
err    = 1e-9

class Decoder(Model):
    """
    Recurrent Neural Network-based Decoder.
    It is used for:
        (1) Evaluation: compute the probability P(Y|X)
        (2) Prediction: sample the best result based on P(Y|X)
        (3) Beam-search
        (4) Scheduled Sampling (how to implement it?)
    """

    def __init__(self,
                 config, rng, prefix='dec',
                 mode='RNN', embed=None,
                 highway=False):
        """
        mode = RNN: use a RNN Decoder
        """
        super(Decoder, self).__init__()
        self.config = config
        self.rng = rng
        self.prefix = prefix
        self.name = prefix
        self.mode = mode

        self.highway = highway
        self.init = initializations.get('glorot_uniform')
        self.sigmoid = activations.get('sigmoid')

        # use standard drop-out for input & output.
        # I believe it should not use for context vector.
        self.dropout = config['dropout']
        if self.dropout > 0:
            logger.info('Use standard-dropout!!!!')
            self.D   = Dropout(rng=self.rng, p=self.dropout, name='{}_Dropout'.format(prefix))

        """
        Create all elements of the Decoder's computational graph.
        """
        # create Embedding layers
        logger.info("{}_create embedding layers.".format(self.prefix))
        if embed:
            self.Embed = embed
        else:
            self.Embed = Embedding(
                self.config['dec_voc_size'],
                self.config['dec_embedd_dim'], # 150
                name="{}_embed".format(self.prefix))
            self._add(self.Embed)

        # create Initialization Layers
        logger.info("{}_create initialization layers.".format(self.prefix))
        if not config['bias_code']:
            self.Initializer = Zero()
        else:
            self.Initializer = Dense(
                config['dec_contxt_dim'],
                config['dec_hidden_dim'],
                activation='tanh',
                name="{}_init".format(self.prefix)
            )

        # create RNN cells
        logger.info("{}_create RNN cells.".format(self.prefix))
        if 'location_embed' in self.config:
            if config['location_embed']:
                dec_embedd_dim = 2 * self.config['dec_embedd_dim']
            else:
                dec_embedd_dim = self.config['dec_embedd_dim']
        else:
            dec_embedd_dim = self.config['dec_embedd_dim']

        self.RNN = RNN(
            dec_embedd_dim,
            self.config['dec_hidden_dim'],
            self.config['dec_contxt_dim'],
            name="{}_cell".format(self.prefix)
        )

        self._add(self.Initializer)
        self._add(self.RNN)

        # HighWay Gating
        if highway:
            logger.info("HIGHWAY CONNECTION~~~!!!")
            assert self.config['context_predict']
            assert self.config['dec_contxt_dim'] == self.config['dec_hidden_dim']

            self.C_x = self.init((self.config['dec_contxt_dim'],
                                  self.config['dec_hidden_dim']))
            self.H_x = self.init((self.config['dec_hidden_dim'],
                                  self.config['dec_hidden_dim']))
            self.b_x = initializations.get('zero')(self.config['dec_hidden_dim'])

            self.C_x.name = '{}_Cx'.format(self.prefix)
            self.H_x.name = '{}_Hx'.format(self.prefix)
            self.b_x.name = '{}_bx'.format(self.prefix)
            self.params += [self.C_x, self.H_x, self.b_x]

        # create readout layers
        logger.info("_create Readout layers")

        # 1. hidden layers readout.
        self.hidden_readout = Dense(
            self.config['dec_hidden_dim'],
            self.config['output_dim']
            if self.config['deep_out']
            else self.config['dec_voc_size'],
            activation='linear',
            name="{}_hidden_readout".format(self.prefix)
        )

        # 2. previous word readout
        self.prev_word_readout = None
        if self.config['bigram_predict']:
            self.prev_word_readout = Dense(
                dec_embedd_dim,
                self.config['output_dim']
                if self.config['deep_out']
                else self.config['dec_voc_size'],
                activation='linear',
                name="{}_prev_word_readout".format(self.prefix),
                learn_bias=False
            )

        # 3. context readout
        self.context_readout = None
        if self.config['context_predict']:
            if not self.config['leaky_predict']:
                self.context_readout = Dense(
                    self.config['dec_contxt_dim'],
                    self.config['output_dim']
                    if self.config['deep_out']
                    else self.config['dec_voc_size'],
                    activation='linear',
                    name="{}_context_readout".format(self.prefix),
                    learn_bias=False
                )
            else:
                assert self.config['dec_contxt_dim'] == self.config['dec_hidden_dim']
                self.context_readout = self.hidden_readout

        # option: deep output (maxout)
        if self.config['deep_out']:
            self.activ = Activation(config['deep_out_activ'])
            # self.dropout = Dropout(rng=self.rng, p=config['dropout'])
            self.output_nonlinear = [self.activ]  # , self.dropout]
            self.output = Dense(
                self.config['output_dim'] / 2
                if config['deep_out_activ'] == 'maxout2'
                else self.config['output_dim'],

                self.config['dec_voc_size'],
                activation='softmax',
                name="{}_output".format(self.prefix),
                learn_bias=False
            )
        else:
            self.output_nonlinear = []
            self.output = Activation('softmax')

        # registration:
        self._add(self.hidden_readout)

        if not self.config['leaky_predict']:
            self._add(self.context_readout)

        self._add(self.prev_word_readout)
        self._add(self.output)

        if self.config['deep_out']:
            self._add(self.activ)
        # self._add(self.dropout)

        logger.info("create decoder ok.")

    @staticmethod
    def _grab_prob(probs, X, block_unk=False):
        assert probs.ndim == 3

        batch_size = probs.shape[0]
        max_len = probs.shape[1]
        vocab_size = probs.shape[2]

        probs = probs.reshape((batch_size * max_len, vocab_size))
        return probs[T.arange(batch_size * max_len), X.flatten(1)].reshape(X.shape)  # advanced indexing

    """
    Build the decoder for evaluation
    """
    def prepare_xy(self, target):
        # Word embedding
        Y, Y_mask = self.Embed(target, True)  # (nb_samples, max_len, embedding_dim)

        if self.config['use_input']:
            X = T.concatenate([alloc_zeros_matrix(Y.shape[0], 1, Y.shape[2]), Y[:, :-1, :]], axis=1)
        else:
            X = 0 * Y

        # option ## drop words.

        X_mask    = T.concatenate([T.ones((Y.shape[0], 1)), Y_mask[:, :-1]], axis=1)
        Count     = T.cast(T.sum(X_mask, axis=1), dtype=theano.config.floatX)
        return X, X_mask, Y, Y_mask, Count

    def build_decoder(self, target, context=None,
                      return_count=False,
                      train=True):

        """
        Build the Decoder Computational Graph
        For training/testing
        """
        X, X_mask, Y, Y_mask, Count = self.prepare_xy(target)

        # input drop-out if any.
        if self.dropout > 0:
            X = self.D(X, train=train)

        # Initial state of RNN
        Init_h = self.Initializer(context)
        if not self.highway:
            X_out  = self.RNN(X, X_mask, C=context, init_h=Init_h, return_sequence=True)

            # Readout
            readout = self.hidden_readout(X_out)
            if self.dropout > 0:
                readout = self.D(readout, train=train)

            if self.config['context_predict']:
                readout += self.context_readout(context).dimshuffle(0, 'x', 1)
        else:
            X      = X.dimshuffle((1, 0, 2))
            X_mask = X_mask.dimshuffle((1, 0))

            def _recurrence(x, x_mask, prev_h, c):
                # compute the highway gate for context vector.
                xx    = dot(c, self.C_x, self.b_x) + dot(prev_h, self.H_x)  # highway gate.
                xx    = self.sigmoid(xx)

                cy    = xx * c   # the path without using RNN
                x_out = self.RNN(x, mask=x_mask, C=c, init_h=prev_h, one_step=True)
                hx    = (1 - xx) * x_out
                return x_out, hx, cy

            outputs, _ = theano.scan(
                _recurrence,
                sequences=[X, X_mask],
                outputs_info=[Init_h, None, None],
                non_sequences=[context]
            )

            # hidden readout + context readout
            readout   = self.hidden_readout( outputs[1].dimshuffle((1, 0, 2)))
            if self.dropout > 0:
                readout = self.D(readout, train=train)

            readout  += self.context_readout(outputs[2].dimshuffle((1, 0, 2)))

            # return to normal size.
            X      = X.dimshuffle((1, 0, 2))
            X_mask = X_mask.dimshuffle((1, 0))

        if self.config['bigram_predict']:
            readout += self.prev_word_readout(X)

        for l in self.output_nonlinear:
            readout = l(readout)

        prob_dist = self.output(readout)  # (nb_samples, max_len, vocab_size)
        # log_old  = T.sum(T.log(self._grab_prob(prob_dist, target)), axis=1)
        log_prob = T.sum(T.log(self._grab_prob(prob_dist, target) + err) * X_mask, axis=1)
        log_ppl  = log_prob / Count

        if return_count:
            return log_prob, Count
        else:
            return log_prob, log_ppl

    """
    Sample one step
    """

    def _step_sample(self, prev_word, prev_stat, context):
        # word embedding (note that for the first word, embedding should be all zero)
        if self.config['use_input']:
            X = T.switch(
                prev_word[:, None] < 0,
                alloc_zeros_matrix(prev_word.shape[0], self.config['dec_embedd_dim']),
                self.Embed(prev_word)
            )
        else:
            X = alloc_zeros_matrix(prev_word.shape[0], self.config['dec_embedd_dim'])

        if self.dropout > 0:
            X = self.D(X, train=False)

        # apply one step of RNN
        if not self.highway:
            X_proj = self.RNN(X, C=context, init_h=prev_stat, one_step=True)
            next_stat = X_proj

            # compute the readout probability distribution and sample it
            # here the readout is a matrix, different from the learner.
            readout = self.hidden_readout(next_stat)
            if self.dropout > 0:
                readout = self.D(readout, train=False)

            if self.config['context_predict']:
                readout += self.context_readout(context)
        else:
            xx     = dot(context, self.C_x, self.b_x) + dot(prev_stat, self.H_x)  # highway gate.
            xx     = self.sigmoid(xx)

            X_proj = self.RNN(X, C=context, init_h=prev_stat, one_step=True)
            next_stat = X_proj

            readout  = self.hidden_readout((1 - xx) * X_proj)
            if self.dropout > 0:
                readout = self.D(readout, train=False)

            readout += self.context_readout(xx * context)

        if self.config['bigram_predict']:
            readout += self.prev_word_readout(X)

        for l in self.output_nonlinear:
            readout = l(readout)

        next_prob = self.output(readout)
        next_sample = self.rng.multinomial(pvals=next_prob).argmax(1)
        return next_prob, next_sample, next_stat

    """
    Build the sampler for sampling/greedy search/beam search
    """

    def build_sampler(self):
        """
        Build a sampler which only steps once.
        Typically it only works for one word a time?
        """
        logger.info("build sampler ...")
        if self.config['sample_stoch'] and self.config['sample_argmax']:
            logger.info("use argmax search!")
        elif self.config['sample_stoch'] and (not self.config['sample_argmax']):
            logger.info("use stochastic sampling!")
        elif self.config['sample_beam'] > 1:
            logger.info("use beam search! (beam_size={})".format(self.config['sample_beam']))

        # initial state of our Decoder.
        context = T.matrix()  # theano variable.

        init_h = self.Initializer(context)
        logger.info('compile the function: get_init_state')
        self.get_init_state \
            = theano.function([context], init_h, name='get_init_state')
        logger.info('done.')

        # word sampler: 1 x 1
        prev_word = T.vector('prev_word', dtype='int64')
        prev_stat = T.matrix('prev_state', dtype='float32')
        next_prob, next_sample, next_stat \
            = self._step_sample(prev_word, prev_stat, context)

        # next word probability
        logger.info('compile the function: sample_next')
        inputs = [prev_word, prev_stat, context]
        outputs = [next_prob, next_sample, next_stat]

        self.sample_next = theano.function(inputs, outputs, name='sample_next')
        logger.info('done')
        pass

    """
    Build a Stochastic Sampler which can use SCAN to work on GPU.
    However it cannot be used in Beam-search.
    """

    def build_stochastic_sampler(self):
        context = T.matrix()
        init_h = self.Initializer(context)

        logger.info('compile the function: sample')
        pass

    """
    Generate samples, either with stochastic sampling or beam-search!
    """

    def get_sample(self, context, k=1, maxlen=30, stochastic=True, argmax=False, fixlen=False):
        # beam size
        if k > 1:
            assert not stochastic, 'Beam search does not support stochastic sampling!!'

        # fix length cannot use beam search
        # if fixlen:
        #     assert k == 1

        # prepare for searching
        sample = []
        score = []
        if stochastic:
            score = 0

        live_k = 1
        dead_k = 0

        hyp_samples = [[]] * live_k
        hyp_scores = np.zeros(live_k).astype(theano.config.floatX)
        hyp_states = []

        # get initial state of decoder RNN with context
        next_state = self.get_init_state(context)
        next_word = -1 * np.ones((1,)).astype('int64')  # indicator for the first target word (bos target)

        # Start searching!
        for ii in range(maxlen):
            # print(next_word)
            ctx = np.tile(context, [live_k, 1])
            next_prob, next_word, next_state \
                = self.sample_next(next_word, next_state, ctx)  # wtf.

            if stochastic:
                # using stochastic sampling (or greedy sampling.)
                if argmax:
                    nw = next_prob[0].argmax()
                    next_word[0] = nw
                else:
                    nw = next_word[0]

                sample.append(nw)
                score += next_prob[0, nw]

                if (not fixlen) and (nw == 0):  # sample reached the end
                    break

            else:
                # using beam-search
                # we can only computed in a flatten way!
                cand_scores = hyp_scores[:, None] - np.log(next_prob)
                cand_flat = cand_scores.flatten()
                ranks_flat = cand_flat.argsort()[:(k - dead_k)]

                # fetch the best results.
                voc_size = next_prob.shape[1]
                trans_index = ranks_flat / voc_size
                word_index = ranks_flat % voc_size
                costs = cand_flat[ranks_flat]

                # get the new hyp samples
                new_hyp_samples = []
                new_hyp_scores = np.zeros(k - dead_k).astype(theano.config.floatX)
                new_hyp_states = []

                for idx, [ti, wi] in enumerate(zip(trans_index, word_index)):
                    new_hyp_samples.append(hyp_samples[ti] + [wi])
                    new_hyp_scores[idx] = copy.copy(costs[idx])
                    new_hyp_states.append(copy.copy(next_state[ti]))

                # check the finished samples
                new_live_k = 0
                hyp_samples = []
                hyp_scores = []
                hyp_states = []

                for idx in range(len(new_hyp_samples)):
                    if (new_hyp_states[idx][-1] == 0) and (not fixlen):
                        sample.append(new_hyp_samples[idx])
                        score.append(new_hyp_scores[idx])
                        dead_k += 1
                    else:
                        new_live_k += 1
                        hyp_samples.append(new_hyp_samples[idx])
                        hyp_scores.append(new_hyp_scores[idx])
                        hyp_states.append(new_hyp_states[idx])

                hyp_scores = np.array(hyp_scores)
                live_k = new_live_k

                if new_live_k < 1:
                    break
                if dead_k >= k:
                    break

                next_word = np.array([w[-1] for w in hyp_samples])
                next_state = np.array(hyp_states)
                pass
            pass

        # end.
        if not stochastic:
            # dump every remaining one
            if live_k > 0:
                for idx in range(live_k):
                    sample.append(hyp_samples[idx])
                    score.append(hyp_scores[idx])

        return sample, score
