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
from emolga.models.decoder import Decoder

from nltk.stem.porter import *

logger = logging.getLogger(__name__)
RNN    = GRU             # change it here for other RNN models.
err    = 1e-9

class DecoderAtt(Decoder):
    """
    Recurrent Neural Network-based Decoder [for CopyNet-b Only]
    with Attention Mechanism
    """
    def __init__(self,
                 config, rng, prefix='dec',
                 mode='RNN', embed=None,
                 copynet=False, identity=False):
        super(DecoderAtt, self).__init__(
                config, rng, prefix,
                 mode, embed, False)
        self.init     = initializations.get('glorot_uniform')
        self.copynet  = copynet
        self.identity = identity
        # attention reader
        self.attention_reader = Attention(
            self.config['dec_hidden_dim'],
            self.config['dec_contxt_dim'],
            1000,
            name='source_attention',
            coverage=self.config['coverage']
        )
        self._add(self.attention_reader)

        # if use copynet
        if self.copynet:
            if not self.identity:
                self.Is = Dense(
                    self.config['dec_contxt_dim'],
                    self.config['dec_embedd_dim'],
                    name='in-trans'
                )
            else:
                assert self.config['dec_contxt_dim'] == self.config['dec_embedd_dim']
                self.Is = Identity(name='ini')

            self.Os = Dense(
                self.config['dec_readout_dim']
                if not self.config['location_embed']
                    else self.config['dec_readout_dim'] + self.config['dec_embedd_dim'],
                self.config['dec_contxt_dim'],
                name='out-trans'
            )

            if self.config['copygate']:
                self.Gs = Dense(
                    self.config['dec_readout_dim'] + self.config['dec_embedd_dim'],
                    1,
                    name='copy-gate',
                    activation='linear',
                    learn_bias=True,
                    negative_bias=True
                )
                self._add(self.Gs)

            if self.config['location_embed']:
                self._add(self.Is)
            self._add(self.Os)

        logger.info('adjust decoder ok.')

    """
    Build the decoder for evaluation
    """
    def prepare_xy(self, target, cc_matrix):
        # target:      (nb_samples, index_seq)
        # cc_matrix:   (nb_samples, maxlen_t, maxlen_s)
        # context:     (nb_samples)
        Y,  Y_mask  = self.Embed(target, True)  # (nb_samples, maxlen_t, embedding_dim)
        X           = T.concatenate([alloc_zeros_matrix(Y.shape[0], 1, Y.shape[2]), Y[:, :-1, :]], axis=1)

        # LL          = T.concatenate([alloc_zeros_matrix(Y.shape[0], 1, cc_matrix.shape[2]),
        #                              cc_matrix[:, :-1, :]], axis=1)
        LL = cc_matrix

        XL_mask     = T.cast(T.gt(T.sum(LL, axis=2), 0), dtype='float32')
        if not self.config['use_input']:
            X *= 0

        X_mask    = T.concatenate([T.ones((Y.shape[0], 1)), Y_mask[:, :-1]], axis=1)
        Count     = T.cast(T.sum(X_mask, axis=1), dtype=theano.config.floatX)
        return X, X_mask, LL, XL_mask, Y_mask, Count

    """
    The most different part. Be cautious!!
    Very different from traditional RNN search.
    """
    def build_decoder(self,
                      target,
                      cc_matrix,
                      context,
                      c_mask,
                      return_count=False,
                      train=True):
        """
        Build the Computational Graph ::> Context is essential
        """
        assert c_mask is not None, 'context must be supplied for this decoder.'
        assert context.ndim == 3, 'context must have 3 dimentions.'
        # context: (nb_samples, max_len, contxt_dim)
        context_A = self.Is(context)  # (nb_samples, max_len, embed_dim)
        X, X_mask, LL, XL_mask, Y_mask, Count = self.prepare_xy(target, cc_matrix)

        # input drop-out if any.
        if self.dropout > 0:
            X     = self.D(X, train=train)

        # Initial state of RNN
        Init_h   = self.Initializer(context[:, 0, :])  # default order ->
        Init_a   = T.zeros((context.shape[0], context.shape[1]), dtype='float32')
        coverage = T.zeros((context.shape[0], context.shape[1]), dtype='float32')

        X        = X.dimshuffle((1, 0, 2))
        X_mask   = X_mask.dimshuffle((1, 0))
        LL       = LL.dimshuffle((1, 0, 2))            # (maxlen_t, nb_samples, maxlen_s)
        XL_mask  = XL_mask.dimshuffle((1, 0))          # (maxlen_t, nb_samples)

        def _recurrence(x, x_mask, ll, xl_mask, prev_h, prev_a, cov, cc, cm, ca):
            """
            x:      (nb_samples, embed_dims)
            x_mask: (nb_samples, )
            ll:     (nb_samples, maxlen_s)
            xl_mask:(nb_samples, )
            -----------------------------------------
            prev_h: (nb_samples, hidden_dims)
            prev_a: (nb_samples, maxlen_s)
            cov:    (nb_samples, maxlen_s)  *** coverage ***
            -----------------------------------------
            cc:     (nb_samples, maxlen_s, context_dim)
            cm:     (nb_samples, maxlen_s)
            ca:     (nb_samples, maxlen_s, ebd_dim)
            """
            # compute the attention and get the context vector
            prob  = self.attention_reader(prev_h, cc, Smask=cm, Cov=cov)
            ncov  = cov + prob

            cxt   = T.sum(cc * prob[:, :, None], axis=1)

            # compute input word embedding (mixed)
            x_in  = T.concatenate([x, T.sum(ca * prev_a[:, :, None], axis=1)], axis=-1)

            # compute the current hidden states of the RNN.
            x_out = self.RNN(x_in, mask=x_mask, C=cxt, init_h=prev_h, one_step=True)

            # compute the current readout vector.
            r_in  = [x_out]
            if self.config['context_predict']:
                r_in  += [cxt]
            if self.config['bigram_predict']:
                r_in  += [x_in]

            # copynet decoding
            r_in    = T.concatenate(r_in, axis=-1)
            r_out = self.hidden_readout(x_out)  # (nb_samples, voc_size)
            if self.config['context_predict']:
                r_out += self.context_readout(cxt)
            if self.config['bigram_predict']:
                r_out += self.prev_word_readout(x_in)

            for l in self.output_nonlinear:
                r_out = l(r_out)

            key     = self.Os(r_in)  # (nb_samples, cxt_dim) :: key
            Eng     = T.sum(key[:, None, :] * cc, axis=-1)

            # # gating
            if self.config['copygate']:
                gt     = self.sigmoid(self.Gs(r_in))  # (nb_samples, 1)
                r_out += T.log(gt.flatten()[:, None])
                Eng   += T.log(1 - gt.flatten()[:, None])

                # r_out *= gt.flatten()[:, None]
                # Eng   *= 1 - gt.flatten()[:, None]

            EngSum  = logSumExp(Eng, axis=-1, mask=cm, c=r_out)

            next_p  = T.concatenate([T.exp(r_out - EngSum), T.exp(Eng - EngSum) * cm], axis=-1)
            next_c  = next_p[:, self.config['dec_voc_size']:] * ll           # (nb_samples, maxlen_s)
            next_b  = next_p[:, :self.config['dec_voc_size']]
            sum_a   = T.sum(next_c, axis=1, keepdims=True)                   # (nb_samples,)
            next_a  = (next_c / (sum_a + err)) * xl_mask[:, None]            # numerically consideration
            return x_out, next_a, ncov, sum_a, next_b

        outputs, _ = theano.scan(
            _recurrence,
            sequences=[X, X_mask, LL, XL_mask],
            outputs_info=[Init_h, Init_a, coverage, None, None],
            non_sequences=[context, c_mask, context_A]
        )
        X_out, source_prob, coverages, source_sum, prob_dist = [z.dimshuffle((1, 0, 2)) for z in outputs]
        X        = X.dimshuffle((1, 0, 2))
        X_mask   = X_mask.dimshuffle((1, 0))
        XL_mask  = XL_mask.dimshuffle((1, 0))

        # unk masking
        U_mask   = T.ones_like(target) * (1 - T.eq(target, 1))
        U_mask  += (1 - U_mask) * (1 - XL_mask)

        # The most different part is here !!
        log_prob = T.sum(T.log(
                   T.clip(self._grab_prob(prob_dist, target) * U_mask + source_sum.sum(axis=-1) + err, 1e-10, 1.0)
                   ) * X_mask, axis=1)
        log_ppl  = log_prob / (Count + err)

        if return_count:
            return log_prob, Count
        else:
            return log_prob, log_ppl

    """
    Sample one step
    """

    def _step_sample(self,
                     prev_word,
                     prev_stat,
                     prev_loc,
                     prev_cov,
                     context,
                     c_mask,
                     context_A):
        """
        Get the probability of next word, sec 3.2 and 3.3
        :param prev_word    :   index of previous words, size=(1, live_k)
        :param prev_stat    :   output encoding of last time, size=(1, live_k, output_dim)
        :param prev_loc     :   information needed for copy-based predicting
        :param prev_cov     :   information needed for copy-based predicting
        :param context      :   encoding of source text, shape = [live_k, sent_len, 2*output_dim]
        :param c_mask       :   mask fof source text, shape = [live_k, sent_len]
        :param context_A: an identity layer (do nothing but return the context)
        :returns:
            next_prob       : probabilities of next word, shape=(1, voc_size+sent_len)
                                next_prob0[:voc_size] is generative probability
                                next_prob0[voc_size:voc_size+sent_len] is copy probability
            next_sample     : only useful for stochastic
            next_stat       : output (decoding) vector after time t
            ncov            :
            next_stat       :
        """

        assert c_mask is not None, 'we need the source mask.'
        # word embedding (note that for the first word, embedding should be all zero)
        # if prev_word[:, None] < 0 (only the starting sysbol index=-1)
        #   then return zeros
        #       return alloc_zeros_matrix(prev_word.shape[0], 2 * self.config['dec_embedd_dim']),
        #   else return embedding of the previous words
        #       return self.Embed(prev_word)

        X = T.switch(
            prev_word[:, None] < 0,
            alloc_zeros_matrix(prev_word.shape[0], 2 * self.config['dec_embedd_dim']),
            T.concatenate([self.Embed(prev_word),
                           T.sum(context_A * prev_loc[:, :, None], axis=1)
                           ], axis=-1)
        )

        if self.dropout > 0:
            X = self.D(X, train=False)

        # apply one step of RNN
        Probs  = self.attention_reader(prev_stat, context, c_mask, Cov=prev_cov)
        ncov   = prev_cov + Probs

        cxt    = T.sum(context * Probs[:, :, None], axis=1)

        X_proj, zz, rr = self.RNN(X, C=cxt,
                                  init_h=prev_stat,
                                  one_step=True,
                                  return_gates=True)
        next_stat = X_proj

        # compute the readout probability distribution and sample it
        # here the readout is a matrix, different from the learner.
        readin      = [next_stat]
        if self.config['context_predict']:
            readin += [cxt]
        if self.config['bigram_predict']:
            readin += [X]

        # if gating
        # if self.config['copygate']:
        #     gt      = self.sigmoid(self.Gs(readin))   # (nb_samples, dim)
        #     readin *= 1 - gt
        #     readout = self.hidden_readout(next_stat * gt[:, :self.config['dec_hidden_dim']])
        #     if self.config['context_predict']:
        #         readout += self.context_readout(
        #                 cxt * gt[:, self.config['dec_hidden_dim']:
        #                          self.config['dec_hidden_dim'] + self.config['dec_contxt_dim']])
        #     if self.config['bigram_predict']:
        #         readout += self.prev_word_readout(
        #                 X * gt[:, -2 * self.config['dec_embedd_dim']:])
        # else:
        readout = self.hidden_readout(next_stat)
        if self.config['context_predict']:
            readout += self.context_readout(cxt)
        if self.config['bigram_predict']:
            readout += self.prev_word_readout(X)

        for l in self.output_nonlinear:
            readout = l(readout)

        readin      = T.concatenate(readin, axis=-1)
        key         = self.Os(readin)
        Eng         = T.sum(key[:, None, :] * context, axis=-1)

        # # gating
        if self.config['copygate']:
            gt       = self.sigmoid(self.Gs(readin))  # (nb_samples, 1)
            readout += T.log(gt.flatten()[:, None])
            Eng     += T.log(1 - gt.flatten()[:, None])

        EngSum      = logSumExp(Eng, axis=-1, mask=c_mask, c=readout)

        next_prob   = T.concatenate([T.exp(readout - EngSum), T.exp(Eng - EngSum) * c_mask], axis=-1)
        next_sample = self.rng.multinomial(pvals=next_prob).argmax(1)
        return next_prob, next_sample, next_stat, ncov, next_stat

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
        context   = T.tensor3()  # theano variable. shape=(n_sample, sent_len, 2*output_dim)
        c_mask    = T.matrix()   # mask of the input sentence.
        context_A = self.Is(context) # an identity layer (do nothing but return the context)

        init_h = self.Initializer(context[:, 0, :])
        init_a = T.zeros((context.shape[0], context.shape[1]))
        cov    = T.zeros((context.shape[0], context.shape[1]))

        logger.info('compile the function: get_init_state')
        self.get_init_state \
            = theano.function([context], [init_h, init_a, cov], name='get_init_state')
        logger.info('done.')

        # word sampler: 1 x 1
        prev_word = T.vector('prev_word', dtype='int64')
        prev_stat = T.matrix('prev_state', dtype='float32')
        prev_a    = T.matrix('prev_a', dtype='float32')
        prev_cov  = T.matrix('prev_cov', dtype='float32')

        next_prob, next_sample, next_stat, ncov, alpha \
            = self._step_sample(prev_word,
                                prev_stat,
                                prev_a,
                                prev_cov,
                                context,
                                c_mask,
                                context_A)

        # next word probability
        logger.info('compile the function: sample_next')
        inputs  = [prev_word, prev_stat, prev_a, prev_cov, context, c_mask]
        outputs = [next_prob, next_sample, next_stat, ncov, alpha]
        self.sample_next = theano.function(inputs, outputs, name='sample_next')
        logger.info('done')

    """
    Generate samples, either with stochastic sampling or beam-search!

    [:-:] I have to think over how to modify the BEAM-Search!!
    """
    def get_sample(self,
                   context,  # the RNN encoding of source text at each time step, shape = [1, sent_len, 2*output_dim]
                   c_mask,  # shape = [1, sent_len]
                   sources,  # shape = [1, sent_len]
                   k=1, maxlen=30, stochastic=True,  # k = config['sample_beam'], maxlen = config['max_len']
                   argmax=False, fixlen=False,
                   return_attend=False,
                   type='extractive',
                   generate_ngram=True
                   ):
        # beam size
        if k > 1:
            assert not stochastic, 'Beam search does not support stochastic sampling!!'

        # fix length cannot use beam search
        # if fixlen:
        #     assert k == 1

        # prepare for searching
        Lmax   = self.config['dec_voc_size']
        sample = [] # predited sequences
        attention_probs    = [] # don't know what's this
        attend = []
        score  = [] # probability of predited sequences
        state = [] # the output encoding of predited sequences

        if stochastic:
            score = 0

        live_k = 1
        dead_k = 0

        hyp_samples = [[]] * live_k
        hyp_scores  = np.zeros(live_k).astype(theano.config.floatX)
        hyp_attention_probs    = [[]] * live_k
        hyp_attends = [[]] * live_k

        # get initial state of decoder RNN with encoding
        #   feed in the encoding of time=0(why 0?! because the X_out of RNN is reverse?), do tanh(W*x+b) and output next_state shape=[1,output_dim]
        #   copy_word_prob and coverage are zeros[context.shape]
        previous_state, copy_word_prob, coverage = self.get_init_state(context)
        # indicator for the first target word (bos target), starts with [-1]
        previous_word = -1 * np.ones((1,)).astype('int32')

        # if aim is extractive, then set the initial beam size to be voc_size
        if type == 'extractive':
            input = sources[0]
            input_set = set(input)
            sequence_set = set()

            if generate_ngram:
                for i in range(len(input)): # loop over start
                    for j in range(1, maxlen): # loop over length
                        if i+j > len(input)-1:
                            break
                        hash_token = [str(s) for s in input[i:i+j]]
                        sequence_set.add('-'.join(hash_token))
                logger.info("Possible n-grams: %d" % len(sequence_set))

        # Start searching!
        for ii in range(maxlen):
            # make live_k copies of context, c_mask and source, to predict next words at once.
            #   np.tile(context, [live_k, 1, 1]) means copying along the axis=0
            context_copies     = np.tile(context, [live_k, 1, 1]) # shape = [live_k, sent_len, 2*output_dim]
            c_mask_copies      = np.tile(c_mask,  [live_k, 1])    # shape = [live_k, sent_len]
            source_copies      = np.tile(sources, [live_k, 1])    # shape = [live_k, sent_len]

            # process word
            def process_():
                """
                copy_mask[i] indicates which words in source have been copied (whether the previous_word[i] appears in source text)
                size = size(source_copies) = [live_k, sent_len]
                Caution:     word2idx['<eol>'] = 0, word2idx['<unk>'] = 1
                """
                copy_mask  = np.zeros((source_copies.shape[0], source_copies.shape[1]), dtype='float32')

                for i in range(previous_word.shape[0]): # loop over the previous_words, index of previous words, size=(1, live_k)
                    #   Note that the model predict a OOV word in the way like voc_size+position_in_source
                    #   if a previous predicted word is OOV (next_word[i] >= Lmax):
                    #       means it predicts the position of word in source text (next_word[i]=voc_size+position_in_source)
                    #           1. set copy_mask to 1, indicates which last word is copied;
                    #           2. set next_word to the real index of this word (source_copies[previous_word[i] - Lmax])
                    #   else:
                    #       means not a OOV word, but may be still copied from source
                    #       check if any word in source_copies[i] is same to previous_word[i]
                    if previous_word[i] >= Lmax:
                        copy_mask[i][previous_word[i] - Lmax] = 1.
                        previous_word[i] = source_copies[i][previous_word[i] - Lmax]
                    else:
                        copy_mask[i] = (source_copies[i] == previous_word[i, None])
                        # for k in range(sss.shape[1]):
                        #     ll[i][k] = (sss[i][k] == next_word[i])
                return copy_mask, previous_word

            copy_mask, previous_word = process_()
            copy_flag = (np.sum(copy_mask, axis=1, keepdims=True) > 0) # boolean indicates if any copy available

            # get the copy probability (eq 6 in paper?)
            next_a  = copy_word_prob * copy_mask # keep the copied ones
            next_a  = next_a / (err + np.sum(next_a, axis=1, keepdims=True)) * copy_flag # normalize
            '''
            Get the probability of next word, sec 3.2 and 3.3
                Return:
                    next_prob0  : probabilities of next word, shape=(live_k, voc_size+sent_len)
                                    next_prob0[:, :voc_size] is generative probability
                                    next_prob0[:, voc_size:voc_size+sent_len] is copy probability
                    next_word   : only useful for stochastic
                    next_state  : output (decoding) vector after time t
                    coverage    :
                    alpha       : just next_state, only useful if return_attend

                Inputs:
                    previous_word       : index of previous words, size=(1, live_k)
                    previous_state      : output encoding of last time, size=(1, live_k, output_dim)
                    next_a, coverage    : information needed for copy-based predicting
                    encoding_copies     : shape = [live_k, sent_len, 2*output_dim]
                    c_mask_copies       : shape = [live_k, sent_len]

                    if don't do copying, only previous_word,previous_state,context_copies,c_mask_copies are needed for predicting
            '''
            next_prob0, next_word, next_state, coverage, alpha \
                = self.sample_next(previous_word, previous_state, next_a, coverage, context_copies, c_mask_copies)
            if not self.config['decode_unk']: # eliminate the probability of <unk>
                next_prob0[:, 1]          = 0.
                next_prob0 /= np.sum(next_prob0, axis=1, keepdims=True)

            def merge_():
                # merge the probabilities, p(w) = p_generate(w)+p_copy(w)
                temple_prob  = copy.copy(next_prob0)
                source_prob  = copy.copy(next_prob0[:, Lmax:])
                for i in range(next_prob0.shape[0]): # loop over all the previous words
                    for j in range(source_copies.shape[1]): # loop over all the source words
                        if (source_copies[i, j] < Lmax) and (source_copies[i, j] != 1): # if word source_copies[i, j] in voc and not a unk
                            temple_prob[i, source_copies[i, j]] += source_prob[i, j] # add the copy prob to generative prob
                            temple_prob[i, Lmax + j]   = 0. # set the corresponding copy prob to be 0

                return temple_prob, source_prob
            # if word in voc, add the copy prob to generative prob and keep generate prob only, else keep the copy prob only
            generate_word_prob, copy_word_prob   = merge_()
            next_prob0[:, Lmax:] = 0. # [not quite useful]set the latter (copy) part to be zeros, actually next_prob0 become really generate_word_prob
            # print('0', next_prob0[:, 3165])
            # print('01', next_prob[:, 3165])
            # # print(next_prob[0, Lmax:])
            # print(ss_prob[0, :])

            if stochastic:
                # using stochastic sampling (or greedy sampling.)
                if argmax:
                    nw = generate_word_prob[0].argmax()
                    next_word[0] = nw
                else:
                    nw = self.rng.multinomial(pvals=generate_word_prob).argmax(1)

                sample.append(nw)
                score += generate_word_prob[0, nw]

                if (not fixlen) and (nw == 0):  # sample reached the end
                    break

            else:
                '''
                using beam-search, keep the top (k-dead_k) results (dead_k is disabled by memray)
                we can only computed in a flatten way!
                '''
                # add the score of new predicted word to the score of whole sequence, the reason why the score of longer sequence getting smaller
                #       add a 1e-10 to avoid log(0)
                #       size(hyp_scores)=[live_k,1], size(generate_word_prob)=[live_k,voc_size+sent_len]
                cand_scores     = hyp_scores[:, None] - np.log(generate_word_prob + 1e-10)
                cand_flat       = cand_scores.flatten()
                ranks_flat      = cand_flat.argsort()[:(k - dead_k)] # get the index of top k predictions

                # recover(stack) the flat results, fetch the best results.
                voc_size        = generate_word_prob.shape[1]
                sequence_index  = ranks_flat / voc_size # flat_index/voc_size is the original sequence index
                next_word_index = ranks_flat % voc_size # flat_index%voc_size is the original word index
                costs           = cand_flat[ranks_flat]

                # get the new hyp samples
                new_hyp_samples         = []
                new_hyp_attention_probs = []
                new_hyp_attends         = []
                new_hyp_scores          = np.zeros(k - dead_k).astype(theano.config.floatX)
                new_hyp_states          = []
                new_hyp_coverage        = []
                new_hyp_copy_word_prob  = []

                for idx, [ti, wi] in enumerate(zip(sequence_index, next_word_index)):
                    ti = int(ti)
                    wi = int(wi)
                    new_hyp_samples.append(hyp_samples[ti] + [wi])
                    new_hyp_scores[idx] = copy.copy(costs[idx])

                    new_hyp_states.append(copy.copy(next_state[ti]))
                    new_hyp_coverage.append(copy.copy(coverage[ti]))
                    new_hyp_copy_word_prob.append(copy.copy(copy_word_prob[ti]))

                    # what's the ppp? generative attention and copy attention?
                    if not return_attend:
                        # probability of current predicted word (generative part and both generative/copying part)
                        new_hyp_attention_probs.append(hyp_attention_probs[ti] + [[next_prob0[ti][wi], generate_word_prob[ti][wi]]])
                    else:
                        # copying probability and attention probability of current predicted word
                        new_hyp_attention_probs.append(hyp_attention_probs[ti] + [(copy_word_prob[ti], alpha[ti])])

                # check the finished samples
                new_live_k          = 0
                hyp_samples         = []
                hyp_scores          = []
                hyp_states          = []
                hyp_coverage        = []
                hyp_attention_probs = []
                hyp_copy_word_prob  = []

                for idx in range(len(new_hyp_samples)):
                    # [bug] change to new_hyp_samples[idx][-1] == 0
                    # if (new_hyp_states[idx][-1] == 0) and (not fixlen):
                    if (new_hyp_samples[idx][-1] == 0 and not fixlen):
                        '''
                        predict an <eos>, this sequence is done
                        put successful prediction into result list
                        '''
                        # worth-noting that if the word index is larger than voc_size, it means a OOV word
                        sample.append(new_hyp_samples[idx])
                        attention_probs.append(new_hyp_attention_probs[idx])
                        score.append(new_hyp_scores[idx])
                        state.append(new_hyp_states[idx])
                        # dead_k += 1
                    if new_hyp_samples[idx][-1] != 0:
                        '''
                        sequence prediction not complete
                        put into candidate list for next round prediction
                        '''
                        # limit predictions must appear in text
                        if type == 'extractive':
                            if new_hyp_samples[idx][-1] not in input_set:
                                continue
                            if generate_ngram:
                                if '-'.join([str(s) for s in new_hyp_samples[idx]]) not in sequence_set:
                                    continue

                        new_live_k += 1
                        hyp_samples.append(new_hyp_samples[idx])
                        hyp_attention_probs.append(new_hyp_attention_probs[idx])
                        hyp_scores.append(new_hyp_scores[idx])
                        hyp_states.append(new_hyp_states[idx])
                        hyp_coverage.append(new_hyp_coverage[idx])
                        hyp_copy_word_prob.append(new_hyp_copy_word_prob[idx])

                hyp_scores = np.array(hyp_scores)
                live_k = new_live_k

                if new_live_k < 1:
                    break
                # if dead_k >= k:
                #     break

                # prepare the variables for predicting next round
                previous_word   = np.array([w[-1] for w in hyp_samples])
                previous_state  = np.array(hyp_states)
                coverage        = np.array(hyp_coverage)
                copy_word_prob  = np.array(hyp_copy_word_prob)
                pass

            logger.info('\t Depth=%d, get %d outputs' % (ii, len(sample)))

        # end.
        if not stochastic:
            # dump every remaining one
            if live_k > 0:
                for idx in range(live_k):
                    sample.append(hyp_samples[idx])
                    attention_probs.append(hyp_attention_probs[idx])
                    score.append(hyp_scores[idx])
                    state.append(hyp_states[idx])

        # sort the result
        result = zip(sample, score, attention_probs, state)
        sorted_result = sorted(result, key=lambda entry: entry[1], reverse=False)
        sample, score, attention_probs, state = zip(*sorted_result)
        return sample, score, attention_probs, state
