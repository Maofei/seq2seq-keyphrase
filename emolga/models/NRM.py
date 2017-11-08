import theano
import logging
import copy
import math

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
from emolga.models.decoderAtt import DecoderAtt
from nltk.stem.porter import *

logger = logging.getLogger(__name__)
RNN    = GRU             # change it here for other RNN models.
err    = 1e-9

class NRM(Model):
    """
    Neural Responding Machine
    A Encoder-Decoder based responding model.
    """
    def __init__(self,
                 config, n_rng, rng,
                 mode='Evaluation',
                 use_attention=False,
                 copynet=False,
                 identity=False):
        super(NRM, self).__init__()

        self.config   = config
        self.n_rng    = n_rng  # numpy random stream
        self.rng      = rng  # Theano random stream
        self.mode     = mode  # it's RNN
        self.name     = 'nrm'
        self.attend   = use_attention # True
        self.copynet  = copynet # True
        self.identity = identity # False

    def build_(self, lr=None, iterations=None):
        logger.info("build the Neural Responding Machine")

        # encoder-decoder:: <<==>>
        self.encoder = Encoder(self.config, self.rng, prefix='enc', mode=self.mode)
        if not self.attend:
            self.decoder = Decoder(self.config, self.rng, prefix='dec', mode=self.mode)
        else:
            self.decoder = DecoderAtt(self.config, self.rng, prefix='dec', mode=self.mode,
                                      copynet=self.copynet, identity=self.identity)

        self._add(self.encoder)
        self._add(self.decoder)

        # objectives and optimizers
        if self.config['optimizer'] == 'adam':
            self.optimizer = optimizers.get(self.config['optimizer'],
                                         kwargs=dict(rng=self.rng,
                                                     save=False,
                                                     clipnorm = self.config['clipnorm']
                                                     ))
        else:
            self.optimizer = optimizers.get(self.config['optimizer'])
        if lr is not None:
            self.optimizer.lr.set_value(floatX(lr))
            self.optimizer.iterations.set_value(floatX(iterations))
        logger.info("build ok.")

    def compile_(self, mode='all', contrastive=False):
        # compile the computational graph.
        # INFO: the parameters.
        # mode: 'train'/ 'display'/ 'policy' / 'all'

        # ps = 'params: {\n'
        # for p in self.params:
        #     ps += '{0}: {1}\n'.format(p.name, p.eval().shape)
        # ps += '}.'
        # logger.info(ps)

        param_num = np.sum([np.prod(p.shape.eval()) for p in self.params])
        logger.info("total number of the parameters of the model: {}".format(param_num))

        if mode == 'train' or mode == 'all':
            self.compile_train()

        if mode == 'display' or mode == 'all':
            self.compile_sample()

        if mode == 'inference' or mode == 'all':
            self.compile_inference()

    def compile_train(self):

        # questions (theano variables)
        inputs    = T.imatrix()  # padded input word sequence (for training)
        target    = T.imatrix()  # padded target word sequence (for training)
        cc_matrix = T.tensor3()

        # encoding & decoding

        code, _, c_mask, _ = self.encoder.build_encoder(inputs, None, return_sequence=True, return_embed=True)
        # code: (nb_samples, max_len, contxt_dim)
        if 'explicit_loc' in self.config:
            if self.config['explicit_loc']:
                print('use explicit location!!')
                max_len = code.shape[1]
                expLoc  = T.eye(max_len, self.config['encode_max_len'], dtype='float32')[None, :, :]
                expLoc  = T.repeat(expLoc, code.shape[0], axis=0)
                code    = T.concatenate([code, expLoc], axis=2)

        # self.decoder.build_decoder(target, cc_matrix, code, c_mask)
        #       feed target(index vector of target), cc_matrix(copy matrix), code(encoding of source text), c_mask (mask of source text) into decoder, get objective value
        #       logPxz,logPPL are tensors in [nb_samples,1], cross-entropy and Perplexity of each sample
        # normal seq2seq
        logPxz, logPPL     = self.decoder.build_decoder(target, cc_matrix, code, c_mask)

        # responding loss
        loss_rec = -logPxz
        loss_ppl = T.exp(-logPPL)
        loss     = T.mean(loss_rec)

        updates  = self.optimizer.get_updates(self.params, loss)

        logger.info("compiling the compuational graph ::training function::")

        # input contains inputs, target and cc_matrix
        train_inputs = [inputs, target, cc_matrix]

        self.train_ = theano.function(train_inputs,
                                      [loss_rec, loss_ppl],
                                      updates=updates,
                                      name='train_fun',
                                      allow_input_downcast=True)
        self.train_guard = theano.function(train_inputs,
                                      [loss_rec, loss_ppl],
                                      updates=updates,
                                      name='train_nanguard_fun',
                                      mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
        self.validate_ = theano.function(train_inputs,
                                      [loss_rec, loss_ppl],
                                      name='validate_fun',
                                      allow_input_downcast=True)

        logger.info("training functions compile done.")

        # # add monitoring:
        # self.monitor['context'] = context
        # self._monitoring()
        #
        # # compiling monitoring
        # self.compile_monitoring(train_inputs)

    def compile_sample(self):
        if not self.attend:
            self.encoder.compile_encoder(with_context=False)
        else:
            self.encoder.compile_encoder(with_context=False, return_sequence=True, return_embed=True)

        self.decoder.build_sampler()
        logger.info("sampling functions compile done.")

    def compile_inference(self):
        pass

    def generate_(self, inputs, mode='display', return_attend=False, return_all=False):
        # assert self.config['sample_stoch'], 'RNNLM sampling must be stochastic'
        # assert not self.config['sample_argmax'], 'RNNLM sampling cannot use argmax'

        args = dict(k=self.config['sample_beam'],
                    maxlen=self.config['max_len'],
                    stochastic=self.config['sample_stoch'] if mode == 'display' else None,
                    argmax=self.config['sample_argmax'] if mode == 'display' else None,
                    return_attend=return_attend)
        context, _, c_mask, _, Z, R = self.encoder.gtenc(inputs)
        # c_mask[0, 3] = c_mask[0, 3] * 0
        # L   = context.shape[1]
        # izz = np.concatenate([np.arange(3), np.asarray([1,2]), np.arange(3, L)])
        # context = context[:, izz, :]
        # c_mask  = c_mask[:, izz]
        # inputs  = inputs[:, izz]
        # context, _, c_mask, _ = self.encoder.encode(inputs)
        # import pylab as plt
        # # visualize_(plt.subplots(), Z[0][:, 300:], normal=False)
        # visualize_(plt.subplots(), context[0], normal=False)

        if 'explicit_loc' in self.config:
            if self.config['explicit_loc']:
                max_len = context.shape[1]
                expLoc  = np.eye(max_len, self.config['encode_max_len'], dtype='float32')[None, :, :]
                expLoc  = np.repeat(expLoc, context.shape[0], axis=0)
                context = np.concatenate([context, expLoc], axis=2)

        sample, score, ppp, _    = self.decoder.get_sample(context, c_mask, inputs, **args)
        if return_all:
            return sample, score, ppp

        if not args['stochastic']:
            score  = score / np.array([len(s) for s in sample])
            idz    = score.argmin()
            sample = sample[idz]
            score  = score.min()
            ppp    = ppp[idz]
        else:
            score /= float(len(sample))

        return sample, np.exp(score), ppp


    def generate_multiple(self, inputs, mode='display', return_attend=False, return_all=True, return_encoding=False):
        # assert self.config['sample_stoch'], 'RNNLM sampling must be stochastic'
        # assert not self.config['sample_argmax'], 'RNNLM sampling cannot use argmax'
        args = dict(k=self.config['sample_beam'],
                    maxlen=self.config['max_len'],
                    stochastic=self.config['sample_stoch'] if mode == 'display' else None,
                    argmax=self.config['sample_argmax'] if mode == 'display' else None,
                    return_attend=return_attend,
                    type=self.config['predict_type']
                    )
        '''
        Return the encoding of input.
            Similar to encoder.encode(), but gate values are returned as well
            I think only gtenc with attention
            default: with_context=False, return_sequence=True, return_embed=True
        '''

        """
        return
            context:  a list of vectors [nb_sample, max_len, 2*enc_hidden_dim], encoding of each time state (concatenate both forward and backward RNN)
            _:      embedding of text X [nb_sample, max_len, enc_embedd_dim]
            c_mask: mask, an array showing which elements in context are not 0 [nb_sample, max_len]
            _: encoding of end of X, seems not make sense for bidirectional model (head+tail) [nb_sample, 2*enc_hidden_dim]
            Z:  value of update gate, shape=(nb_sample, 1)
            R:  value of update gate, shape=(nb_sample, 1)
        but.. Z and R are not used here
        """
        context, _, c_mask, _, Z, R = self.encoder.gtenc(inputs)
        # c_mask[0, 3] = c_mask[0, 3] * 0
        # L   = context.shape[1]
        # izz = np.concatenate([np.arange(3), np.asarray([1,2]), np.arange(3, L)])
        # context = context[:, izz, :]
        # c_mask  = c_mask[:, izz]
        # inputs  = inputs[:, izz]
        # context, _, c_mask, _ = self.encoder.encode(inputs)
        # import pylab as plt
        # # visualize_(plt.subplots(), Z[0][:, 300:], normal=False)
        # visualize_(plt.subplots(), context[0], normal=False)

        if 'explicit_loc' in self.config: # no
            if self.config['explicit_loc']:
                max_len = context.shape[1]
                expLoc  = np.eye(max_len, self.config['encode_max_len'], dtype='float32')[None, :, :]
                expLoc  = np.repeat(expLoc, context.shape[0], axis=0)
                context = np.concatenate([context, expLoc], axis=2)

        sample, score, ppp, output_encoding    = self.decoder.get_sample(context, c_mask, inputs, **args)
        if return_all:
            if return_encoding:
                return context, sample, score, output_encoding
            else:
                return sample, score
        return sample, score

    def evaluate_(self, inputs, outputs, idx2word, inputs_unk=None, encode=True):
        def cut_zero_yes(sample, idx2word, ppp=None, Lmax=None):
            if Lmax is None:
                Lmax = self.config['dec_voc_size']
            if ppp is None:
                if 0 not in sample:
                    return ['{}'.format(idx2word[w].encode('utf-8'))
                            if w < Lmax else '{}'.format(idx2word[inputs[w - Lmax]].encode('utf-8'))
                            for w in sample]

                return ['{}'.format(idx2word[w].encode('utf-8'))
                        if w < Lmax else '{}'.format(idx2word[inputs[w - Lmax]].encode('utf-8'))
                        for w in sample[:sample.index(0)]]
            else:
                if 0 not in sample:
                    return ['{0} ({1:1.1f})'.format(
                            idx2word[w].encode('utf-8'), p)
                            if w < Lmax
                            else '{0} ({1:1.1f})'.format(
                            idx2word[inputs[w - Lmax]].encode('utf-8'), p)
                            for w, p in zip(sample, ppp)]
                idz = sample.index(0)
                return ['{0} ({1:1.1f})'.format(
                        idx2word[w].encode('utf-8'), p)
                        if w < Lmax
                        else '{0} ({1:1.1f})'.format(
                        idx2word[inputs[w - Lmax]].encode('utf-8'), p)
                        for w, p in zip(sample[:idz], ppp[:idz])]

    def evaluate_multiple(self, inputs, outputs,
                          original_input, original_outputs,
                          samples, scores, idx2word,
                          number_to_predict=10):
        '''
        inputs_unk is same as inputs except for filtered out all the low-freq words to 1 (<unk>)
        return the top few keywords, number is set in config
        :param: original_input, same as inputs, the vector of one input sentence
        :param: original_outputs, vectors of corresponding multiple outputs (e.g. keyphrases)
        :return:
        '''

        def cut_zero(sample, idx2word, Lmax=None):
            sample = list(sample)
            if Lmax is None:
                Lmax = self.config['dec_voc_size']
            if 0 not in sample:
                return ['{}'.format(idx2word[w].encode('utf-8')) for w in sample]
            # return the string before 0 (<eol>)
            return ['{}'.format(idx2word[w].encode('utf-8')) for w in sample[:sample.index(0)]]

        stemmer = PorterStemmer()
        # Generate keyphrases
        # if inputs_unk is None:
        #     samples, scores = self.generate_multiple(inputs[None, :], return_all=True)
        # else:
        #     samples, scores = self.generate_multiple(inputs_unk[None, :], return_all=True)

        # Evaluation part
        outs = []
        metrics = []

        # load stopword
        with open(self.config['path'] + '/dataset/stopword/stopword_en.txt') as stopword_file:
            stopword_set = set([stemmer.stem(w.strip()) for w in stopword_file])

        for input_sentence, target_list, predict_list, score_list in zip(inputs, original_outputs, samples, scores):
            '''
            enumerate each document, process target/predict/score and measure via p/r/f1
            '''
            target_outputs = []
            predict_outputs = []
            predict_scores = []
            predict_set = set()
            correctly_matched = np.asarray([0] * max(len(target_list), len(predict_list)), dtype='int32')

            # stem the original input
            stemmed_input = [stemmer.stem(w) for w in cut_zero(input_sentence, idx2word)]

            # convert target index into string
            for target in target_list:
                target = cut_zero(target, idx2word)
                target = [stemmer.stem(w) for w in target]

                keep = True
                # whether do filtering on groundtruth phrases. if config['target_filter']==None, do nothing
                if self.config['target_filter']:
                    match = None
                    for i in range(len(stemmed_input) - len(target) + 1):
                        match = None
                        j = 0
                        for j in range(len(target)):
                            if target[j] != stemmed_input[i + j]:
                                match = False
                                break
                        if j == len(target) - 1 and match == None:
                            match = True
                            break

                    if match == True:
                        # if match and 'appear-only', keep this phrase
                        if self.config['target_filter'] == 'appear-only':
                            keep = keep and True
                        elif self.config['target_filter'] == 'non-appear-only':
                            keep = keep and False
                    elif match == False:
                        # if not match and 'appear-only', discard this phrase
                        if self.config['target_filter'] == 'appear-only':
                            keep = keep and False
                        # if not match and 'non-appear-only', keep this phrase
                        elif self.config['target_filter'] == 'non-appear-only':
                            keep = keep and True

                if not keep:
                    continue

                target_outputs.append(target)

            # convert predict index into string
            for id, (predict, score) in enumerate(zip(predict_list, score_list)):
                predict = cut_zero(predict, idx2word)
                predict = [stemmer.stem(w) for w in predict]

                # filter some not good ones
                keep = True
                if len(predict) == 0:
                    keep = False
                number_digit = 0
                for w in predict:
                    if w.strip() == '<unk>':
                        keep = False
                    if w.strip() == '<digit>':
                        number_digit += 1

                if len(predict) >= 1 and (predict[0] in stopword_set or predict[-1] in stopword_set):
                    keep = False

                if len(predict) <= 1:
                    keep = False

                # whether do filtering on predicted phrases. if config['predict_filter']==None, do nothing
                if self.config['predict_filter']:
                    match = None
                    for i in range(len(stemmed_input) - len(predict) + 1):
                        match = None
                        j = 0
                        for j in range(len(predict)):
                            if predict[j] != stemmed_input[i + j]:
                                match = False
                                break
                        if j == len(predict) - 1 and match == None:
                            match = True
                            break

                    if match == True:
                        # if match and 'appear-only', keep this phrase
                        if self.config['predict_filter'] == 'appear-only':
                            keep = keep and True
                        elif self.config['predict_filter'] == 'non-appear-only':
                            keep = keep and False
                    elif match == False:
                        # if not match and 'appear-only', discard this phrase
                        if self.config['predict_filter'] == 'appear-only':
                            keep = keep and False
                        # if not match and 'non-appear-only', keep this phrase
                        elif self.config['predict_filter'] == 'non-appear-only':
                            keep = keep and True

                key = '-'.join(predict)
                # remove this phrase and its score from list
                if not keep or number_digit == len(predict) or key in predict_set:
                    continue

                predict_outputs.append(predict)
                predict_scores.append(score)
                predict_set.add(key)

                # check whether correct
                for target in target_outputs:
                    if len(target) == len(predict):
                        flag = True
                        for i, w in enumerate(predict):
                            if predict[i] != target[i]:
                                flag = False
                        if flag:
                            correctly_matched[len(predict_outputs) - 1] = 1
                            # print('%s correct!!!' % predict)

            predict_outputs = np.asarray(predict_outputs)
            predict_scores = np.asarray(predict_scores)
            # normalize the score?
            if self.config['normalize_score']:
                predict_scores = np.asarray([math.log(math.exp(score) / len(predict)) for predict, score in
                                             zip(predict_outputs, predict_scores)])
                score_list_index = np.argsort(predict_scores)
                predict_outputs = predict_outputs[score_list_index]
                predict_scores = predict_scores[score_list_index]
                correctly_matched = correctly_matched[score_list_index]

            metric_dict = {}
            metric_dict['p'] = float(sum(correctly_matched[:number_to_predict])) / float(number_to_predict)

            if len(target_outputs) != 0:
                metric_dict['r'] = float(sum(correctly_matched[:number_to_predict])) / float(len(target_outputs))
            else:
                metric_dict['r'] = 0

            if metric_dict['p'] + metric_dict['r'] != 0:
                metric_dict['f1'] = 2 * metric_dict['p'] * metric_dict['r'] / float(
                    metric_dict['p'] + metric_dict['r'])
            else:
                metric_dict['f1'] = 0

            metric_dict['valid_target_number'] = len(target_outputs)
            metric_dict['target_number'] = len(target_list)
            metric_dict['correct_number'] = sum(correctly_matched[:number_to_predict])

            metrics.append(metric_dict)

            # print(stuff)
            a = '[SOURCE]: {}\n'.format(' '.join(cut_zero(input_sentence, idx2word)))
            logger.info(a)

            b = '[TARGET]: %d/%d targets\n\t\t' % (len(target_outputs), len(target_list))
            for id, target in enumerate(target_outputs):
                b += ' '.join(target) + '; '
            b += '\n'
            logger.info(b)
            c = '[DECODE]: %d/%d predictions' % (len(predict_outputs), len(predict_list))
            for id, (predict, score) in enumerate(zip(predict_outputs, predict_scores)):
                if correctly_matched[id] == 1:
                    c += ('\n\t\t[%.3f]' % score) + ' '.join(predict) + ' [correct!]'
                    # print(('\n\t\t[%.3f]'% score) + ' '.join(predict) + ' [correct!]')
                else:
                    c += ('\n\t\t[%.3f]' % score) + ' '.join(predict)
                    # print(('\n\t\t[%.3f]'% score) + ' '.join(predict))
            c += '\n'

            # c = '[DECODE]: {}'.format(' '.join(cut_zero(phrase, idx2word)))
            # if inputs_unk is not None:
            #     k = '[_INPUT]: {}\n'.format(' '.join(cut_zero(inputs_unk.tolist(),  idx2word, Lmax=len(idx2word))))
            #     logger.info(k)
            # a += k
            logger.info(c)
            a += b + c
            d = 'Precision=%.4f, Recall=%.4f, F1=%.4f\n' % (metric_dict['p'], metric_dict['r'], metric_dict['f1'])
            logger.info(d)
            a += d

            outs.append(a)

        return outs, metrics

        def cut_zero_no(sample, idx2word, ppp=None, Lmax=None):
            if Lmax is None:
                Lmax = self.config['dec_voc_size']
            if ppp is None:
                if 0 not in sample:
                    return ['{}'.format(idx2word[w])
                            if w < Lmax else '{}'.format(idx2word[inputs[w - Lmax]].encode('utf-8'))
                            for w in sample]

                return ['{}'.format(idx2word[w])
                        if w < Lmax else '{}'.format(idx2word[inputs[w - Lmax]].encode('utf-8'))
                        for w in sample[:sample.index(0)]]
            else:
                if 0 not in sample:
                    return ['{0} ({1:1.1f})'.format(
                            idx2word[w], p)
                            if w < Lmax
                            else '{0} ({1:1.1f})'.format(
                            idx2word[inputs[w - Lmax]], p)
                            for w, p in zip(sample, ppp)]
                idz = sample.index(0)
                return ['{0} ({1:1.1f})'.format(
                        idx2word[w].encode('utf-8'), p)
                        if w < Lmax
                        else '{0} ({1:1.1f})'.format(
                        idx2word[inputs[w - Lmax]], p)
                        for w, p in zip(sample[:idz], ppp[:idz])]

        if inputs_unk is None:
            result, _, ppp = self.generate_(inputs[None, :])
        else:
            result, _, ppp = self.generate_(inputs_unk[None, :])

        if encode:
            cut_zero = cut_zero_yes
        else:
            cut_zero = cut_zero_no
        pp0, pp1 = [np.asarray(p) for p in zip(*ppp)]
        pp = (pp1 - pp0) / pp1
        # pp = (pp1 - pp0) / pp1
        logger.info(len(ppp))

        logger.info('<Environment> [lr={0}][iter={1}]'.format(self.optimizer.lr.get_value(),
                                                        self.optimizer.iterations.get_value()))

        a = '[SOURCE]: {}\n'.format(' '.join(cut_zero(inputs.tolist(),  idx2word, Lmax=len(idx2word))))
        b = '[TARGET]: {}\n'.format(' '.join(cut_zero(outputs.tolist(), idx2word, Lmax=len(idx2word))))
        c = '[DECODE]: {}\n'.format(' '.join(cut_zero(result, idx2word)))
        d = '[CpRate]: {}\n'.format(' '.join(cut_zero(result, idx2word, pp.tolist())))
        e = '[CpRate]: {}\n'.format(' '.join(cut_zero(result, idx2word, result)))
        logger.info(a)
        logger.info( '{0} -> {1}'.format(len(a.split()), len(b.split())))

        if inputs_unk is not None:
            k = '[_INPUT]: {}\n'.format(' '.join(cut_zero(inputs_unk.tolist(),  idx2word, Lmax=len(idx2word))))
            logger.info( k )
            a += k

        logger.info(b)
        logger.info(c)
        logger.info(d)
        # print(e)
        a += b + c + d
        return a

    def analyse_(self, inputs, outputs, idx2word, inputs_unk=None, return_attend=False, name=None, display=False):
        def cut_zero(sample, idx2word, ppp=None, Lmax=None):
            if Lmax is None:
                Lmax = self.config['dec_voc_size']
            if ppp is None:
                if 0 not in sample:
                    return ['{}'.format(idx2word[w].encode('utf-8'))
                            if w < Lmax else '{}'.format(idx2word[inputs[w - Lmax]].encode('utf-8'))
                            for w in sample]

                return ['{}'.format(idx2word[w].encode('utf-8'))
                        if w < Lmax else '{}'.format(idx2word[inputs[w - Lmax]].encode('utf-8'))
                        for w in sample[:sample.index(0)]]
            else:
                if 0 not in sample:
                    return ['{0} ({1:1.1f})'.format(
                            idx2word[w].encode('utf-8'), p)
                            if w < Lmax
                            else '{0} ({1:1.1f})'.format(
                            idx2word[inputs[w - Lmax]].encode('utf-8'), p)
                            for w, p in zip(sample, ppp)]
                idz = sample.index(0)
                return ['{0} ({1:1.1f})'.format(
                        idx2word[w].encode('utf-8'), p)
                        if w < Lmax
                        else '{0} ({1:1.1f})'.format(
                        idx2word[inputs[w - Lmax]].encode('utf-8'), p)
                        for w, p in zip(sample[:idz], ppp[:idz])]

        if inputs_unk is None:
            result, _, ppp = self.generate_(inputs[None, :],
                                            return_attend=return_attend)
        else:
            result, _, ppp = self.generate_(inputs_unk[None, :],
                                            return_attend=return_attend)

        source = '{}'.format(' '.join(cut_zero(inputs.tolist(),  idx2word, Lmax=len(idx2word))))
        target = '{}'.format(' '.join(cut_zero(outputs.tolist(), idx2word, Lmax=len(idx2word))))
        decode = '{}'.format(' '.join(cut_zero(result, idx2word)))

        if display:
            print(source)
            print(target)
            print(decode)

            idz    = result.index(0)
            p1, p2 = [np.asarray(p) for p in zip(*ppp)]
            print(p1.shape)
            import pylab as plt
            # plt.rc('text', usetex=True)
            # plt.rc('font', family='serif')
            visualize_(plt.subplots(), 1 - p1[:idz, :].T, grid=True, name=name)
            visualize_(plt.subplots(), 1 - p2[:idz, :].T, name=name)

            # visualize_(plt.subplots(), 1 - np.mean(p2[:idz, :], axis=1, keepdims=True).T)
        return target == decode

    def analyse_cover(self, inputs, outputs, idx2word, inputs_unk=None, return_attend=False, name=None, display=False):
        def cut_zero(sample, idx2word, ppp=None, Lmax=None):
            if Lmax is None:
                Lmax = self.config['dec_voc_size']
            if ppp is None:
                if 0 not in sample:
                    return ['{}'.format(idx2word[w].encode('utf-8'))
                            if w < Lmax else '{}'.format(idx2word[inputs[w - Lmax]].encode('utf-8'))
                            for w in sample]

                return ['{}'.format(idx2word[w].encode('utf-8'))
                        if w < Lmax else '{}'.format(idx2word[inputs[w - Lmax]].encode('utf-8'))
                        for w in sample[:sample.index(0)]]
            else:
                if 0 not in sample:
                    return ['{0} ({1:1.1f})'.format(
                            idx2word[w].encode('utf-8'), p)
                            if w < Lmax
                            else '{0} ({1:1.1f})'.format(
                            idx2word[inputs[w - Lmax]].encode('utf-8'), p)
                            for w, p in zip(sample, ppp)]
                idz = sample.index(0)
                return ['{0} ({1:1.1f})'.format(
                        idx2word[w].encode('utf-8'), p)
                        if w < Lmax
                        else '{0} ({1:1.1f})'.format(
                        idx2word[inputs[w - Lmax]].encode('utf-8'), p)
                        for w, p in zip(sample[:idz], ppp[:idz])]

        if inputs_unk is None:
            results, _, ppp = self.generate_(inputs[None, :],
                                            return_attend=return_attend,
                                            return_all=True)
        else:
            results, _, ppp = self.generate_(inputs_unk[None, :],
                                            return_attend=return_attend,
                                            return_all=True)

        source = '{}'.format(' '.join(cut_zero(inputs.tolist(),  idx2word, Lmax=len(idx2word))))
        target = '{}'.format(' '.join(cut_zero(outputs.tolist(), idx2word, Lmax=len(idx2word))))
        # decode = '{}'.format(' '.join(cut_zero(result, idx2word)))

        score  = [target == '{}'.format(' '.join(cut_zero(result, idx2word))) for result in results]
        return max(score)
