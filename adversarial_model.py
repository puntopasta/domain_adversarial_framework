from keras.models import Model
from keras import layers
from keras.optimizers import Nadam
import numpy as np
import keras, os
import tensorflow as tf
from keras.engine import Layer
import keras.backend as K
from scipy import stats
from sklearn.metrics import precision_recall_curve


def reverse_gradient(X, hp_lambda):
    '''Flips the sign of the incoming gradient during training.'''
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]

    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y


class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = hp_lambda

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'hp_lambda': self.hp_lambda}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Architecture:

    def __init__(self, n_layers_ar=5, n_layers_fft=3, time_distributed=False, adversarial_strength = 0.6):
        self.n_layers_ar = n_layers_ar
        self.n_layers_fft = n_layers_fft
        self.time_distributed = time_distributed
        self.adversarial_strength = adversarial_strength


    def encoder(self, inp_ar, inp_fft):
        '''
        Example encoder architecture using a causal convolutional architecture
        Inspired by Wavenet: Oord, Aaron van den, et al. "Wavenet: A generative model for raw audio." arXiv preprint arXiv:1609.03499 (2016).
        '''
        def conv_block(inp, scope, n_layers):
            skips = []
            block = inp
            for i in range(n_layers):
                n_filters = np.power(2, n_layers - i + 2)
                dilation = int(np.power(2, i))
                block = layers.Conv1D(filters=n_filters, kernel_size=5,
                                      dilation_rate=dilation, padding='causal',
                                      name=scope + '_c{i}_f32k5d{d}'.format(i=i, d=dilation))(block)

                block = layers.BatchNormalization(name=scope + '_bn_{}'.format(i))(block)

                block = layers.LeakyReLU(name=scope + '_relu_{}'.format(i))(block)

                skip = layers.Conv1D(1, 1, kernel_initializer='ones',
                                     name=scope + '_skip_{}'.format(i))(block)
                skips.append(skip)

            block = layers.Conv1D(filters=1, kernel_size=5, dilation_rate=int(np.power(2, n_layers)),
                                  padding='causal', name=scope + '_c_final')(block)
            block = layers.BatchNormalization(name=scope + '_bn_final')(block)
            block = layers.LeakyReLU(name=scope + '_relu_final')(block)
            return block, skips

        ar_block, ar_skips = conv_block(inp_ar, scope='ar', n_layers=5)
        concat = layers.concatenate([ar_skips].append(ar_block))
        return concat

    def classifier(self, inp, output_shape):
        '''
        The classifier takes the encoder and predicts the target labels of the model.
        '''
        lstm = layers.CuDNNLSTM(32, return_sequences=True, name='final_lstm')(inp)
        return layers.TimeDistributed(layers.Dense(output_shape[-1], activation='sigmoid'), name='classifier')(lstm)

    def adversarial(self, inp, domain_shape):
        '''
        The adverasrial classifier takes the encoder and predicts the domain.
        '''
        adversarial_model = keras.models.Sequential()
        adversarial_model.add(GradientReversal(self.adversarial_strength, name='gradient_reversal'))
        if not self.time_distributed:
            adversarial_model.add(layers.Flatten(name='flatten_matrix'))
        adversarial_model.add(layers.Dense(domain_shape[-1], activation='softmax', name='domain_classifier'))

        return adversarial_model(inp), adversarial_model


from keras.layers import Dropout,Dense,BatchNormalization, Bidirectional, CuDNNLSTM, Concatenate
from keras.constraints import maxnorm


class EarlyStop:
    '''
        Basic early stopping implementation for domain adversarial training loops.
    '''
    def __init__(self, model_dir, early_stop_metric, early_stop_limit, sacred_object = None):
        '''
            model_dir: where to store intermediate models
            early_stop_metric: name of the metric to track
            early_stop_limit: patience of the early stopping algorithm
            sacred_object: integrates with sacred experiment tracking if a sacred object is provided.
        '''
        self.early_stop_metric = early_stop_metric
        self.model_dir = model_dir
        self.early_stop_limit = early_stop_limit
        self.sacred_object = sacred_object
        self.last_es_value = -1
        self.last_es_update = 0

        self.best_models = []

    def check_early_stop(self, es_value, epoch_num, model):
        '''
        es_value: the value to check for early stopping criteron
        epoch_num: epoch number
        model: the model to be saved when a new optimum is reached.
        '''
        if es_value > self.last_es_value:
            self.last_es_value = es_value
            last_es_update = 0
            model_name = 'weights_epoch_{e}_{es_metric}_{l:.2f}.h5'.format(e=epoch_num,
                                                                              l=self.last_es_value,
                                                                              es_metric=self.early_stop_metric)
            filename = os.path.join(self.model_dir, model_name)
            print('Top ROC AUC: {v:.2f} for model {m}'.format(v=self.last_es_value, m=filename))
            model.save_weights(filename)
            if self.sacred_object is not None:
                self.sacred_object.add_artifact(filename, name=model_name)

            self.best_models.append(filename)
        else:
            self.last_es_update = self.last_es_update + 1

        return self.last_es_update == self.early_stop_limit

    def get_best_model(self):
        return self.best_models[-1]


class AdversarialModel:
'''
    Class that trains adverarial models.
'''
    history = {
        'train_loss': [],
        'train_acc': [],
        'pr_rec_auc_train': [],
        'train_a_loss': [],
        'train_a_acc': [],
        'test_loss': [],
        'test_acc': [],
        'pr_rec_auc_test': [],
        'n_adv_training': []
    }

    def __init__(self, input_shapes, output_shape, domain_shape, model_dir,
                 mode='normal', delay_adversary_epochs=0, architecture=None, sacred_object=None):
        '''
            input_shapes: shape of model inputs
            output_shape: shape of model output
            domain_shape: shape of the domain
            model_dir: directory to store model files
            mode:   'normal' (Default) where the model is trained without adverary, 
                    'adversarial' to train model in adversarial fashion  (see Ganin, Yaroslav, et al. "Domain-adversarial training of neural networks." The Journal of Machine Learning Research 17.1 (2016): 2096-2030. APA)
                    'conditional' trains model in conditional adverarial fashion (see Zhao, Mingmin, et al. "Learning sleep stages from radio signals: A conditional adversarial architecture." Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017.)
            delay_adversarial_epochs: number of epochs to postpone adversarial training (only applies when mode = 'adversarial')
            architecture: architecture object (instance of Architecture class)
            sacred_object: optional sacred object which will be used to record metrics in sacred

        '''
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if len([f for f in os.listdir(model_dir) if f.find('classifier') >= 0]) > 0:
            raise FileExistsError('Model directory is not empty: {}.'.format(model_dir))
        self.model_dir = model_dir
        self.mode = mode

        self.sacred_object = sacred_object

        if architecture is None:
            architecture = Architecture()
        self.time_distributed = architecture.time_distributed

        inputs = [layers.Input(shape[1:], name=n) for shape, n in zip(input_shapes, ['INP_AR','INP_FFT'])]

        # base encoder and classifier blocks
        e = architecture.encoder(inputs[0], inputs[1])
        c = architecture.classifier(e, output_shape=output_shape)

        # latent feature extraction model
        self.latent_feature_extractor = Model(inputs=inputs, outputs=e)

        # classifier model
        self.classifier_model = Model(inputs=inputs, outputs=c)
        self.classifier_model.compile(loss='binary_crossentropy', optimizer=Nadam(lr=0.0001),metrics=['acc'], sample_weight_mode='temporal')

        self.delay_adversary_epochs = delay_adversary_epochs
        if self.mode == 'adversarial' or self.mode == 'normal':
            a, self.adversarial_model_disconnected = architecture.adversarial(e, domain_shape=domain_shape)

            self.adversarial_model = Model(inputs=inputs, outputs=a)

            self.adversarial_model.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=0.0001), metrics=['acc'])
            self.adversarial_model_disconnected.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=0.0001),metrics=['acc'])
        elif self.mode == 'conditional':
            # conditional adversarial model
            inp_conditional = layers.Input((c.shape.as_list()[1], c.shape.as_list()[2]), name='COND-INPUT')
            latent = layers.concatenate([e, inp_conditional])
            cond_a, self.adversarial_model_disconnected= architecture.adversarial(latent, domain_shape=domain_shape)
            self.adversarial_model = Model(inputs=inputs+[inp_conditional], outputs=cond_a)

            self.adversarial_model.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=0.0001), metrics=['acc'])
            self.adversarial_model_disconnected.compile(loss='categorical_crossentropy', optimizer=Nadam(lr=0.0001),metrics=['acc'])

    @classmethod
    def train_generator(cls, x, y, d, w, m, batch_size):
        while True:
            idx = np.arange(len(x[0]))
            np.random.shuffle(idx)
            for i in np.arange(0, len(x[0]), batch_size):
                selection = idx[i:i + batch_size]
                yield [xi[selection] for xi in x], y[selection], d[selection], w[selection], m[selection]

    def get_epoch_nums_passed(self):
        return len(self.history[list(self.history.keys())[0]])

    def log_batch_level(self, key, value, cur_batch):
        if self.sacred_object is not None:
            count = self.train_batches * self.get_epoch_nums_passed() + cur_batch
            self.sacred_object.log_scalar(self.run_prefix+'_'+key, value, count)

    def log_epoch_level(self, key, value):
        self.history[key].append(value)
        if self.sacred_object is not None:
            count = self.train_batches * self.get_epoch_nums_passed()
            self.sacred_object.log_scalar(self.run_prefix+'_'+key, value, count)

    def _evaluate_classifier(self, x, y, mask):
        
        loss, acc = self.classifier_model.evaluate(x, y, sample_weight=mask, verbose=0)
        pred = self.classifier_model.predict(x)
        pred = np.concatenate([p[m] for p,m in zip(pred,mask)])
        true = np.concatenate([t[m] for t, m in zip(y, mask)])
        pr_curve, rec_curve, _ = precision_recall_curve(np.round(true), pred)
        pr_rec_auc = np.trapz(y=list(reversed(pr_curve)), x=list(reversed(rec_curve)))
        return loss, acc, pr_rec_auc

    def _prepare_adversarial_data(self, x, d, disconnected = False):
        adv_input = self.latent_feature_extractor.predict(x.copy()) if disconnected else x.copy()
        if self.mode == 'conditional':
            if disconnected:
                adv_input = np.concatenate([adv_input, self.classifier_model.predict(x)], -1)
            else:
                adv_input.append(self.classifier_model.predict(x))
        adv_output = np.repeat(np.expand_dims(d, 1), x[0].shape[1], axis=1) if self.time_distributed else d
        return adv_input, adv_output

    def _evaluate_adversary(self, x, d):
        adv_input, adv_output = self._prepare_adversarial_data(x, d)
        a_loss, a_acc = self.adversarial_model.evaluate(adv_input, adv_output, verbose=0)

        return a_loss, a_acc

    def _adversary_step(self, x, d, mask):
        adv_input, adv_output = self._prepare_adversarial_data(x, d, disconnected=True)

        a_loss_before, a_acc = self.adversarial_model_disconnected.train_on_batch(adv_input, adv_output)
        a_loss = a_loss_before
        if self.mode == 'normal':
            return 0, a_loss_before, a_loss

        adv_input, adv_output = self._prepare_adversarial_data(x, d, disconnected=False)
        epoch_num = self.get_epoch_nums_passed()
        n_adv_training = 0
        if self.delay_adversary_epochs == 'auto':
            while a_loss < self.y_entropy:
                a_loss, a_acc = self.adversarial_model.train_on_batch(adv_input, y=adv_output, sample_weight=mask)
                n_adv_training += 1
        elif epoch_num >= self.delay_adversary_epochs:
            self.adversarial_model.train_on_batch(adv_input, y=adv_output)
            n_adv_training += 1

        return n_adv_training, a_loss_before, a_loss

    def train_adversarial(self, x_train, y_train, d_train, w_train, mask_train, x_test, y_test, mask_test,
                          batch_size, nr_epochs=400, earlystop=100, early_stop_metric='pr_rec_auc_test', run_prefix=''):
        '''
        The training function.
        x_train, x_test: inputs
        y_train, y_test: targets
        d_train: training data (domain)
        w_train: weights for training data
        mask_train, mask_test: mask for sequence data
        batch_size: batch size
        nr_epochs: maximum number of epochs to train
        earlystop: maximum epochs before early stopping
        early_stop_metric: metric to track for early stopping
        run_prefix: prefix to use in metric logging

        '''
        self.y_entropy = stats.entropy(y_train[mask_train].flatten())
        self.train_batches = int(len(x_train[0]) / batch_size)
        self.run_prefix= run_prefix
        gen = self.train_generator(x_train, y_train, d_train, w_train, mask_train, batch_size)
        from tqdm import tqdm
        epoch_looper = tqdm(range(nr_epochs), total=nr_epochs, unit='epoch')
        early_stopper = EarlyStop(model_dir=self.model_dir,early_stop_metric=early_stop_metric,
                                  early_stop_limit=earlystop, sacred_object=self.sacred_object)
        for epoch_num in epoch_looper:
            n_adv_training_per_epoch = 0
            for batch_num in range(self.train_batches):
                x, y, d, w, m = next(gen)
                loss, acc = self.classifier_model.train_on_batch(x=x, y=y, sample_weight=w)
                n_adv_training, adv_loss_before, adv_loss_after = self._adversary_step(x=x, d=d, mask=m)
                n_adv_training_per_epoch += n_adv_training
                self.log_batch_level('batch_loss', loss, cur_batch=batch_num)
                self.log_batch_level('batch_acc', acc, cur_batch=batch_num)
                self.log_batch_level('batch_adv_loss_before_reversal', adv_loss_before, cur_batch=batch_num)
                self.log_batch_level('batch_adv_loss_after_reversal', adv_loss_before, cur_batch=batch_num)
                self.log_batch_level('batch_n_adv_training', n_adv_training, cur_batch=batch_num)

            train_loss, train_acc, pr_rec_auc_train = self._evaluate_classifier(x_train, y_train, mask=mask_train)
            test_loss, test_acc, pr_rec_auc_test = self._evaluate_classifier(x_test, y_test, mask=mask_test)
            train_a_loss, train_a_acc = self._evaluate_adversary(x=x_train, d=d_train)

            self.log_epoch_level('train_loss', train_loss), self.log_epoch_level('train_acc', train_acc)
            self.log_epoch_level('train_a_loss', train_a_loss), self.log_epoch_level('train_a_acc', train_a_acc)
            self.log_epoch_level('test_loss', test_loss), self.log_epoch_level('test_acc', test_acc)
            self.log_epoch_level('pr_rec_auc_train', pr_rec_auc_train)
            self.log_epoch_level('pr_rec_auc_test', pr_rec_auc_test)
            self.log_epoch_level('n_adv_training', n_adv_training_per_epoch)

            epoch_looper.desc = "[Test performance] %0.2f \t D PRAUC=%0.2f" % (test_loss, pr_rec_auc_test)

            if early_stopper.check_early_stop(es_value=self.history[early_stop_metric][-1],
                                              epoch_num=epoch_num,model=self.classifier_model):
                break

        self.classifier_model.load_weights(early_stopper.get_best_model())




