from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import os
import pickle
from absl import flags, logging

import bert_init
import metrics
from attention import MutilHeadAttention
from bert import modeling
import cg_optimization as optimization
from bert import tokenization
import tensorflow as tf

tf.disable_v2_behavior()
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", './bert_model/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", './bert_model/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", './bert_model/bert_model.ckpt',
    "Initial checkpoint (usually from a pre-trained BERT model).")

# if you download cased checkpoint you should use "False",if uncased you should use
# "True"
# if we used in bio-medical field，don't do lower case would be better!

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", True,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_string("middle_output", "middle_data", "Dir was used to store middle data!")
flags.DEFINE_bool("crf", True, "use crf!")

G_VARIABLE_SCOPE = 'generator_scope'
D_VARIABLE_SCOPE = 'discriminator_scope'


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 mask,
                 segment_ids,
                 label_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.mask = mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file_path):
        """Read a BIO data!"""
        print('input_file: ', input_file_path)
        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            rf = input_file.readlines()
            result = []
            for line in rf:
                line = line.strip()
                data = line.split('\t')
                if len(data) >= 2:
                    sentence = data[0]
                    labels = data[1]
                    result.append((labels, sentence))
        return result


class SLProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test"
        )

    def get_labels(self):
        """
        here "X" used to represent "##eer","##soo" and so on!
        "[PAD]" for padding
        :return:
        """
        return ["[PAD]", "E-GPE.NAM", "S-PER.NOM", "S-GPE.NAM", "M-ORG.NAM", "S-LOC.NOM", "M-LOC.NAM", "S-PER.NAM",
                "E-ORG.NAM", "M-PER.NAM", "E-PER.NAM", "B-GPE.NAM", "M-PER.NOM", "O", "B-ORG.NAM", "B-LOC.NOM",
                "E-LOC.NOM", "M-GPE.NAM", "B-LOC.NAM", "E-ORG.NOM", "E-LOC.NAM", "M-ORG.NOM", "B-ORG.NOM", "M-LOC.NOM",
                "E-PER.NOM", "B-GPE.NOM", "B-PER.NOM", "E-GPE.NOM", "B-PER.NAM", "X", "[CLS]",
                "[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            texts = tokenization.convert_to_unicode(line[1])
            labels = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=texts, label=labels))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    """
    :param ex_index: example num
    :param example:
    :param label_list: all labels
    :param max_seq_length:
    :param tokenizer: WordPiece tokenization
    :param mode:
    :return: feature
    IN this part we should rebuild input sentences to the following format.
    example:[Jim,Hen,##son,was,a,puppet,##eer]
    labels: [I-PER,I-PER,X,O,O,O,X]
    """
    label_map = {}
    # here start with zero this means that "[PAD]" is zero
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    with open(FLAGS.middle_output + "/label2id.pkl", 'wb') as w:
        pickle.dump(label_map, w)
    textlist = list(example.text)
    labellist = example.label.split('|')
    tokens = []
    labels = []
    for i, (word, label) in enumerate(zip(textlist, labellist)):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for i, _ in enumerate(token):
            if i == 0:
                labels.append(label)
            else:
                labels.append("X")
    # only Account for [CLS] with "- 1".
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 1)]
        labels = labels[0:(max_seq_length - 1)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    # after that we don't add "[SEP]" because we want a sentence don't have
    # stop tag, because i think its not very necessary.
    # or if add "[SEP]" the model even will cause problem, special the crf layer was used.
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    mask = [1] * len(input_ids)
    # use zero to padding and you should
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("[PAD]")
    assert len(input_ids) == max_seq_length
    assert len(mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(ntokens) == max_seq_length
    if ex_index < 3:
        logging.info("*** Example ***")
        logging.info("guid: %s" % example.guid)
        logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
    feature = InputFeatures(
        input_ids=input_ids,
        mask=mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
    )
    # we need ntokens because if we do predict it can help us return to original token.
    return feature, ntokens, label_ids


def filed_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file, mode=None):
    writer = tf.python_io.TFRecordWriter(output_file)
    batch_tokens = []
    batch_labels = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature, ntokens, label_ids = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer,
                                                             mode)
        batch_tokens.append(ntokens)
        batch_labels.append(label_ids)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["mask"] = create_int_feature(feature.mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    # sentence token in each batch
    writer.close()
    return batch_tokens, batch_labels


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),

    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d

    return input_fn


# all above are related to data preprocess
# Following i about the model

def hidden2tag(hiddenlayer, numclass):
    linear = tf.keras.layers.Dense(numclass, activation=None)
    return linear(hiddenlayer)


def crf_loss(logits, labels, mask, num_labels, mask2len):
    """
    :param logits:
    :param labels:
    :param mask2len:each sample's length
    :return:
    """
    # TODO
    with tf.variable_scope("crf_loss"):
        trans = tf.compat.v1.get_variable(
            "transition",
            shape=[num_labels, num_labels],
            initializer=tf.contrib.layers.xavier_initializer()
        )

    log_likelihood, transition = tf.contrib.crf.crf_log_likelihood(logits, labels, transition_params=trans,
                                                                   sequence_lengths=mask2len)
    loss = tf.math.reduce_mean(-log_likelihood)

    return loss, transition


def softmax_layer(logits, labels, num_labels, mask):
    logits = tf.reshape(logits, [-1, num_labels])
    labels = tf.reshape(labels, [-1])
    mask = tf.cast(mask, dtype=tf.float32)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=one_hot_labels)
    loss *= tf.reshape(mask, [-1])
    loss = tf.reduce_sum(loss)
    total_size = tf.reduce_sum(mask)
    total_size += 1e-12  # to avoid division by 0 for all-0 weights
    loss /= total_size
    # predict not mask we could filtered it in the prediction part.
    probabilities = tf.math.softmax(logits, axis=-1)
    predict = tf.math.argmax(probabilities, axis=-1)
    return loss, predict


def get_generator(bert_config, is_training, input_ids, mask,
                  segment_ids, labels, num_labels, use_one_hot_embeddings):
    with tf.variable_scope(G_VARIABLE_SCOPE, reuse=False):
        generator_model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings
        )

        output_layer = generator_model.get_sequence_output()
        _shape = modeling.get_shape_list(output_layer)

        # output_layer shape is
        if is_training:
            output_layer = tf.keras.layers.Dropout(rate=0.1)(output_layer)

        # to labels
        logits = tf.layers.dense(output_layer, num_labels, activation=None)
        # TODO test shape
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])

        if FLAGS.crf:
            mask2len = tf.reduce_sum(mask, axis=1)
            loss, trans = crf_loss(logits, labels, mask, num_labels, mask2len)
            predict, viterbi_score = tf.contrib.crf.crf_decode(logits, trans, mask2len)
            return generator_model, loss, logits, predict
        else:
            loss, predict = softmax_layer(logits, labels, num_labels, mask)
            return generator_model, loss, logits, predict


def get_discriminator(bert_config, is_training, input_ids, mask,
                      segment_ids, num_labels_g, num_labels_d, use_one_hot_embeddings, logits_g, fake_labels,
                      reuse=False):
    with tf.compat.v1.variable_scope(D_VARIABLE_SCOPE, reuse=reuse):
        discriminator_model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings
        )
        output_layer = discriminator_model.get_sequence_output()

        if is_training:
            output_layer = tf.keras.layers.Dropout(rate=0.1)(output_layer)

        logits_d = tf.layers.dense(output_layer, num_labels_g, activation=None)
        # attention with logits_g
        attention = MutilHeadAttention(d_model=num_labels_g, num_heads=1)
        output, _ = attention(logits_g, k=logits_g, q=logits_d, mask=None)

        logits_d = tf.layers.dense(output, num_labels_d, activation=None)
        logits_d = tf.reshape(logits_d, [-1, FLAGS.max_seq_length, num_labels_d])

        loss, predict = softmax_layer(logits_d, fake_labels, num_labels_d, mask)
        return discriminator_model, loss, num_labels_d, predict


def _get_fake_data(labels, predict_g):
    """Sample from the generator to create corrupted input."""
    fake_labels = (1 - tf.cast(
        tf.equal(labels, predict_g), tf.int32))
    fake_labels = tf.cast(fake_labels, tf.int32)
    return fake_labels


def create_model(bert_config, is_training, input_ids, mask,
                 segment_ids, labels, num_labels_g, use_one_hot_embeddings):
    loss_weight = tf.Variable(8, name='loss_weight', dtype=tf.float32)
    generator_model, loss_g, logits_g, predict_g = get_generator(bert_config, is_training, input_ids, mask,
                                                                 segment_ids, labels, num_labels_g,
                                                                 use_one_hot_embeddings)

    if is_training:
        fake_labels = _get_fake_data(labels, predict_g)
        real_labels = _get_fake_data(labels, labels)

        bert_config_d = copy.deepcopy(bert_config)
        bert_config_d.num_hidden_layers = 3
        discriminator_model, loss_d_fake, num_labels_d, _ = get_discriminator(bert_config_d, is_training, input_ids, mask,
                                                                              segment_ids, num_labels_g, 2,
                                                                              use_one_hot_embeddings, logits_g,
                                                                              fake_labels, reuse=False)

        logits_real = tf.one_hot(labels, depth=num_labels_g, dtype=tf.float32)
        _, loss_d_real, _, _ = get_discriminator(bert_config_d, is_training, input_ids, mask,
                                                 segment_ids, num_labels_g, num_labels_d,
                                                 use_one_hot_embeddings, logits_real,
                                                 real_labels, reuse=True)

        loss_d = loss_d_fake + loss_d_real
        loss = loss_g + loss_d_fake * loss_weight
        return loss, logits_g, predict_g, loss_weight, loss_g, loss_d
    else:
        return loss_g, logits_g, predict_g, None, loss_g, None


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        mask = features["mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        loss_total, logits, predicts, loss_weight, loss_g, loss_d = create_model(bert_config, is_training,
                                                                                 input_ids,
                                                                                 mask, segment_ids, label_ids,
                                                                                 num_labels,
                                                                                 use_one_hot_embeddings)
        all_vars = tf.trainable_variables()

        d_vars = [v for v in all_vars if D_VARIABLE_SCOPE in v.name]
        g_vars = [v for v in all_vars if v not in d_vars]

        scaffold_fn = None
        initialized_variable_names_g = None
        initialized_variable_names_d = None
        if init_checkpoint:
            (assignment_map_g, initialized_variable_names_g) = bert_init.get_assignment_map_from_checkpoint(g_vars,
                                                                                                            init_checkpoint,
                                                                                                            G_VARIABLE_SCOPE)
            (assignment_map_d, initialized_variable_names_d) = bert_init.get_assignment_map_from_checkpoint(d_vars,
                                                                                                            init_checkpoint,
                                                                                                            D_VARIABLE_SCOPE)

            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map_g)
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map_d)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map_g)
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map_d)

        logging.info("**** Trainable Variables ****")
        for var in all_vars:
            init_string = ""
            if var.name in initialized_variable_names_g:
                init_string = ", *INIT_FROM_CKPT_G*"
            elif var.name in initialized_variable_names_d:
                init_string = ", *INIT_FROM_CKPT_D*"
            logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                         init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps,
            #                                          use_tpu)

            d_train_op = optimization.create_optimizer("d", d_vars,
                                                       loss_d, learning_rate, num_train_steps, 0,
                                                       use_tpu)

            g_train_op = optimization.create_optimizer("g", g_vars,
                                                       loss_total, learning_rate, num_train_steps, num_warmup_steps,
                                                       use_tpu)

            logging_hook = tf.train.LoggingTensorHook(
                {"d_loss": loss_d, "g_loss": loss_g, "loss_weight": loss_weight}, every_n_iter=1)
            # {"d_loss": loss_d, "g_loss": loss_g}, every_n_iter=1)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=loss_d + loss_total,
                train_op=tf.group(d_train_op, g_train_op),
                training_hooks=[logging_hook],
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
                # def metric_fn(label_ids, logits, num_labels, mask):
                #     predictions = tf.math.argmax(logits, axis=-1, output_type=tf.int32)
                #     cm = metrics.streaming_confusion_matrix(label_ids, predictions, num_labels - 1, weights=mask)
                #     return {
                #         "confusion_matrix": cm
                #     }
                #     #
                #
                # eval_metrics = (metric_fn, [label_ids, logits, num_labels, mask])
                # output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                #     mode=mode,
                #     loss=total_loss,
                #     eval_metrics=eval_metrics,
                #     scaffold_fn=scaffold_fn)
                pass
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predicts, scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn


def _write_base(batch_tokens, id2label, prediction, batch_labels, wf, i):
    token = batch_tokens[i]
    predict = id2label[prediction]
    true_l = id2label[batch_labels[i]]
    if token != "[PAD]" and token != "[CLS]" and true_l != "X":
        #
        if predict == "X" and not predict.startswith("##"):
            predict = "O"
        line = "{}\t{}\t{}\n".format(token, true_l, predict)
        wf.write(line)


def Writer(output_predict_file, result, batch_tokens, batch_labels, id2label):
    with open(output_predict_file, 'w') as wf:

        if FLAGS.crf:
            predictions = []
            for m, pred in enumerate(result):
                predictions.extend(pred)

            i = -1
            for idx in range(len(batch_tokens)):
                content = ''
                label = ''
                pred = ''
                for token_idx in range(len(batch_tokens[idx])):
                    i += 1
                    prediction = predictions[i]
                    token = batch_tokens[idx][token_idx]
                    predict = id2label[prediction]
                    true_l = id2label[batch_labels[idx][token_idx]]

                    if token != "[PAD]" and token != "[CLS]" and token != "[UNK]" and true_l != "X":
                        #
                        if predict == "X" and not predict.startswith("##"):
                            predict = "O"
                        content += token
                        pred += predict + ' '
                        label += true_l + ' '

                pred = ':'.join(pred.split())
                label = ':'.join(label.strip().split())

                line = "{}\t{}\t{}\n".format(content, label, pred)
                wf.write(line)


def main(_):
    logging.set_verbosity(logging.INFO)
    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))
    processor = SLProcessor()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)

        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        if not os.path.exists(train_file):
            _, _ = filed_based_convert_examples_to_features(
                train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_examples))
        logging.info("  Batch size = %d", FLAGS.train_batch_size)
        logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        batch_tokens, batch_labels = filed_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        logging.info("***** Running evaluation *****")
        logging.info("  Num examples = %d", len(eval_examples))
        logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        # if FLAGS.use_tpu:
        #     eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
        # eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)
        result = estimator.evaluate(input_fn=eval_input_fn)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as wf:
            logging.info("***** Eval results *****")
            confusion_matrix = result["confusion_matrix"]
            p, r, f = metrics.calculate(confusion_matrix, len(label_list) - 1)
            logging.info("***********************************************")
            logging.info("********************P = %s*********************", str(p))
            logging.info("********************R = %s*********************", str(r))
            logging.info("********************F = %s*********************", str(f))
            logging.info("***********************************************")

    if FLAGS.do_predict:
        with open(FLAGS.middle_output + '/label2id.pkl', 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

        predict_examples = processor.get_test_examples(FLAGS.data_dir)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        batch_tokens, batch_labels = filed_based_convert_examples_to_features(predict_examples, label_list,
                                                                              FLAGS.max_seq_length, tokenizer,
                                                                              predict_file)

        logging.info("***** Running prediction*****")
        logging.info("  Num examples = %d", len(predict_examples))
        logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")
        # here if the tag is "X" means it belong to its before token, here for convenient evaluate use
        # conlleval.pl we  discarding it directly
        Writer(output_predict_file, result, batch_tokens, batch_labels, id2label)


if __name__ == "__main__":
    FLAGS.data_dir = './data/ner/weibo'
    FLAGS.output_dir = './gan_output'
    FLAGS.learning_rate = 2e-5
    FLAGS.num_train_epochs = 10
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
