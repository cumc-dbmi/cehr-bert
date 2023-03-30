import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects

from keras_transformer.bert import MaskedPenalizedSparseCategoricalCrossentropy
from keras_transformer.extras import ReusableEmbedding, TiedOutputEmbedding
from utils.model_utils import create_concept_mask


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (tf.cast(mask, dtype='float32') * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits,
                                      axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def get_config(self):
        config = super().get_config()
        config['d_model'] = self.d_model
        config['num_heads'] = self.num_heads
        return config

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def split_heads_query_key_value(self, batch_size, k, q, v):
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        return k, q, v

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        k, q, v = self.split_heads_query_key_value(batch_size, k, q, v)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1,
                                              3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1,
                                       self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, *args, **kwargs):
        super(EncoderLayer, self).__init__(*args, **kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config['d_model'] = self.d_model
        config['num_heads'] = self.num_heads
        config['dff'] = self.dff
        config['rate'] = self.rate
        return config

    def call(self, x, mask, **kwargs):
        attn_output, attn_weights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=kwargs.get('training'))
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=kwargs.get('training'))
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, attn_weights


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff=2148, dropout_rate=0.1, *args,
                 **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, dropout_rate, name='transformer' + str(i))
            for i in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def get_config(self):
        config = super().get_config()
        config['num_layers'] = self.num_layers
        config['d_model'] = self.d_model
        config['num_heads'] = self.num_heads
        config['dff'] = self.dff
        config['dropout_rate'] = self.dropout_rate
        return config

    def call(self, x, mask, **kwargs):
        attention_weights = []
        for i in range(self.num_layers):
            x, attn_weights = self.enc_layers[i](x, mask, **kwargs)
            attention_weights.append(attn_weights)
        return x, tf.stack(attention_weights, axis=0)  # (batch_size, input_seq_len, d_model)


class GptDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff=2148, dropout_rate=0.1, *args,
                 **kwargs):
        super(GptDecoder, self).__init__(*args, **kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.decoder_layers = [
            GptDecoderLayer(d_model, num_heads, dff, dropout_rate, name='transformer' + str(i))
            for i in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def get_config(self):
        config = super().get_config()
        config['num_layers'] = self.num_layers
        config['d_model'] = self.d_model
        config['num_heads'] = self.num_heads
        config['dff'] = self.dff
        config['dropout_rate'] = self.dropout_rate
        return config

    def call(self, x, mask, **kwargs):
        attention_weights = []
        for i in range(self.num_layers):
            x, attn_weights = self.decoder_layers[i](x, mask, **kwargs)
            attention_weights.append(attn_weights)
        return x, tf.stack(attention_weights, axis=0)  # (batch_size, input_seq_len, d_model)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, *args, **kwargs):
        super(DecoderLayer, self).__init__(*args, **kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config['d_model'] = self.d_model
        config['num_heads'] = self.num_heads
        config['dff'] = self.dff
        config['rate'] = self.rate
        return config

    def call(self, x, enc_output, decoder_mask, encoder_mask, **kwargs):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x,
                                               decoder_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1,
                                               encoder_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, **kwargs)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, **kwargs)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class GptDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, *args, **kwargs):
        super(GptDecoderLayer, self).__init__(*args, **kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha1 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config['d_model'] = self.d_model
        config['num_heads'] = self.num_heads
        config['dff'] = self.dff
        config['rate'] = self.rate
        return config

    def call(self, x, decoder_mask, **kwargs):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x,
                                               decoder_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, **kwargs)
        out3 = self.layernorm3(ffn_output + out1)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1


class SimpleDecoderLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            d_model,
            num_heads,
            dff=512,
            rate=0.1,
            *args,
            **kwargs
    ):
        super(SimpleDecoderLayer, self).__init__(
            *args,
            **kwargs
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.multi_head_attention_layer = MultiHeadAttention(
            d_model,
            num_heads
        )
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.mha_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha_dropout_layer = tf.keras.layers.Dropout(rate)
        self.ffn_dropout_layer = tf.keras.layers.Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config['d_model'] = self.d_model
        config['num_heads'] = self.num_heads
        config['dff'] = self.dff
        config['rate'] = self.rate
        return config

    def call(
            self,
            decoder_input,
            enc_output,
            encoder_mask,
            **kwargs
    ):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn, attn_weights_block = self.multi_head_attention_layer(
            enc_output,
            enc_output,
            decoder_input,
            encoder_mask
        )  # (batch_size, target_seq_len, d_model)
        attn = self.mha_dropout_layer(attn, **kwargs)
        out2 = self.mha_layernorm(attn + decoder_input)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn_dropout_layer(ffn_output, **kwargs)
        out3 = self.ffn_layernorm(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block


class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            embedding_size,
            max_sequence_length=512,
            *args,
            **kwargs
    ):
        super(PositionalEncodingLayer, self).__init__(*args, **kwargs)
        self.embedding_size = embedding_size
        self.max_sequence_length = max_sequence_length
        # TODO: change this to dynamic in the future
        self.pos_encoding = tf.squeeze(positional_encoding(10000, self.embedding_size))

    def get_config(self):
        config = super().get_config()
        config['max_sequence_length'] = self.max_sequence_length
        config['embedding_size'] = self.embedding_size
        return config

    def call(self, visit_concept_orders):
        # Normalize the visit_orders using the smallest visit_concept_orders
        # Take the absolute value to make sure the padded values are not negative after
        # normalization
        visit_concept_orders = tf.abs(visit_concept_orders - tf.expand_dims(
            tf.math.reduce_min(visit_concept_orders, axis=1), axis=-1))
        # Get the same positional encodings for the concepts with the same visit_order
        positional_embeddings = tf.gather(self.pos_encoding, visit_concept_orders, axis=0)
        return positional_embeddings


class TimeEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size, is_time_delta=False, *args, **kwargs):
        super(TimeEmbeddingLayer, self).__init__(*args, **kwargs)
        self.embedding_size = embedding_size
        self.is_time_delta = is_time_delta
        self.w = self.add_weight(shape=(1, self.embedding_size),
                                 trainable=True,
                                 initializer=tf.keras.initializers.GlorotNormal(),
                                 name=f'time_embedding_weight_{self.name}')
        self.phi = self.add_weight(shape=(1, self.embedding_size),
                                   trainable=True,
                                   initializer=tf.keras.initializers.GlorotNormal(),
                                   name=f'time_embedding_phi_{self.name}')

    def get_config(self):
        config = super().get_config()
        config['embedding_size'] = self.embedding_size
        config['is_time_delta'] = self.is_time_delta
        return config

    def call(self, time_stamps):
        time_stamps = tf.cast(time_stamps, tf.float32)
        if self.is_time_delta:
            time_stamps = tf.concat(
                [time_stamps[:, 0:1] * 0, time_stamps[:, 1:] - time_stamps[:, :-1]], axis=-1)
        next_input = tf.expand_dims(time_stamps, axis=-1) * self.w + self.phi
        return tf.sin(next_input)


class VisitEmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, visit_order_size: int,
                 embedding_size: int, *args, **kwargs):
        super(VisitEmbeddingLayer, self).__init__(*args, **kwargs)
        self.visit_order_size = visit_order_size
        self.embedding_size = embedding_size

        self.visit_embedding_layer = tf.keras.layers.Embedding(self.visit_order_size,
                                                               self.embedding_size)

    def get_config(self):
        config = super().get_config()
        config['visit_order_size'] = self.visit_order_size
        config['embedding_size'] = self.embedding_size
        return config

    def call(self, inputs, **kwargs):
        visit_orders, concept_embeddings = inputs
        return self.visit_embedding_layer(visit_orders, **kwargs) + concept_embeddings


class ConceptValueTransformationLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size, *args, **kwargs):
        super(ConceptValueTransformationLayer, self).__init__(*args, **kwargs)
        self.embedding_size = embedding_size
        self.merge_value_transformation_layer = tf.keras.layers.Dense(
            embedding_size,
            name='merge_value_transformation_layer'
        )

    def get_config(self):
        config = super().get_config()
        config['embedding_size'] = self.embedding_size
        return config

    def call(self, concept_embeddings, concept_values, concept_value_masks):
        # Mask out the concept embeddings without a value
        # Combine the concept embeddings with concept_values

        # (batch_size, num_of_visits, num_of_concepts, 1)
        concept_values = tf.expand_dims(
            concept_values,
            axis=-1
        )
        # (batch_size, num_of_visits, num_of_concepts, 1)
        concept_value_masks = tf.expand_dims(
            concept_value_masks,
            axis=-1
        )
        # (batch_size, num_of_visits, num_of_concepts, 1 + embedding_size)
        concept_embeddings_with_val = tf.concat(
            [concept_embeddings, concept_values],
            axis=-1
        )
        # Run through a dense layer to bring the dimension back to embedding_size
        concept_embeddings_with_val = self.merge_value_transformation_layer(
            concept_embeddings_with_val
        )
        # Zero out the positions without a val
        concept_embeddings_with_val = tf.multiply(
            concept_embeddings_with_val,
            tf.cast(concept_value_masks, dtype=tf.float32)
        )
        # Derive the inverse concept value masks for zeroing out the embeddings without a val
        inverse_concept_value_masks = tf.cast(
            tf.logical_not(
                tf.cast(concept_value_masks, dtype=tf.bool)
            ),
            dtype=tf.float32
        )

        # Zero out the position of concept embeddings with a val
        concept_embeddings_without_val = tf.multiply(
            inverse_concept_value_masks,
            concept_embeddings
        )

        # Merge two sets of concept embeddings
        concept_embeddings = concept_embeddings_without_val + concept_embeddings_with_val

        return concept_embeddings


class TemporalTransformationLayer(tf.keras.layers.Layer):
    def __init__(self, time_embeddings_size, embedding_size, *args, **kwargs):
        super(TemporalTransformationLayer, self).__init__(*args, **kwargs)

        self.time_embeddings_size = time_embeddings_size
        self.embedding_size = embedding_size

        # define the time embedding layer for absolute time stamps (since 1970)
        self.time_embedding_layer = TimeEmbeddingLayer(
            embedding_size=time_embeddings_size,
            name='time_embedding_layer'
        )
        # define the age embedding layer for the age w.r.t the medical record
        self.age_embedding_layer = TimeEmbeddingLayer(
            embedding_size=time_embeddings_size,
            name='age_embedding_layer'
        )

        # define positional encoding layer for visit numbers, the visit numbers are normalized
        # by subtracting visit numbers off the first visit number
        self.positional_encoding_layer = PositionalEncodingLayer(
            embedding_size=time_embeddings_size,
            name='positional_encoding_layer'
        )
        # Temporal transformation
        self.temporal_transformation_layer = tf.keras.layers.Dense(
            embedding_size,
            activation='tanh',
            name='temporal_transformation'
        )

    def get_config(self):
        config = super().get_config()
        config['time_embeddings_size'] = self.time_embeddings_size
        config['embedding_size'] = self.embedding_size
        return config

    def call(self, concept_embeddings, pat_seq_age, pat_seq_time, visit_rank_order, **kwargs):
        _, _, num_of_concepts = pat_seq_age.shape

        pt_seq_age_embeddings = self.age_embedding_layer(
            pat_seq_age,
            **kwargs
        )
        pt_seq_time_embeddings = self.time_embedding_layer(
            pat_seq_time,
            **kwargs
        )
        visit_positional_encoding = self.positional_encoding_layer(
            visit_rank_order,
            **kwargs
        )

        visit_positional_encoding = tf.tile(
            visit_positional_encoding[:, :, tf.newaxis, :], [1, 1, num_of_concepts, 1])

        # (batch, num_of_visits, num_of_concepts, embedding_size)
        temporal_concept_embeddings = self.temporal_transformation_layer(
            tf.concat(
                [concept_embeddings,
                 pt_seq_age_embeddings,
                 pt_seq_time_embeddings,
                 visit_positional_encoding],
                axis=-1
            )
        )

        return temporal_concept_embeddings


class TimeAttention(tf.keras.layers.Layer):

    def __init__(self, vocab_size: int,
                 target_seq_len: int,
                 context_seq_len: int,
                 time_window_size: int,
                 return_logits: bool = False,
                 *args, **kwargs):
        super(TimeAttention, self).__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.target_seq_len = target_seq_len
        self.context_seq_len = context_seq_len

        # Save the half window size
        self.half_time_window_size = int(time_window_size / 2)
        # Pad one for time zero, in which the index event occurred
        self.time_window_size = self.half_time_window_size * 2 + 1
        self.return_logits = return_logits

        self.embedding_layer = tf.keras.layers.Embedding(self.vocab_size,
                                                         self.time_window_size,
                                                         embeddings_initializer=tf.keras.initializers.zeros,
                                                         name='time_attention_embedding',
                                                         trainable=kwargs.get('trainable'))
        self.softmax_layer = tf.keras.layers.Softmax()

    def get_config(self):
        config = super().get_config()
        config['vocab_size'] = self.vocab_size
        config['target_seq_len'] = self.target_seq_len
        config['context_seq_len'] = self.context_seq_len
        config['time_window_size'] = self.time_window_size
        config['return_logits'] = self.return_logits
        return config

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        :param kwargs:
        :return:
        """
        target_concepts = inputs[0]
        target_time_stamps = inputs[1]
        context_time_stamps = inputs[2]
        time_mask = inputs[3]

        # shape = (batch_size, target_seq_length, time_window_size)
        concept_time_embeddings = self.embedding_layer(target_concepts)

        # shape = (batch_size, context_seq_length, target_seq_len)
        multiplied_context_time_stamps = tf.tile(tf.expand_dims(context_time_stamps, axis=-1),
                                                 tf.constant([1, 1, self.target_seq_len]))

        # shape = (batch_size, target_seq_length, context_seq_length)
        time_delta = tf.transpose(
            multiplied_context_time_stamps - tf.expand_dims(target_time_stamps, axis=1),
            perm=[0, 2, 1])

        # Clip the time deltas to fit the time window. E.g. if the time window is 101,
        # the allowed time delta values are between -50 to 50
        time_delta_value_clipped = tf.clip_by_value(time_delta,
                                                    clip_value_min=-self.half_time_window_size,
                                                    clip_value_max=self.half_time_window_size)
        # shape = (batch_size, target_seq_length, context_seq_length, full_time_window_size)
        time_delta_one_hot = tf.one_hot(time_delta_value_clipped + self.half_time_window_size,
                                        self.time_window_size)

        # shape = (batch_size, target_seq_length, time_window_size, 1)
        concept_time_embeddings_expanded = tf.expand_dims(concept_time_embeddings, axis=-1)

        # shape = (batch_size, target_seq_length, context_seq_length)
        next_input = tf.squeeze(tf.matmul(time_delta_one_hot, concept_time_embeddings_expanded),
                                axis=-1)

        # add the mask to the scaled tensor.
        if time_mask is not None:
            next_input += (tf.cast(tf.expand_dims(time_mask, axis=1), dtype='float32') * -1e9)

        return next_input if self.return_logits else self.softmax_layer(next_input)


class TimeSelfAttention(TimeAttention):

    def __init__(self,
                 target_seq_len: int,
                 context_seq_len: int,
                 self_attention_return_logits: bool,
                 *args, **kwargs):
        assert target_seq_len == context_seq_len
        super(TimeSelfAttention, self).__init__(target_seq_len=target_seq_len,
                                                context_seq_len=context_seq_len,
                                                *args, **kwargs)
        self.self_attention_return_logits = self_attention_return_logits

    def get_config(self):
        config = super().get_config()
        config['self_attention_return_logits'] = self.self_attention_return_logits
        return config

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        :param kwargs:
        :return:
        """
        concept_ids = inputs[0]
        time_stamps = inputs[1]
        time_mask = inputs[2]

        # shape = (batch_size, seq_len, seq_len)
        self_attention_logits = super().call([concept_ids, time_stamps, time_stamps, time_mask])

        # add the mask to the scaled tensor.
        if time_mask is not None:
            self_attention_logits += (
                    tf.cast(tf.expand_dims(time_mask, axis=1), dtype='float32') * -1e9)

        return self_attention_logits if self.self_attention_return_logits else self.softmax_layer(
            self_attention_logits)


class BertLayer(tf.keras.layers.Layer):

    def __init__(self, model_path: str, *args, **kwargs):
        super(BertLayer, self).__init__(*args, **kwargs)
        bert_model = tf.keras.models.load_model(model_path, custom_objects=get_custom_objects())

        self.model_path = model_path
        self.concept_embedding_layer = bert_model.get_layer('concept_embeddings')
        self.visit_segment_layer = [layer for layer in bert_model.layers if
                                    layer.name in ['visit_embedding_layer',
                                                   'visit_segment_layer']][0]
        self.positional_encoding_layer = bert_model.get_layer('positional_encoding_layer')
        self.time_embedding_layer = bert_model.get_layer('time_embedding_layer')
        self.age_embedding_layer = bert_model.get_layer('age_embedding_layer')
        self.scale_pat_seq_layer = bert_model.get_layer('scale_pat_seq_layer')
        self.encoder_layer = bert_model.get_layer('encoder')
        #         self.conv_1d = tf.keras.layers.Conv1D(1, 1)
        self.attention_dense = tf.keras.layers.Dense(self.scale_pat_seq_layer.units,
                                                     activation='tanh')
        self.dense = tf.keras.layers.Dense(self.scale_pat_seq_layer.units, activation='tanh')

    def get_config(self):
        config = super().get_config()
        config['model_path'] = self.model_path
        return config

    def call(self, inputs, **kwargs):
        (local_concept_ids, local_visit_segments, local_visit_concept_orders,
         local_time_stamps, local_ages, local_mask) = inputs

        batch_size, max_seq_length = local_mask.get_shape().as_list()

        concept_embeddings, _ = self.concept_embedding_layer(local_concept_ids)
        time_embeddings = self.time_embedding_layer(local_time_stamps)
        age_embeddings = self.age_embedding_layer(local_ages)
        positional_encoddings = self.positional_encoding_layer(local_visit_concept_orders)
        concept_mask = create_concept_mask(local_mask, max_seq_length)

        input_for_encoder = self.scale_pat_seq_layer(
            tf.concat([concept_embeddings, time_embeddings, age_embeddings, positional_encoddings],
                      axis=-1))
        input_for_encoder = self.visit_segment_layer([local_visit_segments, input_for_encoder])
        contextualized_embeddings, _ = self.encoder_layer(input_for_encoder, concept_mask)
        _, _, embedding_size = contextualized_embeddings.get_shape().as_list()
        mask_embeddings = tf.tile(tf.expand_dims(local_mask == 0, -1), [1, 1, embedding_size])
        contextualized_embeddings = tf.math.multiply(contextualized_embeddings,
                                                     tf.cast(mask_embeddings, dtype=tf.float32))

        # (batch, seq_len, embeddings_size)
        multi_dim_att = tf.nn.softmax(self.attention_dense(contextualized_embeddings)
                                      + (tf.cast(tf.expand_dims(local_mask, axis=-1),
                                                 dtype='float32') * -1e9), axis=1)
        context_representation = tf.reduce_sum(multi_dim_att * contextualized_embeddings, axis=1)

        #         conv_output = self.conv_1d(contextualized_embeddings)
        #         conv_output += (tf.cast(tf.expand_dims(local_mask, axis=-1), dtype='float32') * -1e9)
        #         context_representation = tf.reshape(
        #             tf.transpose(tf.nn.softmax(conv_output, axis=1), [0, 2, 1]) @ contextualized_embeddings,
        #             (-1, self.conv_1d.filters * embedding_size))

        return self.dense(context_representation)


class ConvolutionBertLayer(tf.keras.layers.Layer):

    def __init__(self,
                 model_path: str,
                 seq_len: int,
                 context_window: int,
                 stride: int, *args, **kwargs):
        super(ConvolutionBertLayer, self).__init__(*args, **kwargs)
        self.model_path = model_path
        self.seq_len = seq_len
        self.context_window = context_window
        self.stride = stride
        self.step = (seq_len - context_window) // stride + 1
        self.bert_layer = BertLayer(model_path=model_path)
        #         self.conv_1d = tf.keras.layers.Conv1D(1, 1)
        self.attention_dense = tf.keras.layers.Dense(self.bert_layer.scale_pat_seq_layer.units,
                                                     activation='tanh')

        assert (self.step - 1) * self.stride + self.context_window == self.seq_len

    def get_config(self):
        config = super().get_config()
        config['model_path'] = self.model_path
        config['seq_len'] = self.seq_len
        config['context_window'] = self.context_window
        config['stride'] = self.stride
        return config

    def call(self, inputs, **kwargs):
        concept_ids, visit_segments, visit_concept_orders, time_stamps, ages, mask = inputs

        bert_outputs = []
        bert_output_masking = []
        for i in range(self.step):
            start_index = i * self.stride
            end_index = i * self.stride + self.context_window

            concept_ids_step = concept_ids[:, start_index:end_index]
            visit_segments_step = visit_segments[:, start_index:end_index]
            time_stamps_step = time_stamps[:, start_index:end_index]
            ages_step = ages[:, start_index:end_index]
            visit_concept_orders_step = visit_concept_orders[:, start_index:end_index]
            mask_step = mask[:, start_index:end_index]

            inputs_step = [concept_ids_step,
                           visit_segments_step,
                           visit_concept_orders_step,
                           time_stamps_step,
                           ages_step,
                           mask_step]

            output_masking = tf.cast(tf.reduce_all(mask_step == 1, axis=-1), dtype=tf.int32)

            output_step = self.bert_layer(inputs_step)
            bert_outputs.append(output_step)
            bert_output_masking.append(output_masking)

        # (batch, step, embedding_size)
        bert_output_tensor = tf.stack(bert_outputs, axis=1)
        # (batch, step)
        bert_output_masking_tensor = tf.stack(bert_output_masking, axis=1)
        # (batch, step, 1)
        #         conv_output = self.conv_1d(bert_output_tensor)

        attn = self.attention_dense(bert_output_tensor)

        attn += (tf.cast(tf.expand_dims(bert_output_masking_tensor, axis=-1),
                         dtype='float32') * -1e9)

        _, _, embedding_size = bert_output_tensor.get_shape().as_list()

        context_representation = tf.reduce_sum(tf.nn.softmax(attn, axis=1) * bert_output_tensor,
                                               axis=1)

        #         context_representation = tf.reshape(
        #             tf.transpose(tf.nn.softmax(conv_output, axis=1), [0, 2, 1]) @ bert_output_tensor,
        #             (-1, self.conv_1d.filters * embedding_size))

        return context_representation


class HiddenPhenotypeLayer(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_unit: int,
                 embedding_size: int,
                 num_heads: int,
                 dropout_rate: float = 0.1,
                 *args, **kwargs):
        super(HiddenPhenotypeLayer, self).__init__(*args, **kwargs)
        self.hidden_unit = hidden_unit
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # num_hidden_state, embedding_size
        self.hidden_unit_embedding = self.add_weight(
            shape=(hidden_unit, embedding_size),
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            name='phenotype_embeddings'
        )

        self.mha_layer = MultiHeadAttention(
            d_model=embedding_size,
            num_heads=num_heads
        )

        self.layer_norm_layer = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)

        self.phenotype_hidden_state_layer = tf.keras.layers.Dense(
            units=1
        )

    def get_config(self):
        config = super().get_config()
        config['hidden_unit'] = self.hidden_unit
        config['embedding_size'] = self.embedding_size
        config['num_heads'] = self.num_heads
        config['dropout_rate'] = self.dropout_rate
        return config

    def call(self, inputs, **kwargs):
        seq_embeddings, mask = inputs
        # Use broadcasting to copy hidden_unit_embedding
        # (batch_size, num_hidden_state, embedding_size)
        expanded_phenotype_embeddings = tf.ones_like(
            seq_embeddings
        )[:, 0:1, 0:1] * self.hidden_unit_embedding[tf.newaxis, :, :]

        # (batch_size, num_hidden_state, embedding_size)
        context_phenotype_embeddings, _ = self.mha_layer(
            seq_embeddings,
            seq_embeddings,
            expanded_phenotype_embeddings,
            mask,
        )

        context_phenotype_embeddings = self.dropout_layer(
            context_phenotype_embeddings,
            **kwargs
        )

        context_phenotype_embeddings = self.layer_norm_layer(
            expanded_phenotype_embeddings + context_phenotype_embeddings,
            **kwargs
        )

        # (batch_size, num_hidden_state)
        phenotype_probability_dist = tf.nn.softmax(
            tf.squeeze(
                self.phenotype_hidden_state_layer(
                    context_phenotype_embeddings
                )
            )
        )

        phenotype_prob_entropy = -tf.reduce_sum(
            phenotype_probability_dist * tf.math.log(phenotype_probability_dist),
            axis=-1
        )
        # self.add_loss(
        #     tf.reduce_mean(
        #         phenotype_prob_entropy
        #     )
        # )

        self.add_metric(
            phenotype_prob_entropy,
            name='phenotype_probability_entropy'
        )

        return context_phenotype_embeddings, phenotype_probability_dist


class VisitPhenotypeLayer(tf.keras.layers.Layer):

    def __init__(
            self,
            num_of_phenotypes: int,
            num_of_phenotype_neighbors: int,
            num_of_concept_neighbors: int,
            embedding_size: int,
            transformer_dropout: float,
            dff: int = 2148,
            phenotype_entropy_weight: float = 2e-05,
            phenotype_euclidean_weight: float = 2e-05,
            phenotype_concept_distance_weight: float = 1e-04,
            *args, **kwargs
    ):
        super(VisitPhenotypeLayer, self).__init__(*args, **kwargs)
        self.num_of_phenotypes = num_of_phenotypes
        self.embedding_size = embedding_size
        self.transformer_dropout = transformer_dropout
        self.dff = dff
        self.num_of_concept_neighbors = num_of_concept_neighbors
        self.num_of_phenotype_neighbors = num_of_phenotype_neighbors
        self.phenotype_entropy_weight = phenotype_entropy_weight
        self.phenotype_euclidean_weight = phenotype_euclidean_weight
        self.phenotype_concept_distance_weight = phenotype_concept_distance_weight

        # We assume there exists hidden phenotype embeddings
        # (num_of_phenotypes, embedding_size)
        self.phenotype_embeddings = self.add_weight(
            shape=(num_of_phenotypes, embedding_size),
            initializer=tf.keras.initializers.GlorotNormal(seed=0),
            trainable=True,
            name='phenotype_embeddings_matrix'
        )

        self.ffn = point_wise_feed_forward_network(
            embedding_size,
            dff
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(transformer_dropout)
        self.dropout2 = tf.keras.layers.Dropout(transformer_dropout)

    def get_config(self):
        config = super().get_config()
        config['num_of_phenotypes'] = self.num_of_phenotypes
        config['embedding_size'] = self.embedding_size
        config['transformer_dropout'] = self.transformer_dropout
        config['dff'] = self.dff
        config['num_of_concept_neighbors'] = self.num_of_concept_neighbors
        config['num_of_phenotype_neighbors'] = self.num_of_phenotype_neighbors
        config['phenotype_entropy_weight'] = self.phenotype_entropy_weight
        config['phenotype_euclidean_weight'] = self.phenotype_euclidean_weight
        config['phenotype_concept_distance_weight'] = self.phenotype_concept_distance_weight
        return config

    def call(self, inputs, **kwargs):
        visit_embeddings, visit_mask, embedding_matrix = inputs

        # Do not compute the entropy for the masked visits
        converted_visit_mask = tf.cast(
            tf.logical_not(
                tf.cast(
                    visit_mask,
                    dtype=tf.bool
                )
            ),
            dtype=tf.float32
        )[:, :, tf.newaxis]

        # (batch_size, num_of_visits, num_of_phenotypes)
        visit_phenotype_probs = tf.nn.softmax(
            visit_embeddings @ tf.transpose(
                self.phenotype_embeddings,
                [1, 0]
            ) * converted_visit_mask
        )

        # calculate phenotype concept distance matrix (num_of_phenotypes, top_k)
        phenotype_concept_dist = tf.reduce_mean(
            -tf.math.top_k(
                -distance_matrix(
                    self.phenotype_embeddings,
                    embedding_matrix
                ),
                k=self.num_of_concept_neighbors
            ).values
        )

        self.add_metric(
            phenotype_concept_dist,
            name='phenotype_concept_dist'
        )

        # Calculate the probability distribution entropy
        phenotype_prob_entropy = -tf.reduce_sum(
            visit_phenotype_probs * tf.math.log(visit_phenotype_probs) * converted_visit_mask,
            axis=-1
        )
        # Add the entropy to the model metrics
        self.add_metric(
            phenotype_prob_entropy,
            name='phenotype_probability_entropy'
        )

        # Add the entropy as a loss to encourage the model to focus on a subset of phenotypes
        # self.add_loss(
        #     tf.reduce_mean(phenotype_prob_entropy) * self.phenotype_entropy_weight,
        # )

        # Get phenotype pairwise distance metrics
        phe_inv_loss, phe_dist_metric, phe_dist_var = self.get_inverse_phenotype_dist_loss_metric()
        #
        # self.add_loss(
        #     phe_inv_loss * self.phenotype_euclidean_weight
        # )

        self.add_metric(
            phe_dist_metric,
            name='phenotype_euclidean_distance'
        )

        self.add_metric(
            phe_dist_var,
            name='phenotype_euclidean_variance'
        )

        # Calculate the contextualized visit embeddings using the pre-defined phenotype embeddings
        # (batch_size, num_of_visits, embedding_size)
        contextualized_phenotype_embeddings = self.dropout1(
            visit_phenotype_probs @ self.phenotype_embeddings,
            training=kwargs.get('training')
        )

        out1 = self.layernorm1(
            visit_embeddings + contextualized_phenotype_embeddings
        )

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=kwargs.get('training'))
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, visit_phenotype_probs

    def get_inverse_phenotype_dist_loss_metric(self):
        r = tf.reduce_sum(self.phenotype_embeddings * self.phenotype_embeddings, 1)
        # turn r into column vector
        r = tf.reshape(r, [-1, 1])
        euclidean_distances_full = r - 2 * tf.matmul(self.phenotype_embeddings, tf.transpose(
            self.phenotype_embeddings)) + tf.transpose(r)

        euclidean_distances = -tf.math.top_k(
            -euclidean_distances_full,
            k=self.num_of_phenotype_neighbors
        ).values

        inv_loss = tf.reduce_mean(
            tf.math.exp(-euclidean_distances)
        )

        var_loss = tf.math.reduce_variance(
            euclidean_distances
        )

        dist_metric = tf.reduce_mean(
            euclidean_distances
        )

        return inv_loss, dist_metric, var_loss


def distance_matrix(matrix_1, matrix_2):
    m = matrix_1.shape[0]
    n = matrix_2.shape[0]

    assert matrix_1.shape[1] == matrix_2.shape[1], f"The number of components for vectors in A \
            {matrix_1.shape[1]} does not match that of B {matrix_2.shape[1]}!"

    matrix_1_dots = tf.reshape(tf.reduce_sum(matrix_1 * matrix_1, axis=1), (m, 1)) * tf.ones((1, n))
    matrix_2_dots = tf.reduce_sum(matrix_2 * matrix_2, axis=1) * tf.ones((m, 1))

    matrix_distance_squared = matrix_1_dots + matrix_2_dots - 2 * matrix_1 @ tf.transpose(matrix_2)

    return tf.sqrt(matrix_distance_squared)


get_custom_objects().update({
    'MultiHeadAttention': MultiHeadAttention,
    'Encoder': Encoder,
    'GptDecoder': GptDecoder,
    'EncoderLayer': EncoderLayer,
    'DecoderLayer': DecoderLayer,
    'GptDecoderLayer': GptDecoderLayer,
    'SimpleDecoderLayer': SimpleDecoderLayer,
    'TimeAttention': TimeAttention,
    'TimeSelfAttention': TimeSelfAttention,
    'PairwiseTimeAttention': TimeSelfAttention,
    'VisitEmbeddingLayer': VisitEmbeddingLayer,
    'PositionalEncodingLayer': PositionalEncodingLayer,
    'TimeEmbeddingLayer': TimeEmbeddingLayer,
    'TemporalTransformationLayer': TemporalTransformationLayer,
    'ConceptValueTransformationLayer': ConceptValueTransformationLayer,
    'ReusableEmbedding': ReusableEmbedding,
    'TiedOutputEmbedding': TiedOutputEmbedding,
    'MaskedPenalizedSparseCategoricalCrossentropy': MaskedPenalizedSparseCategoricalCrossentropy,
    'BertLayer': BertLayer,
    'ConvolutionBertLayer': ConvolutionBertLayer,
    'HiddenPhenotypeLayer': HiddenPhenotypeLayer,
    'VisitPhenotypeLayer': VisitPhenotypeLayer,
    'ConvolutionBertLayer': ConvolutionBertLayer
})