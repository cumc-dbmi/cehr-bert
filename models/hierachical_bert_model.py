from models.custom_layers import *
import numpy as np


def transformer_hierarchical_bert_model(num_of_visits,
                                        num_of_concepts,
                                        concept_vocab_size,
                                        embedding_size,
                                        depth: int,
                                        num_heads: int,
                                        transformer_dropout: float = 0.1,
                                        embedding_dropout: float = 0.6,
                                        l2_reg_penalty: float = 1e-4,
                                        time_embeddings_size: int = 16,
                                        include_second_tiered_learning_objectives: bool = False,
                                        visit_vocab_size: int = None):
    """
    Create a hierarchical bert model

    :param num_of_visits:
    :param num_of_concepts:
    :param concept_vocab_size:
    :param embedding_size:
    :param depth:
    :param num_heads:
    :param transformer_dropout:
    :param embedding_dropout:
    :param l2_reg_penalty:
    :param time_embeddings_size:
    :param include_second_tiered_learning_objectives:
    :param visit_vocab_size:
    :return:
    """
    # If the second tiered learning objectives are enabled, visit_vocab_size needs to be provided
    if include_second_tiered_learning_objectives and not visit_vocab_size:
        raise RuntimeError(f'visit_vocab_size can not be null '
                           f'when the second learning objectives are enabled')

    pat_seq = tf.keras.layers.Input(shape=(num_of_visits, num_of_concepts,), dtype='int32',
                                    name='pat_seq')
    pat_seq_age = tf.keras.layers.Input(shape=(num_of_visits, num_of_concepts,), dtype='int32',
                                        name='pat_seq_age')
    pat_seq_time = tf.keras.layers.Input(shape=(num_of_visits, num_of_concepts,), dtype='int32',
                                         name='pat_seq_time')
    pat_mask = tf.keras.layers.Input(shape=(num_of_visits, num_of_concepts,), dtype='int32',
                                     name='pat_mask')

    visit_segment = tf.keras.layers.Input(shape=(num_of_visits,), dtype='int32',
                                          name='visit_segment')

    visit_time_delta_att = tf.keras.layers.Input(shape=(num_of_visits - 1,), dtype='int32',
                                                 name='visit_time_delta_att')
    visit_mask = tf.keras.layers.Input(shape=(num_of_visits,), dtype='int32', name='visit_mask')

    default_inputs = [pat_seq, pat_seq_age, pat_seq_time, pat_mask,
                      visit_segment, visit_time_delta_att, visit_mask]

    pat_concept_mask = tf.reshape(pat_mask, (-1, num_of_concepts))[:, tf.newaxis, tf.newaxis, :]

    visit_mask_with_att = tf.reshape(tf.stack([visit_mask, visit_mask], axis=2),
                                     (-1, num_of_visits * 2))[:, 1:]

    visit_concept_mask = visit_mask_with_att[:, tf.newaxis, tf.newaxis, :]

    # output the embedding_matrix:
    l2_regularizer = (tf.keras.regularizers.l2(l2_reg_penalty) if l2_reg_penalty else None)
    concept_embedding_layer = ReusableEmbedding(
        concept_vocab_size,
        embedding_size,
        name='bpe_embeddings',
        embeddings_regularizer=l2_regularizer
    )

    # define the visit segment layer
    visit_segment_layer = VisitEmbeddingLayer(visit_order_size=3,
                                              embedding_size=embedding_size,
                                              name='visit_segment_layer')

    # define the time embedding layer for absolute time stamps (since 1970)
    time_embedding_layer = TimeEmbeddingLayer(embedding_size=time_embeddings_size,
                                              name='time_embedding_layer')
    # define the age embedding layer for the age w.r.t the medical record
    age_embedding_layer = TimeEmbeddingLayer(embedding_size=time_embeddings_size,
                                             name='age_embedding_layer')

    temporal_transformation_layer = tf.keras.layers.Dense(embedding_size,
                                                          activation='tanh',
                                                          name='temporal_transformation')

    pt_seq_concept_embeddings, embedding_matrix = concept_embedding_layer(pat_seq)
    pt_seq_age_embeddings = age_embedding_layer(pat_seq_age)
    pt_seq_time_embeddings = time_embedding_layer(pat_seq_time)

    # dense layer for rescale the patient sequence embeddings back to the original size
    pt_seq_concept_embeddings = visit_segment_layer([visit_segment[:, :, tf.newaxis],
                                                     pt_seq_concept_embeddings])

    temporal_concept_embeddings = temporal_transformation_layer(
        tf.concat([pt_seq_concept_embeddings, pt_seq_age_embeddings, pt_seq_time_embeddings],
                  axis=-1, name='concat_for_encoder'))

    temporal_concept_embeddings = tf.reshape(temporal_concept_embeddings,
                                             (-1, num_of_concepts, embedding_size))

    # The first bert applied at the visit level
    concept_encoder = Encoder(name='concept_encoder',
                              num_layers=depth,
                              d_model=embedding_size,
                              num_heads=num_heads,
                              dropout_rate=transformer_dropout)

    contextualized_concept_embeddings, _ = concept_encoder(
        temporal_concept_embeddings,
        pat_concept_mask
    )

    contextualized_concept_embeddings = tf.reshape(
        contextualized_concept_embeddings,
        shape=(-1, num_of_visits, num_of_concepts, embedding_size)
    )

    # Slice out the first contextualized embedding of each visit
    visit_embeddings = contextualized_concept_embeddings[:, :, 0]

    # Reshape the data in visit view back to patient view: (batch, sequence, embedding_size)
    contextualized_concept_embeddings = tf.reshape(
        contextualized_concept_embeddings,
        shape=(-1, num_of_visits * num_of_concepts, embedding_size)
    )

    # Insert the att embeddings between the visit embeddings using the following trick
    identity = tf.constant(
        np.insert(
            np.identity(num_of_visits),
            obj=range(1, num_of_visits),
            values=0,
            axis=1
        ),
        dtype=tf.float32
    )
    expanded_visit_embeddings = tf.transpose(
        tf.transpose(visit_embeddings, perm=[0, 2, 1]) @ identity,
        perm=[0, 2, 1]
    )

    # Look up the embeddings for the att tokens
    att_embeddings, _ = concept_embedding_layer(visit_time_delta_att)
    # Create the inverse "identity" matrix for inserting att embeddings
    identity_inverse = tf.constant(
        np.insert(
            np.identity(num_of_visits - 1),
            obj=range(0, num_of_visits),
            values=0,
            axis=1),
        dtype=tf.float32)

    expanded_att_embeddings = tf.transpose(
        tf.transpose(att_embeddings, perm=[0, 2, 1]) @ identity_inverse,
        perm=[0, 2, 1]
    )

    # Insert the att embeddings between visit embedidngs
    augmented_visit_embeddings = expanded_visit_embeddings + expanded_att_embeddings

    # Second bert applied at the patient level to the visit embeddings
    visit_encoder = Encoder(name='visit_encoder',
                            num_layers=depth,
                            d_model=embedding_size,
                            num_heads=num_heads,
                            dropout_rate=transformer_dropout)

    # Feed augmented visit embeddings into encoders to get contextualized visit embeddings
    contextualized_visit_embeddings, _ = visit_encoder(
        augmented_visit_embeddings,
        visit_concept_mask
    )

    # Use a multi head attention layer to generate the global concept embeddings by attending to
    # the visit embeddings
    decoder_layer = DecoderLayer(d_model=embedding_size, num_heads=num_heads, dff=512)

    global_concept_embeddings, _, _ = decoder_layer(
        contextualized_concept_embeddings,
        contextualized_visit_embeddings,
        tf.reshape(pat_mask, (-1, num_of_visits * num_of_concepts))[:, :, tf.newaxis],
        visit_concept_mask
    )

    # multi_head_attention_layer = MultiHeadAttention(embedding_size, num_heads)
    # global_concept_embeddings, _ = multi_head_attention_layer(
    #     contextualized_visit_embeddings,
    #     contextualized_visit_embeddings,
    #     contextualized_concept_embeddings,
    #     visit_concept_mask,
    #     None)

    # global_concept_embeddings += contextualized_concept_embeddings

    concept_output_layer = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name='concept_prediction_logits')

    contextualized_visit_embeddings_without_att = identity @ contextualized_visit_embeddings

    concept_softmax_layer = tf.keras.layers.Softmax(name='concept_predictions')

    concept_predictions = concept_softmax_layer(
        concept_output_layer([global_concept_embeddings, embedding_matrix])
    )

    outputs = [concept_predictions]

    if include_second_tiered_learning_objectives:
        visit_prediction_dense = tf.keras.layers.Dense(visit_vocab_size,
                                                       name='visit_prediction_dense')
        is_readmission_prediction_dense = tf.keras.layers.Dense(2, activation='sigmoid',
                                                                name='is_readmissions')
        prolonged_length_stay_prediction_dense = tf.keras.layers.Dense(2, activation='sigmoid',
                                                                       name='visit_prolonged_stays')
        visit_softmax_layer = tf.keras.layers.Softmax(name='visit_predictions')

        visit_predictions = visit_softmax_layer(
            visit_prediction_dense(contextualized_visit_embeddings_without_att)
        )

        is_readmission_prediction = is_readmission_prediction_dense(
            contextualized_visit_embeddings_without_att
        )

        prolonged_length_stay_prediction = prolonged_length_stay_prediction_dense(
            contextualized_visit_embeddings_without_att
        )

        outputs.extend(
            [visit_predictions, is_readmission_prediction, prolonged_length_stay_prediction])

    hierarchical_bert = tf.keras.Model(
        inputs=default_inputs,
        outputs=outputs)

    return hierarchical_bert
