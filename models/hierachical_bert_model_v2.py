from models.custom_layers import *


def transformer_hierarchical_bert_model(num_of_visits,
                                        num_of_concepts,
                                        concept_vocab_size,
                                        embedding_size,
                                        depth: int,
                                        num_heads: int,
                                        num_of_exchanges: int,
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
    :param num_of_exchanges:
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

    pat_mask = tf.keras.layers.Input(shape=(num_of_visits, num_of_concepts,), dtype='int32',
                                     name='pat_mask')

    visit_segment = tf.keras.layers.Input(shape=(num_of_visits,), dtype='int32',
                                          name='visit_segment')

    visit_mask = tf.keras.layers.Input(
        shape=(num_of_visits,),
        dtype='int32',
        name='visit_mask')

    visit_rank_order = tf.keras.layers.Input(
        shape=(num_of_visits,),
        dtype='int32',
        name='visit_rank_order')

    default_inputs = [pat_seq, pat_mask, visit_segment, visit_mask, visit_rank_order]

    pat_concept_mask = tf.reshape(pat_mask, (-1, num_of_concepts))[:, tf.newaxis, tf.newaxis, :]

    visit_concept_mask = visit_mask[:, tf.newaxis, tf.newaxis, :]

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

    positional_encoding_layer = PositionalEncodingLayer(
        max_sequence_length=num_of_visits * num_of_concepts,
        embedding_size=time_embeddings_size,
        name='positional_encoding_layer')

    temporal_transformation_layer = tf.keras.layers.Dense(
        embedding_size,
        activation='tanh',
        name='temporal_transformation')

    concept_embeddings, embedding_matrix = concept_embedding_layer(pat_seq)

    # (batch, num_of_visits, time_embedding_size)
    visit_positional_encoding = positional_encoding_layer(visit_rank_order)
    pat_seq_positional_encoding = tf.tile(
        visit_positional_encoding[:, :, tf.newaxis, :], [1, 1, num_of_concepts, 1])

    concept_embeddings = temporal_transformation_layer(
        tf.concat(
            [concept_embeddings,
             pat_seq_positional_encoding],
            axis=-1, name='concat_for_encoder')
    )

    # dense layer for rescale the patient sequence embeddings back to the original size
    concept_embeddings = visit_segment_layer(
        [visit_segment[:, :, tf.newaxis],
         concept_embeddings]
    )

    # The first bert applied at the visit level
    concept_encoder = Encoder(
        name='concept_encoder',
        num_layers=depth,
        d_model=embedding_size,
        num_heads=num_heads,
        dropout_rate=transformer_dropout)

    # Second bert applied at the patient level to the visit embeddings
    visit_encoder_layer = Encoder(
        name='visit_encoder',
        num_layers=depth,
        d_model=embedding_size,
        num_heads=num_heads,
        dropout_rate=transformer_dropout)

    merge_matrix = tf.constant(
        [1] + [0] * (num_of_concepts - 1),
        dtype=tf.float32
    )[tf.newaxis, tf.newaxis, :, tf.newaxis]

    merge_matrix_inverse = tf.constant(
        [0] + [1] * (num_of_concepts - 1),
        dtype=tf.float32
    )[tf.newaxis, tf.newaxis, :, tf.newaxis]

    multi_head_attention_layer = MultiHeadAttention(embedding_size, num_heads)
    global_embedding_dropout_layer = tf.keras.layers.Dropout(transformer_dropout)
    global_concept_embeddings_normalization = tf.keras.layers.LayerNormalization(
        name='global_concept_embeddings_normalization',
        epsilon=1e-6
    )

    for _ in range(num_of_exchanges):
        # Step 1
        # (batch_size * num_of_visits, num_of_concepts, embedding_size)
        concept_embeddings = tf.reshape(
            concept_embeddings,
            shape=(-1, num_of_concepts, embedding_size)
        )

        concept_embeddings, _ = concept_encoder(
            concept_embeddings,  # be reused
            pat_concept_mask  # not change
        )

        # (batch_size, num_of_visits, num_of_concepts, embedding_size)
        concept_embeddings = tf.reshape(
            concept_embeddings,
            shape=(-1, num_of_visits, num_of_concepts, embedding_size)
        )
        # Step 2 generate visit embeddings
        # Slice out the first contextualized embedding of each visit
        # (batch_size, num_of_visits, embedding_size)
        visit_embeddings = concept_embeddings[:, :, 0]

        # Step 3 decoder applied to patient level
        # Feed augmented visit embeddings into encoders to get contextualized visit embeddings
        # x, enc_output, decoder_mask, encoder_mask
        visit_embeddings, _ = visit_encoder_layer(
            visit_embeddings,
            visit_concept_mask
        )

        # Merge the visit embeddings back into the concept embeddings
        concept_embeddings += (
                concept_embeddings * merge_matrix_inverse +
                tf.expand_dims(
                    visit_embeddings,
                    axis=-2
                ) * merge_matrix
        )

    # Reshape the data in visit view back to patient view:
    # (batch, num_of_visits * num_of_concepts, embedding_size)
    concept_embeddings = tf.reshape(
        concept_embeddings,
        shape=(-1, num_of_visits * num_of_concepts, embedding_size)
    )

    global_concept_embeddings, _ = multi_head_attention_layer(
        visit_embeddings,
        visit_embeddings,
        concept_embeddings,
        visit_concept_mask)

    global_concept_embeddings = global_embedding_dropout_layer(
        global_concept_embeddings
    )

    global_concept_embeddings = global_concept_embeddings_normalization(
        global_concept_embeddings + concept_embeddings
    )

    concept_output_layer = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name='concept_prediction_logits')

    concept_softmax_layer = tf.keras.layers.Softmax(
        name='concept_predictions'
    )

    concept_predictions = concept_softmax_layer(
        concept_output_layer([global_concept_embeddings, embedding_matrix])
    )

    outputs = [concept_predictions]

    hierarchical_bert = tf.keras.Model(
        inputs=default_inputs,
        outputs=outputs)

    return hierarchical_bert
