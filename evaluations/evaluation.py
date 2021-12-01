from sys import argv
import argparse
import spark_apps.parameters as p
from evaluations.model_evaluators import *

FULL = 'full'
SEQUENCE_MODEL = 'sequence_model'
BASELINE_MODEL = 'baseline_model'
EVALUATION_CHOICES = [FULL, SEQUENCE_MODEL, BASELINE_MODEL]

LSTM = 'lstm'
VANILLA_BERT_LSTM = 'vanilla_bert_lstm'
VANILLA_BERT_FEED_FORWARD = 'vanilla_bert_feed_forward'
SLIDING_BERT = 'sliding_bert'
TEMPORAL_BERT_LSTM = 'temporal_bert_lstm'
RANDOM_VANILLA_BERT_LSTM = 'random_vanilla_bert_lstm'
HIERARCHICAL_BERT_LSTM = 'hierarchical_bert_lstm'
SEQUENCE_MODEL_EVALUATORS = [LSTM, VANILLA_BERT_LSTM, VANILLA_BERT_FEED_FORWARD, TEMPORAL_BERT_LSTM,
                             SLIDING_BERT, RANDOM_VANILLA_BERT_LSTM, HIERARCHICAL_BERT_LSTM]


def evaluate_sequence_models(args):
    # Load the training data
    dataset = pd.read_parquet(args.sequence_model_data_path)
    if LSTM in args.model_evaluators:
        validate_folder(args.time_attention_model_folder)
        time_attention_tokenizer_path = os.path.join(args.time_attention_model_folder,
                                                     p.tokenizer_path)
        time_aware_model_path = os.path.join(args.time_attention_model_folder,
                                             p.time_attention_model_path)
        BiLstmModelEvaluator(
            dataset=dataset,
            evaluation_folder=args.evaluation_folder,
            num_of_folds=args.num_of_folds,
            is_transfer_learning=args.is_transfer_learning,
            training_percentage=args.training_percentage,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            time_aware_model_path=time_aware_model_path,
            tokenizer_path=time_attention_tokenizer_path,
            sequence_model_name=args.sequence_model_name
        ).eval_model()

    if VANILLA_BERT_FEED_FORWARD in args.model_evaluators:
        validate_folder(args.vanilla_bert_model_folder)
        bert_tokenizer_path = os.path.join(args.vanilla_bert_model_folder,
                                           p.tokenizer_path)
        bert_model_path = os.path.join(args.vanilla_bert_model_folder,
                                       p.bert_model_validation_path)
        BertFeedForwardModelEvaluator(
            dataset=dataset,
            evaluation_folder=args.evaluation_folder,
            num_of_folds=args.num_of_folds,
            is_transfer_learning=args.is_transfer_learning,
            training_percentage=args.training_percentage,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            bert_model_path=bert_model_path,
            tokenizer_path=bert_tokenizer_path,
            is_temporal=False,
            sequence_model_name=args.sequence_model_name).eval_model()

    if SLIDING_BERT in args.model_evaluators:
        validate_folder(args.vanilla_bert_model_folder)
        bert_tokenizer_path = os.path.join(args.vanilla_bert_model_folder, p.tokenizer_path)
        bert_model_path = os.path.join(args.vanilla_bert_model_folder, p.bert_model_validation_path)
        SlidingBertModelEvaluator(
            dataset=dataset,
            evaluation_folder=args.evaluation_folder,
            num_of_folds=args.num_of_folds,
            is_transfer_learning=args.is_transfer_learning,
            training_percentage=args.training_percentage,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            bert_model_path=bert_model_path,
            tokenizer_path=bert_tokenizer_path,
            stride=args.stride,
            context_window=args.context_window,
            sequence_model_name=args.sequence_model_name).eval_model()

    if VANILLA_BERT_LSTM in args.model_evaluators:
        validate_folder(args.vanilla_bert_model_folder)
        bert_tokenizer_path = os.path.join(args.vanilla_bert_model_folder,
                                           p.tokenizer_path)
        bert_model_path = os.path.join(args.vanilla_bert_model_folder,
                                       p.bert_model_validation_path)
        BertLstmModelEvaluator(
            dataset=dataset,
            evaluation_folder=args.evaluation_folder,
            num_of_folds=args.num_of_folds,
            is_transfer_learning=args.is_transfer_learning,
            training_percentage=args.training_percentage,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            bert_model_path=bert_model_path,
            tokenizer_path=bert_tokenizer_path,
            is_temporal=False,
            sequence_model_name=args.sequence_model_name).eval_model()

    if TEMPORAL_BERT_LSTM in args.model_evaluators:
        validate_folder(args.temporal_bert_model_folder)
        temporal_bert_tokenizer_path = os.path.join(args.temporal_bert_model_folder,
                                                    p.tokenizer_path)
        temporal_bert_model_path = os.path.join(args.temporal_bert_model_folder,
                                                p.temporal_bert_validation_model_path)
        BertLstmModelEvaluator(
            dataset=dataset,
            evaluation_folder=args.evaluation_folder,
            num_of_folds=args.num_of_folds,
            is_transfer_learning=args.is_transfer_learning,
            training_percentage=args.training_percentage,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            bert_model_path=temporal_bert_model_path,
            tokenizer_path=temporal_bert_tokenizer_path,
            is_temporal=True,
            sequence_model_name=args.sequence_model_name).eval_model()

    if RANDOM_VANILLA_BERT_LSTM in args.model_evaluators:
        validate_folder(args.vanilla_bert_model_folder)
        bert_model_path = os.path.join(args.vanilla_bert_model_folder,
                                       p.bert_model_validation_path)
        bert_tokenizer_path = os.path.join(args.vanilla_bert_model_folder,
                                           p.tokenizer_path)
        visit_tokenizer_path = os.path.join(args.vanilla_bert_model_folder,
                                            p.visit_tokenizer_path)
        RandomVanillaLstmBertModelEvaluator(
            dataset=dataset,
            evaluation_folder=args.evaluation_folder,
            num_of_folds=args.num_of_folds,
            is_transfer_learning=args.is_transfer_learning,
            training_percentage=args.training_percentage,
            max_seq_length=args.max_seq_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            bert_model_path=bert_model_path,
            tokenizer_path=bert_tokenizer_path,
            visit_tokenizer_path=visit_tokenizer_path,
            is_temporal=False,
            sequence_model_name=args.sequence_model_name,
            embedding_size=args.embedding_size,
            depth=args.depth,
            num_heads=args.num_heads,
            use_time_embedding=args.use_time_embedding,
            time_embeddings_size=args.time_embeddings_size
        ).eval_model()

    if HIERARCHICAL_BERT_LSTM in args.model_evaluators:
        validate_folder(args.vanilla_bert_model_folder)
        bert_model_path = os.path.join(args.vanilla_bert_model_folder,
                                       p.bert_model_validation_path)
        bert_tokenizer_path = os.path.join(args.vanilla_bert_model_folder,
                                           p.tokenizer_path)
        HierarchicalBertEvaluator(
            dataset=dataset,
            evaluation_folder=args.evaluation_folder,
            num_of_folds=args.num_of_folds,
            is_transfer_learning=args.is_transfer_learning,
            training_percentage=args.training_percentage,
            max_num_of_visits=args.max_num_of_visits,
            max_num_of_concepts=args.max_num_of_concepts,
            batch_size=args.batch_size,
            epochs=args.epochs,
            bert_model_path=bert_model_path,
            tokenizer_path=bert_tokenizer_path,
            sequence_model_name=args.sequence_model_name
        ).eval_model()


def evaluate_baseline_models(args):
    # Load the training data
    dataset = pd.read_parquet(args.data_path)

    LogisticRegressionModelEvaluator(dataset=dataset,
                                     evaluation_folder=args.evaluation_folder,
                                     num_of_folds=args.num_of_folds,
                                     is_transfer_learning=args.is_transfer_learning,
                                     training_percentage=args.training_percentage).eval_model()
    XGBClassifierEvaluator(dataset=dataset,
                           evaluation_folder=args.evaluation_folder,
                           num_of_folds=args.num_of_folds,
                           is_transfer_learning=args.is_transfer_learning,
                           training_percentage=args.training_percentage).eval_model()


def create_evaluation_args():
    main_parser = argparse.ArgumentParser(
        description='Arguments for evaluating the models')

    sequence_model_required = BASELINE_MODEL not in argv
    baseline_model_required = SEQUENCE_MODEL not in argv
    lstm_model_required = LSTM in argv
    vanilla_bert_lstm = VANILLA_BERT_LSTM in argv
    temporal_bert_lstm = TEMPORAL_BERT_LSTM in argv
    sliding_bert = SLIDING_BERT in argv
    hierarchical_bert = HIERARCHICAL_BERT_LSTM in argv

    main_parser.add_argument('-a',
                             '--action',
                             dest='action',
                             action='store',
                             choices=EVALUATION_CHOICES,
                             help='The action that determines the evaluation process',
                             required=True)
    main_parser.add_argument('-d',
                             '--data_path',
                             dest='data_path',
                             action='store',
                             help='The training data path',
                             required=baseline_model_required)
    main_parser.add_argument('-ef',
                             '--evaluation_folder',
                             dest='evaluation_folder',
                             action='store',
                             required=True)
    main_parser.add_argument('-n',
                             '--num_of_folds',
                             dest='num_of_folds',
                             action='store',
                             required=False,
                             type=int,
                             default=4)
    main_parser.add_argument('--is_transfer_learning',
                             dest='is_transfer_learning',
                             action='store_true')
    main_parser.add_argument('--training_percentage',
                             dest='training_percentage',
                             required=False,
                             action='store',
                             type=float,
                             default=1.0)

    group = main_parser.add_argument_group('sequence model')
    group.add_argument('-me',
                       '--model_evaluators',
                       dest='model_evaluators',
                       action='store',
                       nargs='+',
                       choices=SEQUENCE_MODEL_EVALUATORS,
                       required=sequence_model_required)
    group.add_argument('-sd',
                       '--sequence_model_data_path',
                       dest='sequence_model_data_path',
                       action='store',
                       required=sequence_model_required)
    group.add_argument('-smn',
                       '--sequence_model_name',
                       dest='sequence_model_name',
                       action='store',
                       default=None)
    group.add_argument('-m',
                       '--max_seq_length',
                       dest='max_seq_length',
                       type=int,
                       action='store',
                       required=sequence_model_required)
    group.add_argument('-b',
                       '--batch_size',
                       dest='batch_size',
                       action='store',
                       type=int,
                       required=sequence_model_required)
    group.add_argument('-p',
                       '--epochs',
                       dest='epochs',
                       action='store',
                       type=int,
                       required=sequence_model_required)
    group.add_argument('-ti',
                       '--time_attention_model_folder',
                       dest='time_attention_model_folder',
                       action='store',
                       required=lstm_model_required)
    group.add_argument('-vb',
                       '--vanilla_bert_model_folder',
                       dest='vanilla_bert_model_folder',
                       action='store',
                       required=vanilla_bert_lstm)
    group.add_argument('-tb',
                       '--temporal_bert_model_folder',
                       dest='temporal_bert_model_folder',
                       action='store',
                       required=temporal_bert_lstm)
    group.add_argument('--stride',
                       dest='stride',
                       action='store',
                       type=int,
                       required=sliding_bert)
    group.add_argument('--context_window',
                       dest='context_window',
                       action='store',
                       type=int,
                       required=sliding_bert)
    group.add_argument('--max_num_of_visits',
                       dest='max_num_of_visits',
                       action='store',
                       type=int,
                       required=hierarchical_bert)
    group.add_argument('--max_num_of_concepts',
                       dest='max_num_of_concepts',
                       action='store',
                       type=int,
                       required=hierarchical_bert)
    group.add_argument('--depth',
                       dest='depth',
                       action='store',
                       type=int,
                       default=5,
                       required=False)
    group.add_argument('-nh',
                       '--num_heads',
                       dest='num_heads',
                       action='store',
                       type=int,
                       default=8,
                       required=False)
    group.add_argument('-iv',
                       '--include_visit',
                       dest='include_visit_prediction',
                       action='store_true',
                       required=False)
    group.add_argument('-ut',
                       '--use_time_embedding',
                       dest='use_time_embedding',
                       action='store_true',
                       required=False)
    group.add_argument('--time_embeddings_size',
                       dest='time_embeddings_size',
                       action='store',
                       type=int,
                       default=16,
                       required=False)
    group.add_argument('--embedding_size',
                       dest='embedding_size',
                       action='store',
                       type=int,
                       default=128,
                       required=False)

    return main_parser


def main(args):
    if args.action == BASELINE_MODEL or args.action == FULL:
        evaluate_baseline_models(args)

    if args.action == SEQUENCE_MODEL or args.action == FULL:
        evaluate_sequence_models(args)


if __name__ == "__main__":
    main(create_evaluation_args().parse_args())
