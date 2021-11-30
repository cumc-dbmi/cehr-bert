import argparse
import datetime

parquet_data_path = 'patient_sequence'
feather_data_path = 'patient_sequence.pickle'
tokenizer_path = 'tokenizer.pickle'
visit_tokenizer_path = 'visit_tokenizer.pickle'
time_attention_model_path = 'time_aware_model.h5'
bert_model_path = 'bert_model_{epoch:02d}_{loss:.2f}.h5'
temporal_bert_model_path = 'temporal_bert_model_{epoch:02d}_{loss:.2f}.h5'
bert_model_validation_path = 'bert_model.h5'
temporal_bert_validation_model_path = 'temporal_bert_model.h5'

mortality_data_path = 'mortality'
heart_failure_data_path = 'heart_failure'
hospitalization_data_path = 'hospitalization'


def valid_date(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)


def create_spark_args():
    parser = argparse.ArgumentParser(
        description='Arguments for spark applications for generating cohort definitions')
    parser.add_argument('-c',
                        '--cohort_name',
                        dest='cohort_name',
                        action='store',
                        help='The cohort name',
                        required=True)
    parser.add_argument('-i',
                        '--input_folder',
                        dest='input_folder',
                        action='store',
                        help='The path for your input_folder where the sequence data is',
                        required=True)
    parser.add_argument('-o',
                        '--output_folder',
                        dest='output_folder',
                        action='store',
                        help='The path for your output_folder',
                        required=True)
    parser.add_argument('-dl',
                        '--date_lower_bound',
                        dest='date_lower_bound',
                        action='store',
                        help='The date filter lower bound for filtering training data',
                        required=True,
                        type=valid_date)
    parser.add_argument('-du',
                        '--date_upper_bound',
                        dest='date_upper_bound',
                        action='store',
                        help='The date filter upper bound for filtering training data',
                        required=True,
                        type=valid_date)
    parser.add_argument('-l',
                        '--age_lower_bound',
                        dest='age_lower_bound',
                        action='store',
                        help='The age lower bound',
                        required=False,
                        type=int,
                        default=0)
    parser.add_argument('-u',
                        '--age_upper_bound',
                        dest='age_upper_bound',
                        action='store',
                        help='The age upper bound',
                        required=False,
                        type=int,
                        default=100)
    parser.add_argument('-ow',
                        '--observation_window',
                        dest='observation_window',
                        action='store',
                        help='The observation window in days for extracting features',
                        required=False,
                        type=int,
                        default=365)
    parser.add_argument('-pw',
                        '--prediction_window',
                        dest='prediction_window',
                        action='store',
                        help='The prediction window in which the prediction is made',
                        required=False,
                        type=int,
                        default=180)
    parser.add_argument('-ps',
                        '--prediction_start_days',
                        dest='prediction_start_days',
                        action='store',
                        help='The prediction start days in which the prediction is made',
                        required=False,
                        type=int,
                        default=1)
    parser.add_argument('-hw',
                        '--hold_off_window',
                        dest='hold_off_window',
                        action='store',
                        help='The hold off window for excluding the features',
                        required=False,
                        type=int,
                        default=0)
    parser.add_argument('--num_of_visits',
                        dest='num_of_visits',
                        action='store',
                        help='The number of visits to qualify for the inclusion of the cohorts',
                        required=False,
                        type=int,
                        default=0)
    parser.add_argument('--num_of_concepts',
                        dest='num_of_concepts',
                        action='store',
                        help='The number of concepts to qualify for the inclusion of the cohorts',
                        required=False,
                        type=int,
                        default=0)
    parser.add_argument('-iw',
                        '--is_window_post_index',
                        dest='is_window_post_index',
                        action='store_true',
                        help='Indicate if the observation window is pre/post the index date')
    parser.add_argument('-iv',
                        '--include_visit_type',
                        dest='include_visit_type',
                        action='store_true',
                        help='Specify whether to include visit types for '
                             'generating the training data')
    parser.add_argument('-ev',
                        '--exclude_visit_tokens',
                        dest='exclude_visit_tokens',
                        action='store_true',
                        help='Specify whether or not to exclude the VS and VE tokens')
    parser.add_argument('-f',
                        '--is_feature_concept_frequency',
                        dest='is_feature_concept_frequency',
                        action='store_true',
                        help='Specify whether the features are concept counts or not')
    parser.add_argument('-ir',
                        '--is_roll_up_concept',
                        dest='is_roll_up_concept',
                        action='store_true',
                        help='Specify whether to roll up the concepts to their ancestors')
    parser.add_argument('-ip',
                        '--is_new_patient_representation',
                        dest='is_new_patient_representation',
                        action='store_true',
                        help='Specify whether to generate the sequence of '
                             'EHR records using the new patient representation')
    parser.add_argument('-ih',
                        '--is_hierarchical_bert',
                        dest='is_hierarchical_bert',
                        action='store_true',
                        help='Specify whether to generate the sequence of '
                             'EHR records using the hierarchical patient representation')
    parser.add_argument('-cbs',
                        '--classic_bert_seq',
                        dest='classic_bert_seq',
                        action='store_true',
                        help='Specify whether to generate the sequence of '
                             'EHR records using the classic BERT sequence representation where '
                             'visits are separated by a SEP token')
    parser.add_argument('--is_first_time_outcome',
                        dest='is_first_time_outcome',
                        action='store_true',
                        help='is the outcome the first time occurrence?')
    parser.add_argument('--is_remove_index_prediction_starts',
                        dest='is_remove_index_prediction_starts',
                        action='store_true',
                        help='is outcome between index_date and prediction start window removed?')
    parser.add_argument('--is_prediction_window_unbounded',
                        dest='is_prediction_window_unbounded',
                        action='store_true',
                        help='is the end of the prediction window unbounded?')
    parser.add_argument('--is_observation_window_unbounded',
                        dest='is_observation_window_unbounded',
                        action='store_true',
                        help='is the observation window unbounded?')
    return parser.parse_args()
