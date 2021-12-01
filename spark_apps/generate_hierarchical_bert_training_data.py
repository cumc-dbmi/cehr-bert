import argparse
import datetime
import os

from pyspark.sql import SparkSession

import spark_apps.parameters as p
from utils.spark_utils import *
from utils.spark_utils import create_hierarchical_sequence_data

VISIT_OCCURRENCE = 'visit_occurrence'
PERSON = 'person'


def main(input_folder, output_folder, domain_table_list, date_filter, max_num_of_visits_per_person):
    spark = SparkSession.builder.appName('Generate Hierarchical Bert Training Data').getOrCreate()

    domain_tables = []
    for domain_table_name in domain_table_list:
        domain_tables.append(preprocess_domain_table(spark, input_folder, domain_table_name))

    visit_occurrence = preprocess_domain_table(spark, input_folder, VISIT_OCCURRENCE)
    person = preprocess_domain_table(spark, input_folder, PERSON)
    patient_event = join_domain_tables(domain_tables) \
        .withColumn('cohort_member_id', F.col('person_id'))

    sequence_data = create_hierarchical_sequence_data(
        person, visit_occurrence, patient_event,
        date_filter=date_filter,
        max_num_of_visits_per_person=max_num_of_visits_per_person
    )

    sequence_data.write.mode('overwrite').parquet(os.path.join(output_folder, p.parquet_data_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for generate training '
                                                 'data for Hierarchical Bert')
    parser.add_argument('-i',
                        '--input_folder',
                        dest='input_folder',
                        action='store',
                        help='The path for your input_folder where the raw data is',
                        required=True)
    parser.add_argument('-o',
                        '--output_folder',
                        dest='output_folder',
                        action='store',
                        help='The path for your output_folder',
                        required=True)
    parser.add_argument('-tc',
                        '--domain_table_list',
                        dest='domain_table_list',
                        nargs='+',
                        action='store',
                        help='The list of domain tables you want to download',
                        required=True)
    parser.add_argument('-d',
                        '--date_filter',
                        dest='date_filter',
                        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'),
                        action='store',
                        required=False,
                        default='2018-01-01')
    parser.add_argument('--max_num_of_visits',
                        dest='max_num_of_visits',
                        action='store',
                        type=int,
                        default=200,
                        help='Max no.of visits per patient to be included',
                        required=False)

    ARGS = parser.parse_args()

    main(ARGS.input_folder,
         ARGS.output_folder,
         ARGS.domain_table_list,
         ARGS.date_filter,
         ARGS.max_num_of_visits)
