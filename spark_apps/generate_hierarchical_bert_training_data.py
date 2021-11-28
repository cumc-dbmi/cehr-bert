import argparse
import datetime
import os

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf

import spark_apps.parameters as p
from utils.spark_utils import *

VISIT_OCCURRENCE = 'visit_occurrence'
PERSON = 'person'
UNKNOWN_CONCEPT = '[UNKNOWN]'


def create_hierarchical_sequence_data(person, visit_occurrence, patient_event,
                                      date_filter=None, max_num_of_visits_per_person=200):
    @pandas_udf('string')
    def pandas_udf_to_att(time_intervals: pd.Series) -> pd.Series:
        return time_intervals.apply(time_token_func)

    if date_filter:
        patient_event = patient_event.where(F.col('date').cast('date') >= date_filter)

    visit_occurrence_person = create_visit_person_join(pandas_udf_to_att, person, visit_occurrence)

    visit_columns = [visit_occurrence_person[fieldName] for fieldName in
                     visit_occurrence_person.schema.fieldNames()]

    patient_columns = [
        F.coalesce(patient_event['standard_concept_id'], F.lit(UNKNOWN_CONCEPT)).alias(
            'standard_concept_id'),
        F.coalesce(patient_event['date'],
                   visit_occurrence['visit_start_date']).alias('date'),
        F.coalesce(patient_event['domain'], F.lit('unknown')).alias('domain')]

    patient_event = visit_occurrence_person.join(patient_event,
                                                 'visit_occurrence_id',
                                                 'left_outer') \
        .select(visit_columns + patient_columns) \
        .withColumn('standard_concept_id', F.col('standard_concept_id').cast('string')) \
        .withColumn('cohort_member_id', F.col('person_id')) \
        .withColumn('age',
                    F.ceil(F.months_between(F.col('date'), F.col("birth_datetime")) / F.lit(12)))

    weeks_since_epoch_udf = (F.unix_timestamp('date') / F.lit(24 * 60 * 60 * 7)).cast('int')

    visit_concept_order_udf = F.row_number().over(
        W.partitionBy('cohort_member_id',
                      'person_id',
                      'visit_occurrence_id').orderBy('date', 'standard_concept_id')
    )

    patient_event = patient_event \
        .withColumn('date', F.col('date').cast('date')) \
        .withColumn('date_in_week', weeks_since_epoch_udf) \
        .withColumn('visit_concept_order', visit_concept_order_udf)

    insert_cls_tokens = patient_event \
        .where('visit_concept_order == 1') \
        .withColumn('standard_concept_id', F.lit('CLS')) \
        .withColumn('domain', F.lit('CLS')) \
        .withColumn('visit_concept_order', F.lit(0)) \
        .withColumn('date', F.col('visit_start_date'))

    struct_columns = ['visit_concept_order', 'standard_concept_id', 'date_in_week', 'age']

    patent_visit_sequence = patient_event.union(insert_cls_tokens) \
        .withColumn('visit_struct_data', F.struct(struct_columns)) \
        .groupBy('cohort_member_id', 'person_id', 'visit_occurrence_id') \
        .agg(F.sort_array(F.collect_set('visit_struct_data')).alias('visit_struct_data'),
             F.first('visit_start_date').alias('visit_start_date'),
             F.first('visit_rank_order').alias('visit_rank_order'),
             F.first('visit_concept_id').alias('visit_concept_id'),
             F.first('is_readmission').alias('is_readmission'),
             F.first('visit_segment').alias('visit_segment'),
             F.first('time_interval_att').alias('time_interval_att'),
             F.first('prolonged_stay').alias('prolonged_stay'),
             F.count('standard_concept_id').alias('num_of_concepts')) \
        .orderBy(['person_id', 'visit_rank_order'])

    patient_visit_sequence = patent_visit_sequence \
        .withColumn('visit_concept_orders', F.col('visit_struct_data.visit_concept_order')) \
        .withColumn('visit_concept_ids', F.col('visit_struct_data.standard_concept_id')) \
        .withColumn('visit_concept_dates', F.col('visit_struct_data.date_in_week')) \
        .withColumn('visit_concept_ages', F.col('visit_struct_data.age')) \
        .withColumn('visit_mask', F.lit(0)) \
        .drop('visit_struct_data')

    visit_struct_data_columns = ['visit_rank_order',
                                 'visit_occurrence_id',
                                 'visit_start_date',
                                 'visit_concept_id',
                                 'prolonged_stay',
                                 'visit_mask',
                                 'visit_segment',
                                 'num_of_concepts',
                                 'is_readmission',
                                 'time_interval_att',
                                 'visit_concept_orders',
                                 'visit_concept_ids',
                                 'visit_concept_dates',
                                 'visit_concept_ages']

    visit_weeks_since_epoch_udf = (F.unix_timestamp(F.col('visit_start_date').cast('date')) / F.lit(
        24 * 60 * 60 * 7)).cast('int')

    patient_sequence = patient_visit_sequence \
        .withColumn('visit_start_date', visit_weeks_since_epoch_udf) \
        .withColumn('visit_struct_data',
                    F.struct(visit_struct_data_columns).alias('visit_struct_data')) \
        .groupBy('cohort_member_id', 'person_id') \
        .agg(F.sort_array(F.collect_list('visit_struct_data')).alias('patient_list'),
             F.sum(F.lit(1) - F.col('visit_mask')).alias('num_of_visits'),
             F.sum('num_of_concepts').alias('num_of_concepts'))

    patient_sequence = patient_sequence \
        .where(F.col('num_of_visits') <= max_num_of_visits_per_person) \
        .withColumn('concept_orders', F.col('patient_list.visit_concept_orders')) \
        .withColumn('concept_ids', F.col('patient_list.visit_concept_ids')) \
        .withColumn('dates', F.col('patient_list.visit_concept_dates')) \
        .withColumn('ages', F.col('patient_list.visit_concept_ages')) \
        .withColumn('visit_dates', F.col('patient_list.visit_start_date')) \
        .withColumn('visit_segments', F.col('patient_list.visit_segment')) \
        .withColumn('visit_masks', F.col('patient_list.visit_mask')) \
        .withColumn('visit_concept_ids',
                    F.col('patient_list.visit_concept_id').cast(T.ArrayType(T.StringType()))) \
        .withColumn('time_interval_atts', F.col('patient_list.time_interval_att')) \
        .withColumn('is_readmissions',
                    F.col('patient_list.is_readmission').cast(T.ArrayType(T.IntegerType()))) \
        .withColumn('visit_prolonged_stays',
                    F.col('patient_list.prolonged_stay').cast(T.ArrayType(T.IntegerType()))) \
        .drop('patient_list')

    return patient_sequence


def create_visit_person_join(pandas_udf_to_att, person, visit_occurrence):
    visit_rank_udf = F.row_number().over(
        W.partitionBy('person_id').orderBy('visit_start_date', 'visit_end_date',
                                           'visit_occurrence_id'))
    visit_segment_udf = F.col('visit_rank_order') % F.lit(2) + 1
    visit_windowing = W.partitionBy('person_id').orderBy('visit_start_date',
                                                         'visit_end_date',
                                                         'visit_occurrence_id')
    readmission_logic = ((F.col('time_interval') <= 30) \
                         & (F.col('visit_concept_id').isin([9201, 262])) \
                         & (F.col('prev_visit_concept_id').isin([9201, 262]))).cast('integer')
    visit_occurrence = visit_occurrence.select('visit_occurrence_id',
                                               'person_id',
                                               'visit_concept_id',
                                               'visit_start_date',
                                               'visit_end_date') \
        .where('visit_start_date IS NOT NULL AND visit_end_date IS NOT NULL') \
        .withColumn('visit_rank_order', visit_rank_udf) \
        .withColumn('visit_segment', visit_segment_udf) \
        .withColumn('prev_visit_occurrence_id', F.lag('visit_occurrence_id').over(visit_windowing)) \
        .withColumn('prev_visit_concept_id', F.lag('visit_concept_id').over(visit_windowing)) \
        .withColumn('prev_visit_start_date', F.lag('visit_start_date').over(visit_windowing)) \
        .withColumn('prev_visit_end_date', F.lag('visit_end_date').over(visit_windowing)) \
        .withColumn('time_interval', F.datediff('visit_start_date', 'prev_visit_end_date')) \
        .withColumn('time_interval',
                    F.when(F.col('time_interval') < 0, F.lit(0)).otherwise(F.col('time_interval'))) \
        .withColumn('time_interval_att', pandas_udf_to_att('time_interval')) \
        .withColumn('is_readmission', readmission_logic)
    visit_occurrence = visit_occurrence \
        .withColumn('prolonged_stay',
                    (F.datediff('visit_end_date', 'visit_start_date') >= 7).cast('integer')) \
        .select('visit_occurrence_id',
                'visit_concept_id',
                'person_id',
                'prolonged_stay',
                'is_readmission',
                'time_interval_att',
                'visit_rank_order',
                'visit_start_date',
                'visit_segment')
    person = person.select('person_id', F.coalesce('birth_datetime',
                                                   F.concat('year_of_birth', F.lit('-01-01')).cast(
                                                       'timestamp')).alias('birth_datetime'))
    return visit_occurrence.join(person, 'person_id')


def main(input_folder, output_folder, domain_table_list, date_filter, max_num_of_visits_per_person):
    spark = SparkSession.builder.appName('Generate Hierarchical Bert Training Data').getOrCreate()

    domain_tables = []
    for domain_table_name in domain_table_list:
        domain_tables.append(preprocess_domain_table(spark, input_folder, domain_table_name))

    visit_occurrence = preprocess_domain_table(spark, input_folder, VISIT_OCCURRENCE)
    person = preprocess_domain_table(spark, input_folder, PERSON)
    patient_event = join_domain_tables(domain_tables)

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
