from os import path
import numpy as np
import math
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Window as W

from utils.logging_utils import *

PERSON = 'person'
VISIT_OCCURRENCE = 'visit_occurrence'

DOMAIN_KEY_FIELDS = {
    'condition_occurrence_id': ('condition_concept_id', 'condition_start_date', 'condition'),
    'procedure_occurrence_id': ('procedure_concept_id', 'procedure_date', 'procedure'),
    'drug_exposure_id': ('drug_concept_id', 'drug_exposure_start_date', 'drug'),
    'measurement_id': ('measurement_concept_id', 'measurement_date', 'measurement')
}

LOGGER = logging.getLogger(__name__)


def time_token_func(time_delta):
    if np.isnan(time_delta):
        return None
    if time_delta < 0:
        return 'W-1'
    if time_delta < 28:
        return f'W{str(math.floor(time_delta / 7))}'
    if time_delta < 360:
        return f'M{str(math.floor(time_delta / 30))}'
    return 'LT'


def get_key_fields(domain_table):
    field_names = domain_table.schema.fieldNames()
    for k, v in DOMAIN_KEY_FIELDS.items():
        if k in field_names:
            return v
    return (get_concept_id_field(domain_table), get_domain_date_field(domain_table),
            get_domain_field(domain_table))


def get_domain_date_field(domain_table):
    # extract the domain start_date column
    return [f for f in domain_table.schema.fieldNames() if 'date' in f][0]


def get_concept_id_field(domain_table):
    return [f for f in domain_table.schema.fieldNames() if 'concept_id' in f][0]


def get_domain_field(domain_table):
    return get_concept_id_field(domain_table).replace('_concept_id', '')


def create_file_path(input_folder, table_name):
    if input_folder[-1] == '/':
        file_path = input_folder + table_name
    else:
        file_path = input_folder + '/' + table_name

    return file_path


def join_domain_tables(domain_tables):
    """Standardize the format of OMOP domain tables using a time frame

    Keyword arguments:
    domain_tables -- the array containing the OMOOP domain tabls except visit_occurrence

    The the output columns of the domain table is converted to the same standard format as the following
    (person_id, standard_concept_id, date, lower_bound, upper_bound, domain).
    In this case, co-occurrence is defined as those concept ids that have co-occurred
    within the same time window of a patient.

    """
    patient_event = None

    for domain_table in domain_tables:
        # extract the domain concept_id from the table fields. E.g. condition_concept_id from
        # condition_occurrence extract the domain start_date column extract the name of the table
        concept_id_field, date_field, table_domain_field = get_key_fields(domain_table)
        # standardize the output columns
        domain_table = domain_table.where(F.col(concept_id_field).cast('string') != '0') \
            .withColumn('date', F.to_date(F.col(date_field)))

        domain_table = domain_table.select(domain_table['person_id'],
                                           domain_table[concept_id_field].alias(
                                               'standard_concept_id'),
                                           domain_table['date'],
                                           domain_table['visit_occurrence_id'],
                                           F.lit(table_domain_field).alias('domain')) \
            .distinct()

        if patient_event is None:
            patient_event = domain_table
        else:
            patient_event = patient_event.union(domain_table)

    return patient_event


def preprocess_domain_table(spark, input_folder, domain_table_name, with_rollup=False):
    domain_table = spark.read.parquet(create_file_path(input_folder, domain_table_name))

    if 'concept' in domain_table_name.lower():
        return domain_table

    # lowercase the schema fields
    domain_table = domain_table.select(
        [F.col(f_n).alias(f_n.lower()) for f_n in domain_table.schema.fieldNames()])

    # Always roll up the drug concepts to the ingredient level
    if domain_table_name == 'drug_exposure' \
            and path.exists(create_file_path(input_folder, 'concept')) \
            and path.exists(create_file_path(input_folder, 'concept_ancestor')):
        concept = spark.read.parquet(create_file_path(input_folder, 'concept'))
        concept_ancestor = spark.read.parquet(
            create_file_path(input_folder, 'concept_ancestor'))
        domain_table = roll_up_to_drug_ingredients(domain_table, concept, concept_ancestor)

    if with_rollup:
        if domain_table_name == 'condition_occurrence' \
                and path.exists(create_file_path(input_folder, 'concept')) \
                and path.exists(create_file_path(input_folder, 'concept_relationship')):
            concept = spark.read.parquet(create_file_path(input_folder, 'concept'))
            concept_relationship = spark.read.parquet(
                create_file_path(input_folder, 'concept_relationship'))
            domain_table = roll_up_diagnosis(domain_table, concept, concept_relationship)

        if domain_table_name == 'procedure_occurrence' \
                and path.exists(create_file_path(input_folder, 'concept')) \
                and path.exists(create_file_path(input_folder, 'concept_ancestor')):
            concept = spark.read.parquet(create_file_path(input_folder, 'concept'))
            concept_ancestor = spark.read.parquet(
                create_file_path(input_folder, 'concept_ancestor'))
            domain_table = roll_up_procedure(domain_table, concept, concept_ancestor)

    return domain_table


def roll_up_to_drug_ingredients(drug_exposure, concept, concept_ancestor):
    # lowercase the schema fields
    drug_exposure = drug_exposure.select(
        [F.col(f_n).alias(f_n.lower()) for f_n in drug_exposure.schema.fieldNames()])

    drug_ingredient = drug_exposure.select('drug_concept_id').distinct() \
        .join(concept_ancestor, F.col('drug_concept_id') == F.col('descendant_concept_id')) \
        .join(concept, F.col('ancestor_concept_id') == F.col('concept_id')) \
        .where(concept['concept_class_id'] == 'Ingredient') \
        .select(F.col('drug_concept_id'), F.col('concept_id').alias('ingredient_concept_id'))

    drug_ingredient_fields = [
        F.coalesce(F.col('ingredient_concept_id'), F.col('drug_concept_id')).alias(
            'drug_concept_id')]
    drug_ingredient_fields.extend(
        [F.col(field_name) for field_name in drug_exposure.schema.fieldNames() if
         field_name != 'drug_concept_id'])

    drug_exposure = drug_exposure.join(drug_ingredient, 'drug_concept_id', 'left_outer') \
        .select(drug_ingredient_fields)

    return drug_exposure


def roll_up_diagnosis(condition_occurrence, concept, concept_relationship):
    list_3dig_code = ['3-char nonbill code', '3-dig nonbill code', '3-char billing code',
                      '3-dig billing code',
                      '3-dig billing E code', '3-dig billing V code', '3-dig nonbill E code',
                      '3-dig nonbill V code']

    condition_occurrence = condition_occurrence.select(
        [F.col(f_n).alias(f_n.lower()) for f_n in condition_occurrence.schema.fieldNames()])

    condition_icd = condition_occurrence.select('condition_source_concept_id').distinct() \
        .join(concept, (F.col('condition_source_concept_id') == F.col('concept_id'))) \
        .where(concept['domain_id'] == 'Condition') \
        .where(concept['vocabulary_id'] != 'SNOMED') \
        .select(F.col('condition_source_concept_id'),
                F.col('vocabulary_id').alias('child_vocabulary_id'),
                F.col('concept_class_id').alias('child_concept_class_id'))

    condition_icd_hierarchy = condition_icd.join(concept_relationship,
                                                 F.col('condition_source_concept_id') == F.col(
                                                     'concept_id_1')) \
        .join(concept, (F.col('concept_id_2') == F.col('concept_id')) & (
        F.col('concept_class_id').isin(list_3dig_code)), how='left') \
        .select(F.col('condition_source_concept_id').alias('source_concept_id'),
                F.col('child_concept_class_id'), F.col('concept_id').alias('parent_concept_id'),
                F.col('concept_name').alias('parent_concept_name'),
                F.col('vocabulary_id').alias('parent_vocabulary_id'),
                F.col('concept_class_id').alias('parent_concept_class_id')).distinct()

    condition_icd_hierarchy = condition_icd_hierarchy.withColumn('ancestor_concept_id', F.when(
        F.col('child_concept_class_id').isin(list_3dig_code), F.col('source_concept_id')).otherwise(
        F.col('parent_concept_id'))) \
        .dropna(subset='ancestor_concept_id')

    condition_occurrence_fields = [F.col(f_n).alias(f_n.lower()) for f_n in
                                   condition_occurrence.schema.fieldNames() if
                                   f_n != 'condition_source_concept_id']
    condition_occurrence_fields.append(F.coalesce(F.col('ancestor_concept_id'),
                                                  F.col('condition_source_concept_id')).alias(
        'condition_source_concept_id'))

    condition_occurrence = condition_occurrence.join(condition_icd_hierarchy, condition_occurrence[
        'condition_source_concept_id'] == condition_icd_hierarchy['source_concept_id'], how='left') \
        .select(condition_occurrence_fields).withColumn('condition_concept_id',
                                                        F.col('condition_source_concept_id'))
    return condition_occurrence


def roll_up_procedure(procedure_occurrence, concept, concept_ancestor):
    def extract_parent_code(concept_code):
        return concept_code.split('.')[0]

    parent_code_udf = F.udf(lambda code: extract_parent_code(code), T.StringType())

    procedure_code = procedure_occurrence.select('procedure_source_concept_id').distinct() \
        .join(concept, F.col('procedure_source_concept_id') == F.col('concept_id')) \
        .where(concept['domain_id'] == 'Procedure') \
        .select(F.col('procedure_source_concept_id').alias('source_concept_id'),
                F.col('vocabulary_id').alias('child_vocabulary_id'),
                F.col('concept_class_id').alias('child_concept_class_id'),
                F.col('concept_code').alias('child_concept_code'))

    # cpt code rollup
    cpt_code = procedure_code.where(F.col('child_vocabulary_id') == 'CPT4')

    cpt_hierarchy = cpt_code.join(concept_ancestor,
                                  cpt_code['source_concept_id'] == concept_ancestor[
                                      'descendant_concept_id']) \
        .join(concept, concept_ancestor['ancestor_concept_id'] == concept['concept_id']) \
        .where(concept['vocabulary_id'] == 'CPT4') \
        .select(F.col('source_concept_id'), F.col('child_concept_class_id'),
                F.col('ancestor_concept_id').alias('parent_concept_id'),
                F.col('min_levels_of_separation'),
                F.col('concept_class_id').alias('parent_concept_class_id'))

    cpt_hierarchy_level_1 = cpt_hierarchy.where(F.col('min_levels_of_separation') == 1) \
        .where(F.col('child_concept_class_id') == 'CPT4') \
        .where(F.col('parent_concept_class_id') == 'CPT4 Hierarchy') \
        .select(F.col('source_concept_id'), F.col('parent_concept_id'))

    cpt_hierarchy_level_1 = cpt_hierarchy_level_1.join(concept_ancestor, (
            cpt_hierarchy_level_1['source_concept_id'] == concept_ancestor['descendant_concept_id'])
                                                       & (concept_ancestor[
                                                              'min_levels_of_separation'] == 1),
                                                       how='left') \
        .select(F.col('source_concept_id'), F.col('parent_concept_id'),
                F.col('ancestor_concept_id').alias('root_concept_id'))

    cpt_hierarchy_level_1 = cpt_hierarchy_level_1.withColumn('isroot', F.when(
        cpt_hierarchy_level_1['root_concept_id'] == 45889197,
        cpt_hierarchy_level_1['source_concept_id']) \
                                                             .otherwise(
        cpt_hierarchy_level_1['parent_concept_id'])) \
        .select(F.col('source_concept_id'), F.col('isroot').alias('ancestor_concept_id'))

    cpt_hierarchy_level_0 = cpt_hierarchy.groupby('source_concept_id').max() \
        .where(F.col('max(min_levels_of_separation)') == 0) \
        .select(F.col('source_concept_id').alias('cpt_level_0_concept_id'))

    cpt_hierarchy_level_0 = cpt_hierarchy.join(cpt_hierarchy_level_0,
                                               cpt_hierarchy['source_concept_id'] ==
                                               cpt_hierarchy_level_0['cpt_level_0_concept_id']) \
        .select(F.col('source_concept_id'), F.col('parent_concept_id').alias('ancestor_concept_id'))

    cpt_hierarchy_rollup_all = cpt_hierarchy_level_1.union(cpt_hierarchy_level_0).drop_duplicates()

    # ICD code rollup
    icd_list = ['ICD9CM', 'ICD9Proc', 'ICD10CM']

    procedure_icd = procedure_code.where(F.col('vocabulary_id').isin(icd_list))

    procedure_icd = procedure_icd.withColumn('parent_concept_code',
                                             parent_code_udf(F.col('child_concept_code'))) \
        .withColumnRenamed('procedure_source_concept_id', 'source_concept_id') \
        .withColumnRenamed('concept_name', 'child_concept_name') \
        .withColumnRenamed('vocabulary_id', 'child_vocabulary_id') \
        .withColumnRenamed('concept_code', 'child_concept_code') \
        .withColumnRenamed('concept_class_id', 'child_concept_class_id')

    procedure_icd_map = procedure_icd.join(concept, (
            procedure_icd['parent_concept_code'] == concept['concept_code'])
                                           & (procedure_icd['child_vocabulary_id'] == concept[
        'vocabulary_id']), how='left') \
        .select('source_concept_id', F.col('concept_id').alias('ancestor_concept_id')).distinct()

    # ICD10PCS rollup
    procedure_10pcs = procedure_code.where(F.col('vocabulary_id') == 'ICD10PCS')

    procedure_10pcs = procedure_10pcs.withColumn('parent_concept_code',
                                                 F.substring(F.col('child_concept_code'), 1, 3)) \
        .withColumnRenamed('procedure_source_concept_id', 'source_concept_id') \
        .withColumnRenamed('concept_name', 'child_concept_name') \
        .withColumnRenamed('vocabulary_id', 'child_vocabulary_id') \
        .withColumnRenamed('concept_code', 'child_concept_code') \
        .withColumnRenamed('concept_class_id', 'child_concept_class_id')

    procedure_10pcs_map = procedure_10pcs.join(concept, (
            procedure_10pcs['parent_concept_code'] == concept['concept_code'])
                                               & (procedure_10pcs['child_vocabulary_id'] == concept[
        'vocabulary_id']), how='left') \
        .select('source_concept_id', F.col('concept_id').alias('ancestor_concept_id')).distinct()

    # HCPCS rollup --- keep the concept_id itself
    procedure_hcpcs = procedure_code.where(F.col('child_vocabulary_id') == 'HCPCS')
    procedure_hcpcs_map = procedure_hcpcs.withColumn('ancestor_concept_id',
                                                     F.col('source_concept_id')) \
        .select('source_concept_id', 'ancestor_concept_id').distinct()

    procedure_hierarchy = cpt_hierarchy_rollup_all \
        .union(procedure_icd_map) \
        .union(procedure_10pcs_map) \
        .union(procedure_hcpcs_map) \
        .distinct()
    procedure_occurrence_fields = [F.col(f_n).alias(f_n.lower()) for f_n in
                                   procedure_occurrence.schema.fieldNames() if
                                   f_n != 'procedure_source_concept_id']
    procedure_occurrence_fields.append(F.coalesce(F.col('ancestor_concept_id'),
                                                  F.col('procedure_source_concept_id')).alias(
        'procedure_source_concept_id'))

    procedure_occurrence = procedure_occurrence.join(procedure_hierarchy, procedure_occurrence[
        'procedure_source_concept_id'] == procedure_hierarchy['source_concept_id'], how='left') \
        .select(procedure_occurrence_fields) \
        .withColumn('procedure_concept_id', F.col('procedure_source_concept_id'))
    return procedure_occurrence


def create_sequence_data(patient_event,
                         date_filter=None,
                         include_visit_type=False,
                         classic_bert_seq=False):
    """
    Create a sequence of the events associated with one patient in a chronological order
    :param patient_event:
    :param date_filter:
    :param include_visit_type:
    :param classic_bert_seq:
    :return:
    """

    if date_filter:
        patient_event = patient_event.where(F.col('date') >= date_filter)

    # Define a list of custom UDFs for creating custom columns
    date_conversion_udf = (F.unix_timestamp('date') / F.lit(24 * 60 * 60 * 7)).cast('int')
    earliest_visit_date_udf = F.min('date_in_week').over(W.partitionBy('visit_occurrence_id'))

    visit_rank_udf = F.dense_rank().over(
        W.partitionBy('cohort_member_id', 'person_id').orderBy('earliest_visit_date'))
    visit_segment_udf = F.col('visit_rank_order') % F.lit(2) + 1

    # Derive columns
    patient_event = patient_event.where('visit_occurrence_id IS NOT NULL') \
        .withColumn('date_in_week', date_conversion_udf) \
        .withColumn('earliest_visit_date', earliest_visit_date_udf) \
        .withColumn('visit_rank_order', visit_rank_udf) \
        .withColumn('visit_segment', visit_segment_udf) \
        .withColumn('priority', F.lit(0))

    if classic_bert_seq:
        # Udf for identifying the earliest date associated with a visit_occurrence_id
        visit_start_date_udf = F.first('date').over(
            W.partitionBy('cohort_member_id', 'person_id', 'visit_occurrence_id').orderBy('date'))

        # Udf for identifying the previous visit_occurrence_id
        prev_visit_occurrence_id_udf = F.lag('visit_occurrence_id').over(
            W.partitionBy('cohort_member_id', 'person_id').orderBy('visit_start_date',
                                                                   'visit_occurrence_id'))

        # We can achieve this by overwriting the record with the earliest time stamp
        separator_events = patient_event.withColumn('visit_start_date', visit_start_date_udf) \
            .withColumn('prev_visit_occurrence_id', prev_visit_occurrence_id_udf) \
            .where('prev_visit_occurrence_id IS NOT NULL') \
            .where('visit_occurrence_id <> prev_visit_occurrence_id') \
            .withColumn('domain', F.lit('Separator')) \
            .withColumn('standard_concept_id', F.lit('SEP')) \
            .withColumn('priority', F.lit(-1)) \
            .withColumn('visit_segment', F.lit(0)) \
            .select(patient_event.schema.fieldNames())

        # Combine this artificial token SEP with the original data
        patient_event = patient_event.union(separator_events)

    order_udf = F.row_number().over(
        W.partitionBy('cohort_member_id', 'person_id').orderBy('earliest_visit_date',
                                                               'visit_occurrence_id',
                                                               'priority', 'date_in_week',
                                                               'standard_concept_id'))
    # Group the data into sequences
    output_columns = ['order', 'date_in_week', 'standard_concept_id',
                      'visit_segment', 'age', 'visit_rank_order']

    if include_visit_type:
        output_columns.append('visit_concept_id')

    # Group by data by person_id and put all the events into a list
    # The order of the list is determined by the order column
    patient_grouped_events = patient_event.withColumn('order', order_udf) \
        .withColumn('date_concept_id_period', F.struct(output_columns)) \
        .groupBy('person_id', 'cohort_member_id') \
        .agg(F.sort_array(F.collect_set('date_concept_id_period')).alias('date_concept_id_period'),
             F.min('earliest_visit_date').alias('earliest_visit_date'),
             F.max('date').alias('max_event_date'),
             F.max('visit_rank_order').alias('num_of_visits'),
             F.count('standard_concept_id').alias('num_of_concepts')) \
        .withColumn('orders',
                    F.col('date_concept_id_period.order').cast(T.ArrayType(T.IntegerType()))) \
        .withColumn('dates', F.col('date_concept_id_period.date_in_week')) \
        .withColumn('concept_ids', F.col('date_concept_id_period.standard_concept_id')) \
        .withColumn('visit_segments', F.col('date_concept_id_period.visit_segment')) \
        .withColumn('ages', F.col('date_concept_id_period.age')) \
        .withColumn('visit_concept_orders', F.col('date_concept_id_period.visit_rank_order'))

    # Default columns in the output dataframe
    columns_for_output = ['cohort_member_id', 'person_id', 'earliest_visit_date',
                          'max_event_date', 'orders', 'dates', 'ages', 'concept_ids',
                          'visit_segments', 'visit_concept_orders', 'num_of_visits',
                          'num_of_concepts']

    if include_visit_type:
        patient_grouped_events = patient_grouped_events \
            .withColumn('visit_concept_ids', F.col('date_concept_id_period.visit_concept_id'))
        columns_for_output.append('visit_concept_ids')

    return patient_grouped_events.select(columns_for_output)


def create_sequence_data_with_att(patient_event, date_filter=None,
                                  include_visit_type=False,
                                  exclude_visit_tokens=False):
    """
    Create a sequence of the events associated with one patient in a chronological order

    :param patient_event:
    :param date_filter:
    :param include_visit_type:
    :param exclude_visit_tokens:
    :return:
    """
    if date_filter:
        patient_event = patient_event.where(F.col('date').cast('date') >= date_filter)

    # Udf for identifying the earliest date associated with a visit_occurrence_id
    visit_start_date_udf = F.first('date').over(
        W.partitionBy('cohort_member_id', 'person_id', 'visit_occurrence_id').orderBy('date'))

    # Udf for identifying the latest date associated with a visit_occurrence_id
    visit_end_date_udf = F.first('date').over(
        W.partitionBy('cohort_member_id', 'person_id', 'visit_occurrence_id').orderBy(
            F.col('date').desc()))

    visit_start_events = patient_event.withColumn('date', visit_start_date_udf) \
        .withColumn('standard_concept_id', F.lit('VS')) \
        .withColumn('domain', F.lit('visit')).distinct()

    visit_end_events = patient_event.withColumn('date', visit_end_date_udf) \
        .withColumn('standard_concept_id', F.lit('VE')) \
        .withColumn('domain', F.lit('visit')).distinct()

    # Calculate the priority, VS has the highest priority, regular concepts have 0 priority
    # and VE has the lowest priority
    priority_udf = F.when(F.col('standard_concept_id') == 'VS', -1).when(
        F.col('standard_concept_id') == 'VE', 1).otherwise(0)
    # Convert Date to days since epoch
    days_since_epoch_udf = (F.unix_timestamp('date') / F.lit(24 * 60 * 60)).cast('int')
    weeks_since_epoch_udf = (F.unix_timestamp('date') / F.lit(24 * 60 * 60 * 7)).cast('int')
    # Get the prev days_since_epoch
    prev_days_since_epoch_udf = F.lag('days_since_epoch').over(
        W.partitionBy('cohort_member_id', 'person_id').orderBy('date', 'priority',
                                                               'visit_occurrence_id'))
    # Compute the time difference between the current record and the previous record
    time_delta_udf = F.when(F.col('prev_days_since_epoch').isNull(), 0).otherwise(
        F.col('days_since_epoch') - F.col('prev_days_since_epoch'))
    visit_date_udf = F.first('days_since_epoch').over(
        W.partitionBy('cohort_member_id', 'person_id', 'visit_occurrence_id').orderBy(
            'days_since_epoch'))
    # Udf for calculating the time token
    time_token_udf = F.udf(time_token_func, T.StringType())
    visit_rank_udf = F.dense_rank().over(
        W.partitionBy('cohort_member_id', 'person_id').orderBy('visit_start_date'))
    visit_segment_udf = F.col('visit_rank_order') % F.lit(2) + 1

    patient_event = patient_event.union(visit_start_events).union(visit_end_events) \
        .withColumn('priority', priority_udf) \
        .withColumn('days_since_epoch', days_since_epoch_udf) \
        .withColumn('prev_days_since_epoch', prev_days_since_epoch_udf) \
        .withColumn('date_in_week', weeks_since_epoch_udf) \
        .withColumn('time_delta', time_delta_udf) \
        .withColumn('visit_start_date', visit_date_udf) \
        .withColumn('time_token', time_token_udf('time_delta')) \
        .withColumn('visit_rank_order', visit_rank_udf) \
        .withColumn('visit_segment', visit_segment_udf)

    time_token_insertions = patient_event.where('standard_concept_id = "VS"') \
        .withColumn('standard_concept_id', F.col('time_token')) \
        .withColumn('priority', F.lit(-2)) \
        .withColumn('visit_segment', F.lit(0)) \
        .withColumn('date_in_week', F.lit(0)) \
        .withColumn('age', F.lit(-1)) \
        .where('prev_days_since_epoch IS NOT NULL')

    if include_visit_type:
        time_token_insertions = time_token_insertions.withColumn('visit_concept_id', F.lit(0))

    unioned_distinct_tokens = patient_event.union(time_token_insertions).distinct()

    order_udf = F.row_number().over(
        W.partitionBy('cohort_member_id', 'person_id').orderBy('visit_start_date',
                                                               'visit_occurrence_id',
                                                               'priority',
                                                               'days_since_epoch',
                                                               'standard_concept_id'))

    struct_columns = ['order', 'date_in_week', 'standard_concept_id', 'visit_segment', 'age',
                      'visit_rank_order']
    output_columns = ['cohort_member_id', 'person_id', 'concept_ids', 'visit_segments', 'orders',
                      'dates', 'ages', 'visit_concept_orders', 'num_of_visits', 'num_of_concepts']

    if include_visit_type:
        struct_columns.append('visit_concept_id')

    if exclude_visit_tokens:
        unioned_distinct_tokens = unioned_distinct_tokens.filter(
            ~F.col('standard_concept_id').isin(['VS', 'VE']))

    patient_grouped_events = unioned_distinct_tokens \
        .withColumn('order', order_udf) \
        .withColumn('data_for_sorting', F.struct(struct_columns)) \
        .groupBy('cohort_member_id', 'person_id') \
        .agg(F.sort_array(F.collect_set('data_for_sorting')).alias('data_for_sorting'),
             F.max('visit_rank_order').alias('num_of_visits'),
             F.count('standard_concept_id').alias('num_of_concepts')) \
        .withColumn('orders',
                    F.col('data_for_sorting.order').cast(T.ArrayType(T.IntegerType()))) \
        .withColumn('dates', F.col('data_for_sorting.date_in_week')) \
        .withColumn('concept_ids', F.col('data_for_sorting.standard_concept_id')) \
        .withColumn('visit_segments', F.col('data_for_sorting.visit_segment')) \
        .withColumn('ages', F.col('data_for_sorting.age')) \
        .withColumn('visit_concept_orders', F.col('data_for_sorting.visit_rank_order'))

    # If include_visit_type is enabled, we add additional information to the default output
    if include_visit_type:
        patient_grouped_events = patient_grouped_events \
            .withColumn('visit_concept_ids', F.col('data_for_sorting.visit_concept_id'))
        output_columns.append('visit_concept_ids')

    return patient_grouped_events.select(output_columns)


def create_concept_frequency_data(patient_event, date_filter=None):
    if date_filter:
        patient_event = patient_event.where(F.col('date') >= date_filter)

    take_concept_ids_udf = F.udf(lambda rows: [row[0] for row in rows], T.ArrayType(T.StringType()))
    take_freqs_udf = F.udf(lambda rows: [row[1] for row in rows], T.ArrayType(T.IntegerType()))

    patient_event = patient_event.groupBy('cohort_member_id', 'person_id',
                                          'standard_concept_id').count() \
        .withColumn('concept_id_freq', F.struct('standard_concept_id', 'count')) \
        .groupBy('cohort_member_id', 'person_id').agg(
        F.collect_list('concept_id_freq').alias('sequence')) \
        .withColumn('concept_ids', take_concept_ids_udf('sequence')) \
        .withColumn('frequencies', take_freqs_udf('sequence')) \
        .select('cohort_member_id', 'person_id', 'concept_ids', 'frequencies')

    return patient_event


def create_retain_data(patient_event):
    import pandas as pd
    from tensorflow.python.keras.preprocessing.text import Tokenizer

    training_data = patient_event.where('visit_occurrence_id IS NOT NULL').distinct() \
        .groupby('person_id', 'visit_occurrence_id') \
        .agg(F.collect_list(F.col('standard_concept_id').cast('string')).alias('concept_ids')) \
        .groupby('person_id').agg(F.collect_list('concept_ids').alias('visit_concept_ids'))

    tokenizer = Tokenizer(filters='', lower=False)
    training_data_pd = training_data.toPandas()
    tokenizer.fit_on_texts(training_data_pd['visit_concept_ids'].explode())

    training_data_pd = training_data_pd.set_index('person_id')
    person_ids = training_data_pd['visit_concept_ids'].explode().index
    token_ids = tokenizer.texts_to_sequences(training_data_pd['visit_concept_ids'].explode())
    tokenized_training_data_pd = pd.DataFrame(zip(person_ids, token_ids),
                                              columns=['person_id', 'token_ids'])
    retain_training_data = tokenized_training_data_pd.groupby('person_id')['token_ids'].apply(
        list).reset_index()
    return retain_training_data


def extract_ehr_records(spark, input_folder, domain_table_list, include_visit_type=False,
                        with_rollup=False):
    """
    Extract the ehr records for domain_table_list from input_folder.

    :param spark:
    :param input_folder:
    :param domain_table_list:
    :param include_visit_type: whether or not to include the visit type to the ehr records
    :param with_rollup: whether ot not to roll up the concepts to the parent levels
    :return:
    """
    domain_tables = []
    for domain_table_name in domain_table_list:
        domain_tables.append(
            preprocess_domain_table(spark, input_folder, domain_table_name, with_rollup))
    patient_ehr_records = join_domain_tables(domain_tables)
    patient_ehr_records = patient_ehr_records.where('visit_occurrence_id IS NOT NULL').distinct()
    person = preprocess_domain_table(spark, input_folder, PERSON)
    patient_ehr_records = patient_ehr_records.join(person, 'person_id') \
        .withColumn('age',
                    F.ceil(F.months_between(F.col('date'), F.col("birth_datetime")) / F.lit(12)))
    if include_visit_type:
        visit_occurrence = preprocess_domain_table(spark, input_folder, VISIT_OCCURRENCE)
        patient_ehr_records = patient_ehr_records.join(visit_occurrence, 'visit_occurrence_id') \
            .select(patient_ehr_records['person_id'], patient_ehr_records['standard_concept_id'],
                    patient_ehr_records['date'], patient_ehr_records['visit_occurrence_id'],
                    patient_ehr_records['domain'], visit_occurrence['visit_concept_id'],
                    patient_ehr_records['age'])

    return patient_ehr_records


def build_ancestry_table_for(spark, concept_ids):
    initial_query = """
    SELECT
        cr.concept_id_1 AS ancestor_concept_id,
        cr.concept_id_2 AS descendant_concept_id,
        1 AS distance 
    FROM global_temp.concept_relationship AS cr
    WHERE cr.concept_id_1 in ({concept_ids}) AND cr.relationship_id = 'Subsumes'
    """

    recurring_query = """
    SELECT
        i.ancestor_concept_id AS ancestor_concept_id,
        cr.concept_id_2 AS descendant_concept_id,
        i.distance + 1 AS distance
    FROM global_temp.ancestry_table AS i
    JOIN global_temp.concept_relationship AS cr
        ON i.descendant_concept_id = cr.concept_id_1 AND cr.relationship_id = 'Subsumes'
    LEFT JOIN global_temp.ancestry_table AS i2
        ON cr.concept_id_2 = i2.descendant_concept_id
    WHERE i2.descendant_concept_id IS NULL
    """

    union_query = """
    SELECT
        *
    FROM global_temp.ancestry_table

    UNION 

    SELECT
        *
    FROM global_temp.candidate
    """

    ancestry_table = spark.sql(
        initial_query.format(concept_ids=','.join([str(c) for c in concept_ids])))
    ancestry_table.createOrReplaceGlobalTempView('ancestry_table')

    candidate_set = spark.sql(recurring_query)
    candidate_set.createOrReplaceGlobalTempView('candidate')

    while candidate_set.count() != 0:
        spark.sql(union_query).createOrReplaceGlobalTempView('ancestry_table')
        candidate_set = spark.sql(recurring_query)
        candidate_set.createOrReplaceGlobalTempView('candidate')

    ancestry_table = spark.sql("""
    SELECT 
        *
    FROM global_temp.ancestry_table
    """)

    spark.sql("""
    DROP VIEW global_temp.ancestry_table
    """)

    return ancestry_table


def get_descendant_concept_ids(spark, concept_ids):
    """
    Query concept_ancestor table to get all descendant_concept_ids for the given list of concept_ids
    :param spark:
    :param concept_ids:
    :return:
    """
    descendant_concept_ids = spark.sql("""
        SELECT DISTINCT
            c.*
        FROM global_temp.concept_ancestor AS ca
        JOIN global_temp.concept AS c 
            ON ca.descendant_concept_id = c.concept_id
        WHERE ca.ancestor_concept_id IN ({concept_ids})
    """.format(concept_ids=','.join([str(c) for c in concept_ids])))
    return descendant_concept_ids


def get_standard_concept_ids(spark, concept_ids):
    standard_concept_ids = spark.sql("""
            SELECT DISTINCT
                c.*
            FROM global_temp.concept_relationship AS cr
            JOIN global_temp.concept AS c 
                ON ca.concept_id_2 = c.concept_id AND cr.relationship_id = 'Maps to'
            WHERE ca.concept_id_1 IN ({concept_ids})
        """.format(concept_ids=','.join([str(c) for c in concept_ids])))
    return standard_concept_ids
