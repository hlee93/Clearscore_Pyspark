from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import desc
from pyspark.sql.window import Window
import json
import tempfile
import os
import uuid

#Initiate spark session.
sc = SparkSession.builder.appName('clear_score').getOrCreate()

def create_spark_df(path):
    """
    Recursively look through all folders for json files, then load them in as a single dataframe.

    :param path: Path to data folders.
    :type path: String.
    :return: Dataframe of all json files found.
    :rtype: PySpark Dataframe.
    """
    return sc.read.option('recursiveFileLookup', 'true').json(path)

def get_most_recent_reports(df):
    """
    Filter dataframe to find most recent reports for each user.

    :param df: Dataframe with users reports.
    :type df: PySpark Dataframe.
    :return: Dataframe of most recent report for each user.
    :rtype: PySpark Dataframe.
    """
    window_spec = Window.partitionBy('user-uuid').orderBy(F.col('pulled-timestamp').desc())
    #Add a rank column based on the window specification
    ranked_df = df.withColumn("rank", F.rank().over(window_spec))
    #Filter for entries with rank 1 (most recent entries)
    most_recent_reports_df = ranked_df.filter(F.col("rank") == 1).drop('rank')
    return most_recent_reports_df
    
def get_scores(df_reports):
    """
    Pull credit scores for users from each report entry.

    :param df_reports: Dataframe with users reports.
    :type df_reports: PySpark Dataframe.
    :return: Dataframe with an additional column containing users credit scores.
    :rtype: PySpark Dataframe.
    """
    df = df_reports.withColumn('score', F.col('report.ScoreBlock.Delphi.Score'))
    return df.withColumn('score', F.expr('CAST(score[0] AS INT)'))

def get_employment_status(accounts_df):
    """
    Pull employment status from each user account entry.

    :param accounts_df: Dataframe with users account information.
    :type accounts_df: PySpark Dataframe.
    :return: Dataframe with an additional column containing users employment status.
    :rtype: PySpark Dataframe.
    """
    return accounts_df.withColumn('employment_status', F.col('account.user.employmentStatus'))
    
def calc_ranges(credit_score_df):  
    """
    Create credit score ranges using the maximum credit score available in the dataframe.
    These range brackets go up in increments of 50.

    :param credit_score_df: Dataframe with users credit scores.
    :type credit_score_df: PySpark Dataframe.
    :return: Integer, Integer, Tuple.
    :rtype: PySpark Dataframe.
    """
    max_score = credit_score_df.select(F.max(F.col('score'))).collect()[0][0]
    #create score ranges
    start = 0
    #Calculate end value. Ensures that value is rounded to next 50 value.
    end =  50 * round(max_score/50)
    step = 50
    #Create tuples containing ranges with intervals of 50.
    ranges = [(i + 1, i + step ) for i in range(start, end, step)]
    ranges[0] = (0 , 50)
    return start, end, ranges  

def get_score_range_count(df_scores):
    """
    Pull credit scores for users from each report entry.

    :param df_scores: Dataframe with users credit scores and reports.
    :type df_scores: PySpark Dataframe.
    :return: Dataframe divided into credit score ranges. The count of users which fall into each range is displayed below.
    :rtype: PySpark Dataframe.
    """
    start, end, ranges = calc_ranges(df_scores)
    #Initialize a list to store aggregation expressions
    agg_exprs = []
    #Create aggregation expressions for each range
    for start, end in ranges:
        agg_expr = F.sum(F.when((F.col('score') >= start) & (F.col('score') <= end), 1).otherwise(0)).alias(f'{start}-{end}')
        agg_exprs.append(agg_expr)
    #Group by and aggregate data into a DataFrame
    return df_scores.groupBy().agg(*agg_exprs)


def create_enriched_df(most_recent_reports_df, df_accounts_employment):
    """
    Creates a combined dataframe which conatins both account and report information on users.
    This dataframe uses the most recent reports for users.

    :param most_recent_reports_df: The most recent credit reports for each user in a dataframe.
    :type most_recent_reports_df: PySpark Dataframe.
    :param df_accounts_employment: Account information for each user in a dataframe.
    :type df_accounts_employment: PySpark Dataframe.
    :return: A dataframe containing bank data, report and account information for users.
    :rtype: PySpark Dataframe.
    """
    cols_to_drop = ['report-id', 'account-id', 'score']
    most_recent_reports_df = most_recent_reports_df.withColumn('number_of_active_bank_accounts', F.col('report.Summary.Payment_Profiles.CPA.Bank.Total_number_of_Bank_Active_accounts_'))
    most_recent_reports_df = most_recent_reports_df.withColumn('total_outstanding_balance', F.col('report.Summary.Payment_Profiles.CPA.Bank.Total_outstanding_balance_on_Bank_active_accounts'))
    #Drop as many unnecessary columns as possible for memory performance
    most_recent_reports_df = most_recent_reports_df.drop(*cols_to_drop)
    enriched_df = most_recent_reports_df.join(df_accounts_employment, on='user-uuid', how='inner')
    enriched_df = enriched_df.withColumn('bank_name', F.col('account.user.bankName'))
    return enriched_df.drop(*['report', 'account', 'accountId', 'createdTimestamp'])

def create_temp_folder():
    """
    Create a temporary local folder for storing the resultant pyspark dataframes as csvs. 

    :return: The path of the newly created folder.
    :rtype: String.
    """
    temp = tempfile.mkdtemp()
    temp_path = os.path.abspath(temp)
    return temp_path

def df_to_csv(df, filename, temp_path):
    """
    Convert pyspark dataframes to csvs.
    Prints the location they were saved to.
    
    :param df: Dataframe required as a csv.
    :type df: PySpark Dataframe.
    :param filename: Chosen name for the csv to be saved under.
    :type filename: String.
    :param temp_path: Path of the temporary folder that the csv will be saved under.
    :type temp_path: string.
    :return: none.
    """
    #coalesce to 1 ensures whole dataframe saved as one csv.
    #Can increase this in event of bigger dataframes.
    df = df.coalesce(1)
    full_file_path = os.path.join(temp_path, filename)
    df.write.csv(full_file_path, header=True)
    print(f'File has been written to "{temp_path}"')
    
def stop_spark():
    sc.stop()