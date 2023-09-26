import pytest
from pyspark.sql import SparkSession
import clearscore_calculations

#Test JSON files
account_example = 'clearscore_test/account_example.json'
report_example = 'clearscore_test/report_example.json'

#Create spark fixture to ensure spark sessions isolated.
@pytest.fixture(scope="session")
def spark():
    sc = (
        SparkSession.builder.master("local[1]")
        .appName("local-tests")
        .config("spark.executor.cores", "1")
        .config("spark.executor.instances", "1")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    yield sc
    sc.stop()
    
def test_create_df(spark):
    #check that both dataframes created from json are not empty
    report_df = clearscore_calculations.create_spark_df(report_example)
    account_df = clearscore_calculations.create_spark_df(account_example)
    assert not report_df.isEmpty()
    assert not account_df.isEmpty()
    
def test_extract_score(spark):
    #test that score is being pulled as correct value
    report_df = clearscore_calculations.create_spark_df(report_example)
    score_df = clearscore_calculations.get_scores(report_df)
    assert score_df.select('score').first()[0] == 638
    
def test_employment_status(spark):
    #test that correct value is pulled for employment status
    account_df = clearscore_calculations.create_spark_df(account_example)
    employment_df = clearscore_calculations.get_employment_status(account_df)
    assert employment_df.select('employment_status').first()[0] == 'FT_EMPLOYED'
    
    
    

