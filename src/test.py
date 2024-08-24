# Set-ExecutionPolicy Unrestricted
# .\ds_tests\Scripts\activate

import os
from pathlib import Path

from pyspark.sql import SparkSession

os.environ['SPARK_HOME'] = "D:/Documentos/spark/"
os.environ['HADOOP_HOME'] = "D:/Documentos/spark/hadoop"
os.environ['JAVA_HOME'] = "C:/Program Files/Java/jdk-21/"

spark = SparkSession.builder.appName("Testes").getOrCreate()

spark.conf.set("spark.sql.session.timeZone", "UTC+0")
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

caminho = Path("D:\\Documentos\\DS_tests\\data\\Iris.csv")

df = spark.read.csv(caminho.__str__(), sep=",")