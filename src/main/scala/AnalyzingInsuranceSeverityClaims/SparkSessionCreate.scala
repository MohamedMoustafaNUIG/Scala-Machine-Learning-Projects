package AnalyzingInsuranceSeverityClaims

import org.apache.spark.sql.SparkSession

object SparkSessionCreate {
  def createSession(): SparkSession = {
    val spark = SparkSession
      .builder
      .master("local[*]") // adjust accordingly
      .config("spark.sql.warehouse.dir", "E:/Exp/") //change accordingly
      .appName("MySparkSession") //change accordingly
      .getOrCreate()
    return spark;
  }
}
