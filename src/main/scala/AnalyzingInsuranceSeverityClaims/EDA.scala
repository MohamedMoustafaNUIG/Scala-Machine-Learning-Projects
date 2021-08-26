package AnalyzingInsuranceSeverityClaims

import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession};

object EDA {
  System.setProperty("hadoop.home.dir", "C:/Users/mmoustafa/workspace/Scala-Machine-Learning-Projects/bins/winutils_bin/");
  val train = "data/allstate_claims_severity/insurance_train.csv";

  def main(args: Array[String]) ={
    val spark = SparkSessionCreate.createSession();

    val trainInput = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .format("com.databricks.spark.csv")
      .load(train)
      .cache
    //println(trainInput.printSchema());
    //println(trainInput.count());
    /*
    trainInput
      .select(
        "id",
        "cat1",
        "cat2",
        "cat3",
        "cont1",
        "cont2",
        "cont3",
        "loss")
      .show()
    */
    //trainInput.select("cat109", "cat110", "cat112", "cat113", "cat116").show()

    val newDF = trainInput.withColumnRenamed("loss", "label");
    newDF.createOrReplaceTempView("insurance");

    /*
    var sqlCommand = "SELECT avg(insurance.label) as AVG_LOSS FROM insurance";
    spark.sql(sqlCommand).show();
    sqlCommand = "SELECT min(insurance.label) as MIN_LOSS FROM insurance";
    spark.sql(sqlCommand).show();
    sqlCommand = "SELECT max(insurance.label) as MAX_LOSS FROM insurance";
    spark.sql(sqlCommand).show();
    */

    /*
    def TransposeDF(df: DataFrame, columns: Seq[String], pivotCol: String): DataFrame = {
      val columnsValue = columns.map(x => "'" + x + "', " + x)
      val stackCols = columnsValue.mkString(",")
      val df_1 = df.selectExpr(pivotCol, "stack(" + columns.size + "," + stackCols + ")")
        .select(pivotCol, "col0", "col1")

      val final_df = df_1.groupBy(col("col0")).pivot(pivotCol).agg(concat_ws("", collect_list(col("col1"))))
        .withColumnRenamed("col0", pivotCol)
      final_df
    };

    def isCateg(c: String): Boolean = {
      c.startsWith("cat");
    };

    var catCols: Array[String] = newDF.columns.filter(c => isCateg(c)) :+ "id" ;

    var countOcc: DataFrame =
      newDF
        .select(newDF.columns.map(c => countDistinct(col(c)).alias(c)): _*)
        .select(catCols.map(c=>col(c)): _*);

    TransposeDF(countOcc, countOcc.columns,"id").sort("id").show(catCols.length+1);
    */

    //newDF.select(newDF.columns.map(c => countDistinct(col(c)).alias(c)): _*).show();

  }
}