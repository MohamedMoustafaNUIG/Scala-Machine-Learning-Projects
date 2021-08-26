package AnalyzingInsuranceSeverityClaims

import org.apache.spark.ml.regression.{RandomForestRegressor, RandomForestRegressionModel};
import org.apache.spark.ml.{ Pipeline, PipelineModel };
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.PipelineModel;

import org.apache.spark.sql._;
import org.apache.spark.sql.functions._;
import org.apache.spark.mllib.evaluation.RegressionMetrics;

object Post_Training {
  System.setProperty("hadoop.home.dir", "C:/Users/mmoustafa/workspace/Scala-Machine-Learning-Projects/bins/winutils_bin/");

  val spark = SparkSessionCreate.createSession();
  import spark.implicits._;

  def main(args: Array[String]): Unit ={
    val sameModel = PipelineModel.load("model/allstate_claims_severity/RF_model");
    /*
    sameModel
      .transform(Preprocessing.testData)
      .select("id", "prediction")
      .withColumnRenamed("prediction", "loss")
      .coalesce(1)
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .save("data/allstate_claims_severity/output/result_RF_reuse.csv");
    */
    /*
    OR
    */

    //Then we restore the same model back:
    val sameCV = CrossValidatorModel.load("model/allstate_claims_severity/RF_pipeline");
    sameCV
      .transform(Preprocessing.testData)
      .select("id", "prediction")
      .withColumnRenamed("prediction", "loss")
      .coalesce(1)
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .save("data/allstate_claims_severity/output/result_RF_pipeline_reuse.csv");
  }
}
