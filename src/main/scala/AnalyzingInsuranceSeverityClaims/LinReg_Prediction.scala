package AnalyzingInsuranceSeverityClaims

import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel};
import org.apache.spark.ml.{ Pipeline, PipelineModel };
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.sql._;
import org.apache.spark.sql.functions._;
import org.apache.spark.mllib.evaluation.RegressionMetrics;

object LinReg_Prediction {
  System.setProperty("hadoop.home.dir", "C:/Users/mmoustafa/workspace/Scala-Machine-Learning-Projects/bins/winutils_bin/");

  val spark = SparkSessionCreate.createSession();
  import spark.implicits._;

  //hyperparameters
  val numFolds = 10; //for cross-validation
  val MaxIter: Seq[Int] = Seq(1000);
  val RegParam: Seq[Double] = Seq(0.001);
  val Tol: Seq[Double] = Seq(1e-6);
  val ElasticNetParam: Seq[Double] = Seq(0.001);

  def main(args: Array[String]): Unit ={
    val model = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label");

    println("Building ML pipeline")
    val pipeline = new Pipeline()
      .setStages((Preprocessing.stringIndexerStages :+ Preprocessing.assembler) :+ model);

    val paramGrid = new ParamGridBuilder()
      .addGrid(model.maxIter, MaxIter)
      .addGrid(model.regParam, RegParam)
      .addGrid(model.tol, Tol)
      .addGrid(model.elasticNetParam, ElasticNetParam)
      .build();

    println("Preparing K-fold Cross Validation and Grid Search: Model tuning");
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(numFolds);

    println("Training model with Linear Regression algorithm");
    val cvModel = cv.fit(Preprocessing.trainingData);

    println("Evaluating model on train and validation set and calculating RMSE");
    val trainPredictionsAndLabels =
      cvModel.transform(Preprocessing.trainingData)
        .select("label", "prediction")
        .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd;

    val validPredictionsAndLabels =
      cvModel.transform(Preprocessing.validationData)
        .select("label", "prediction")
        .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd;

    val trainRegressionMetrics = new RegressionMetrics(trainPredictionsAndLabels);
    val validRegressionMetrics = new RegressionMetrics(validPredictionsAndLabels);

    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel];

    val results =
      "\n=====================================================================\n" +
        s"Param trainSample: ${Preprocessing.trainSample}\n" +
        s"Param testSample: ${Preprocessing.testSample}\n" +
        s"TrainingData count: ${Preprocessing.trainingData.count}\n" +
        s"ValidationData count: ${Preprocessing.validationData.count}\n" +
        s"TestData count: ${Preprocessing.testData.count}\n" +
        "=====================================================================\n" +
        s"Param maxIter = ${MaxIter.mkString(",")}\n"+
        s"Param numFolds = ${numFolds}\n" +
        "=====================================================================\n" +
        s"Training data MSE = ${trainRegressionMetrics.meanSquaredError}\n" +
        s"Training data RMSE = ${trainRegressionMetrics.rootMeanSquaredError}\n" +
        s"Training data R-squared = ${trainRegressionMetrics.r2}\n" +
        s"Training data MAE = ${trainRegressionMetrics.meanAbsoluteError}\n" +
        s"Training data Explained variance = ${trainRegressionMetrics.explainedVariance}\n" +
        "=====================================================================\n" +
        s"Validation data MSE = ${validRegressionMetrics.meanSquaredError}\n" +
        s"Validation data RMSE = ${validRegressionMetrics.rootMeanSquaredError}\n" +
        s"Validation data R-squared = ${validRegressionMetrics.r2}\n" +
        s"Validation data MAE = ${validRegressionMetrics.meanAbsoluteError}\n" +
        s"Validation data Explained variance = ${validRegressionMetrics.explainedVariance}\n" +
        s"CV params explained: ${cvModel.explainParams}\n" +
        s"LR params explained: ${bestModel.stages.last.asInstanceOf[LinearRegressionModel].explainParams}\n" +
        "=====================================================================\n";

    println(results);

    //Evaluate on test set
    println("Run prediction on the test set")
    cvModel.transform(Preprocessing.testData)
      .select("id", "prediction")
      .withColumnRenamed("prediction", "loss")
      .coalesce(1) // to get all the predictions in a single csv file
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .save("data/allstate_claims_severity/output/result_LR.csv");

    spark.stop();
  }
}
