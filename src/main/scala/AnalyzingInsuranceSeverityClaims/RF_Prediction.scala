package AnalyzingInsuranceSeverityClaims

import org.apache.spark.ml.regression.{RandomForestRegressor, RandomForestRegressionModel};
import org.apache.spark.ml.{ Pipeline, PipelineModel };
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.sql._;
import org.apache.spark.sql.functions._;
import org.apache.spark.SparkConf;
import org.apache.spark.mllib.evaluation.RegressionMetrics;


object RF_Prediction {
  System.setProperty("hadoop.home.dir", "C:/Users/mmoustafa/workspace/Scala-Machine-Learning-Projects/bins/winutils_bin/");

  val spark = SparkSessionCreate.createSession();
  import spark.implicits._;

  val NumTrees = Seq(5);
  val MaxBins = Seq(23);
  val numFolds = 10;
  val MaxIter: Seq[Int] = Seq(20);
  val MaxDepth: Seq[Int] = Seq(20);

  def main(args: Array[String]): Unit ={
    val model = new RandomForestRegressor().setFeaturesCol("features").setLabelCol("label");

    println("Building ML pipeline");
    val pipeline =
      new Pipeline()
        .setStages((Preprocessing.stringIndexerStages :+ Preprocessing.assembler) :+ model);

    val paramGrid = new ParamGridBuilder()
      .addGrid(model.numTrees, NumTrees)
      .addGrid(model.maxDepth, MaxDepth)
      .addGrid(model.maxBins, MaxBins)
      .build();

    println("Preparing K-fold Cross Validation and Grid Search: Model tuning");
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(numFolds);

    println("Training model with Random Forest algorithm");
    val cvModel = cv.fit(Preprocessing.trainingData);

    println("Evaluating model on train and validation set and calculating RMSE");
    val trainPredictionsAndLabels =
      cvModel
        .transform(Preprocessing.trainingData)
        .select("label","prediction")
        .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd;

    val validPredictionsAndLabels =
      cvModel
        .transform(Preprocessing.validationData)
        .select("label","prediction")
        .map { case Row(label: Double, prediction: Double) => (label, prediction) }.rdd;

    val trainRegressionMetrics = new RegressionMetrics(trainPredictionsAndLabels);
    val validRegressionMetrics = new RegressionMetrics(validPredictionsAndLabels);

    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel];

    val featureImportances = bestModel
      .stages
      .last
      .asInstanceOf[RandomForestRegressionModel].featureImportances.toArray;

    val FI_to_List_sorted = featureImportances.toList.sorted.toArray;

    val output =
      "\n=====================================================================\n" +
        s"Param trainSample: ${Preprocessing.trainSample}\n" +
        s"Param testSample: ${Preprocessing.testSample}\n" +
        s"TrainingData count: ${Preprocessing.trainingData.count}\n" +
        s"ValidationData count: ${Preprocessing.validationData.count}\n" +
        s"TestData count: ${Preprocessing.testData.count}\n" +
        "=====================================================================\n" +
        s"Param maxIter = ${MaxIter.mkString(",")}\n" +
        s"Param maxDepth = ${MaxDepth.mkString(",")}\n" +
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
        "=====================================================================\n" +
        s"CV params explained: ${cvModel.explainParams}\n" +
        s"RF params explained: ${bestModel.stages.last.asInstanceOf[RandomForestRegressionModel].explainParams}\n" +
        s"RF features importances:\n ${Preprocessing.featureCols.zip(FI_to_List_sorted).map(t => s"t${t._1} = ${t._2}").mkString("\n")}\n" +
        "=====================================================================\n";

    println(output);

    //Evaluate on test set
    println("Run prediction on the test set")
    cvModel.transform(Preprocessing.testData)
      .select("id", "prediction")
      .withColumnRenamed("prediction", "loss")
      .coalesce(1) // to get all the predictions in a single csv file
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .save("data/allstate_claims_severity/output/result_RF.csv");

    //saving single model
    bestModel.write.overwrite().save("model/allstate_claims_severity/RF_model");

    //saving pipeline
    cvModel.write.overwrite().save("model/allstate_claims_severity/RF_pipeline");

    spark.stop();
  }
}