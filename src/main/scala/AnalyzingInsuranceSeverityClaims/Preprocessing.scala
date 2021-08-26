package AnalyzingInsuranceSeverityClaims

import org.apache.spark.ml.feature.{ StringIndexer, StringIndexerModel};
import org.apache.spark.ml.feature.VectorAssembler;

object Preprocessing{
  System.setProperty("hadoop.home.dir", "C:/Users/mmoustafa/workspace/Scala-Machine-Learning-Projects/bins/winutils_bin/");

  var trainSample = 1.0;
  var testSample = 1.0;

  val train = "data/allstate_claims_severity/insurance_train.csv";
  val test = "data/allstate_claims_severity/insurance_test.csv";

  val spark = SparkSessionCreate.createSession();

  import spark.implicits._;

  println("Reading data from " + train + " file");

  val trainInput = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .format("com.databricks.spark.csv")
    .load(train)
    .cache;

  val testInput = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .format("com.databricks.spark.csv")
    .load(test)
    .cache;

  println("Preparing data for training model");
  var data = trainInput
    .withColumnRenamed("loss", "label")
    .sample(false, trainSample);

  var DF = data.na.drop();

  if (data == DF) {
    println("No null values in the DataFrame");
  } else{
    println("Null values exist in the DataFrame");
    data = DF;
  }

  val seed = 12345L;
  val splits = data.randomSplit(Array(0.75, 0.25), seed);
  val (trainingData, validationData) = (splits(0), splits(1));

  trainingData.cache();
  validationData.cache();

  val testData = testInput.sample(false, testSample).cache;

  def isCateg(c: String): Boolean = {
    c.startsWith("cat");
  }

  def categNewCol(c: String): String = {
    if (isCateg(c)) s"idx_${c}" else c;
  }

  def removeTooManyCategs(c: String): Boolean = {
    !(c matches "cat(109$|110$|112$|113$|116$)");
  }

  def onlyFeatureCols(c: String): Boolean = !(c matches "id|label");

  val featureCols = trainingData.columns
    .filter(removeTooManyCategs)
    .filter(onlyFeatureCols)
    .map(categNewCol);

  val stringIndexerStages = trainingData.columns.filter(isCateg)
    .map(
      c => new StringIndexer()
        .setInputCol(c)
        .setOutputCol(categNewCol(c))
        .fit(trainInput.select(c).union(testInput.select(c)))
    );

  val assembler = new VectorAssembler()
    .setInputCols(featureCols)
    .setOutputCol("features");

  def main(args: Array[String]): Unit ={}
}