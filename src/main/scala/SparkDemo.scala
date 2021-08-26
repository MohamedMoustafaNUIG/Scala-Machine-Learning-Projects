import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object SparkDemo {
  System.setProperty("hadoop.home.dir", "C:/Users/mmoustafa/workspace/Scala-Machine-Learning-Projects/bins/winutils_bin/");

  def main(args : Array[String]): Unit ={
    Logger.getRootLogger.setLevel(Level.INFO)
    val sc = new SparkContext("local[*]" , "SparkDemo");
    val lines = sc.textFile("C:/Program Files/Java/jdk1.8.0_261/COPYRIGHT");
    val words = lines.flatMap(line => line.split(' '));
    val wordsKVRdd = words.map(x => (x,1));
    val count =
      wordsKVRdd
        .reduceByKey((x,y) => x + y)
        .map(x => (x._2,x._1))
        .sortByKey(false)
        .map(x => (x._2, x._1))
        .take(10);

    count.foreach(println);
  }
}