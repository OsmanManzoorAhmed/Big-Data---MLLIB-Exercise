package questions

import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

object Main {

  def main(args: Array[String]) = {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val processor = new GamingProcessor()
    
    // load the data and tranform it
    val data = processor.convert(getClass().getResource("/osge_pool-1-thread-1.data").toString.drop(5))
    val indexed = processor.indexer(data)

    val assembled = processor.featureAssembler(indexed)

    val scaled = processor.scaler(assembled,"scaledFeatures")

    //split the dataset into training(90%) and prediction(10%) sets
    val splitted = scaled.randomSplit(Array(0.9,0.1), 10L)

    //train the model
    val model = processor.createModel(splitted(0),"scaledFeatures","paid_customer","prediction")

    //predict
    val predictions = processor.predict(model, splitted(1))

    //calculate how many correct predictions
    
    val evaluator = new BinaryClassificationEvaluator()
    .setLabelCol("paid_customer")
    .setRawPredictionCol("prediction")

    val res = evaluator.evaluate(predictions)
    
    
    println(Console.YELLOW+"Predicted correctly: " +Console.GREEN+ res*100 + "%")
    
    //finally stop spark
    processor.spark.stop()
  }
}