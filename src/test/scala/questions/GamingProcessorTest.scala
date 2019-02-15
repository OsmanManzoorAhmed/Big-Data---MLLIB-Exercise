package questions

import org.apache.spark.sql.DataFrame
import org.apache.spark.SparkContext
import org.apache.spark.sql.Row
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.types._

import org.apache.log4j.Logger
import org.apache.log4j.Level

import org.scalatest._
import Matchers._
import org.scalatest.concurrent.TimeLimitedTests
import org.scalatest.concurrent.Interruptor
import org.scalatest.time.SpanSugar._


class GamingProcessorTest extends FlatSpec with GivenWhenThen with AppendedClues with TimeLimitedTests {

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  override val defaultTestInterruptor = new Interruptor {
    override def apply(testThread: Thread): Unit = {
      testThread.stop()
    }
  }

  val processor = new GamingProcessor()
  val path = getClass().getResource("/sample.data").toString.drop(5)
  val parquetPath = processor.spark.read.parquet(getClass().getResource("/sample.parquet/").toString().drop(5))

  def timeLimit = 200 seconds

  val columns = StructType(Array(StructField("gender",DoubleType,false),
  StructField("age",DoubleType,false),
  StructField("country",StringType,true),
  StructField("friend_count",DoubleType,false),
  StructField("lifetime",DoubleType,false),
  StructField("game1",DoubleType,false),
  StructField("game2",DoubleType,false),
  StructField("game3",DoubleType,false),
  StructField("game4",DoubleType,false),
  StructField("paid_customer",DoubleType,false)))

  "convert" should "load the data and return DataFrame" in {
    val attempt = processor.convert(path)


    attempt.count() should equal (50) withClue("wrong number of rows.")

    attempt.schema.fields.map(x=>(x.name,x.dataType)) should
    be (columns.map(x=>(x.name,x.dataType))) withClue("wrong schema")


  }

  "indexer" should "transform categorical features" in {
    
    val attempt = processor.indexer(parquetPath)

    attempt.count() should equal (50) withClue("wrong number of rows.")

    attempt.schema.fields.zip(columns).foreach{x =>
      x._1.name should equal(x._2.name) withClue("wrong name")
      x._1.dataType should be (x._2.dataType) withClue("wrong type")}

  }
}