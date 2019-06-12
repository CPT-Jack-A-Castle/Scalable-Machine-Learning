import pyspark
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, Binarizer
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import time, threading, argparse
from pyspark.mllib.classification import LogisticRegressionModel

parser  = argparse.ArgumentParser()
parser.add_argument("-s", "--sub-question", help="Sub question number", dest="que", type=int)

args = parser.parse_args()

spark = SparkSession.builder.master("local[20]").config("spark.local.dir", "/fastdata/xxxxxxx"). appName("COM 6012: Assignment 2 - Q1").getOrCreate()
sc = spark.sparkContext

data = spark.read.option("header", "false").csv("Data/HIGGS.csv.gz").cache()

schemaNames = data.schema.names

for i in range(len(data.columns)):
	data = data.withColumn(schemaNames[i], data[schemaNames[i]].cast(DoubleType()))
data = data.withColumnRenamed('_c0', 'label')
n_col = len(data.columns)
assembler = VectorAssembler(inputCols = schemaNames[1:], outputCol = 'features')

data_vec = assembler.transform(data)

data_vec = data_vec.select("features", "label")

(small_dataset, _) = data_vec.randomSplit([0.25, 0.75], 9008)
(s_train, s_test) = small_dataset.randomSplit([0.7, 0.3], 9008)
(f_train, f_test) = data_vec.randomSplit([0.7, 0.3], 9008)

def thread_full_classification(n_col, train, test, pipeline, evaluator, algo_name, metric_name, z):
	col_names = ["lepton_pT","lepton_eta","lepton_phi","missing_energy_magnitude","missing_energy_phi","jet_1_pt","jet_1_eta","jet_1_phi","jet_1_b-tag","jet_2_pt","jet_2_eta","jet_2_phi","jet_2_b-tag","jet_3_pt","jet_3_eta","jet_3_phi","jet_3_b-tag","jet_4_pt","jet_4_eta","jet_4_phi","jet_4_b-tag","m_jj","m_jjj","m_lv","m_jlv","m_bb","m_wbb","m_wwbb"]
	start_time = time.time()
	print("[*] Starting training of full dataset with %s" %algo_name)

	model = pipeline.fit(train)
	print("[*] Done training for algorithm: %s with metric: %s. Training time: %f\n[*] Testing and making predictions" %(algo_name, metric_name, time.time() - start_time))
	prediction = model.transform(test)

	metric_val = evaluator.evaluate(prediction)
	print("[*] Done evaluating data and making predictions.%s: %s: %f" %(algo_name, metric_name, metric_val))

	if z != 2:
		fi = model.stages[0].featureImportances
		imp_feat = np.zeros(n_col)
		imp_feat[fi.indices] = fi.values
	
		print("[*] %s most important features" %(algo_name))
		for i in imp_feat.argsort()[-3:][::-1]:
			print("[*] - %s" %col_names[i])
	else:
		fi = model.stages[0].coefficients.values
		for i in np.abs(fi).argsort()[-3:][::-1]:
			print("[*] - %s" %col_names[i])

			
def run_metric(train, test, pipeline, parambuilder,evaluator, numFolds=3):
	crossval = CrossValidator(estimator=pipeline, estimatorParamMaps=parambuilder, evaluator=evaluator, numFolds=numFolds)
	cvModel = crossval.fit(train)
	bestParams = cvModel.getEstimatorParamMaps()[np.argmax(cvModel.avgMetrics)]
	predictions = cvModel.transform(test)
	return bestParams, evaluator.evaluate(predictions)

def que1():
	for i,ct in enumerate([DecisionTreeClassifier(seed=9008), DecisionTreeRegressor(predictionCol="prediction_c", seed=9008), LogisticRegression()]):
		binarizer = None
		if i == 0:
			print("[*] DecisionTree Classifier")
			paramB = ParamGridBuilder().addGrid(ct.maxDepth, [5, 10, 20]).addGrid(ct.maxBins, [16, 32]).addGrid(ct.impurity, ["gini", "entropy"]).build()
			continue
		elif i == 1:
			print("[*] DecisionTree Regressor")
			paramB = ParamGridBuilder().addGrid(ct.maxDepth, [5, 10, 20]).addGrid(ct.maxBins, [16, 32]).addGrid(ct.minInfoGain, [0.0, 0.25, 0.3]).build()
			binarizer = Binarizer(threshold=0.5, inputCol="prediction_c", outputCol="prediction") 
		else:
			print("[*] Logistic Regression")
			paramB = ParamGridBuilder().addGrid(ct.maxIter, [5, 10, 15]).addGrid(ct.regParam, [0.05, 0.1, 0.5]).build()
	
		if binarizer is not None: pipeline = Pipeline(stages=[ct, binarizer])
		else: pipeline = Pipeline(stages=[ct])

		print("[*] Running for areaUnderROC")
		bp, metric_roc = run_metric(s_train, s_test, pipeline, paramB, BinaryClassificationEvaluator(rawPredictionCol = "prediction", metricName="areaUnderROC"))
		print("[*] Done for areaUnderROC")
		print("[*] Best Params: %s, AreaUnderROC value: %f" %(bp, metric_roc))

		print("[*] Running for accuracy")
		mp, metric_acc = run_metric(s_train, s_test, pipeline, paramB, MulticlassClassificationEvaluator(predictionCol = "prediction", metricName="accuracy"))
		print("[*] Done for accuracy")
		print("[*] Best Params: %s, Accuracy value: %f" %(mp, metric_acc))



def que2():
	algo_name = ["Decision Tree Classifier", "Decision Tree Regressor", "Logistic Regression"]
	for i,ct in enumerate([DecisionTreeClassifier(seed=9008, maxDepth=10, maxBins=16, impurity="gini"), DecisionTreeRegressor(predictionCol="prediction_c", seed=9008, maxDepth=10, maxBins=16, minInfoGain=0.0), LogisticRegression(regParam=0.05, maxIter=15)]):
		if i == 1:
			pipeline = Pipeline(stages=[ct, Binarizer(threshold=0.5, inputCol="prediction_c", outputCol="prediction")]) 
		else:
			pipeline = Pipeline(stages=[ct])
		
		thread_full_classification(n_col, f_train, f_test, pipeline, BinaryClassificationEvaluator(rawPredictionCol = "prediction", metricName="areaUnderROC"), algo_name[i], "areaUnderROC", i)
		thread_full_classification(n_col, f_train, f_test, pipeline, MulticlassClassificationEvaluator(predictionCol = "prediction", metricName="accuracy"), algo_name[i], "accuracy", i)

			

if args.que == 1:
	que1()

elif args.que == 2:
	que2()
else:
	que1()
	que2()
