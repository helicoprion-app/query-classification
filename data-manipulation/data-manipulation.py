from pymongo import MongoClient
import pandas as pd
import numpy as np
import pickle
import schedule 
import time

def analyse():

  def df_creator(results):

    dataframe = pd.DataFrame(results)
    dataframe = dataframe.rename(columns={"tags":"categories"})

    values = {"categories":"[]", "isSpam":"DOĞRU"}
    dataframe = dataframe.fillna(value=values)

    dataframe["snippet"] = dataframe["snippet"].str.lower()
    dataframe["categories"] = [''.join(map(str, l)) for l in dataframe['categories']]
    dataframe = dataframe.replace(r'^\s*$', np.NaN, regex=True)

    dataframe.drop_duplicates(subset="snippet", inplace=True)
    dataframe.drop_duplicates(subset="relatedLink", inplace=True)

    def isSpam_converter(row):

      if (row["isSpam"] == "DOĞRU"):
        value = "true"
        
      else:
        value = "false"

      return value

    dataframe["isSpam"] = dataframe.apply(lambda row: isSpam_converter(row), axis=1)

    dataframe.drop(columns=["status"], inplace=True)

    dataframe = pd.get_dummies(dataframe, columns=["isSpam"], prefix= ["isSpam"])
    dataframe = pd.get_dummies(dataframe, columns=["categories"], prefix=["categories"])
    dataframe = pd.get_dummies(dataframe, columns=["queryId"], prefix=["queryId"])

    dataframe.drop(columns=["isSpam_false", "categories_[]", "queryId_600196eb1db2e4001734cc78"], axis=1, inplace=True)

    dataframe = dataframe.rename(columns={"isSpam_true":"isSpam"})
    dataframe.dropna(subset=["snippet"], inplace=True)
    dataframe = dataframe.reset_index(drop=True)

    return dataframe

  f = open('models/snippet_classifiers/nb_classifier.pickle', 'rb')
  nb_classifier = pickle.load(f)
  f.close()

  f = open('models/snippet_classifiers/knnclass_classifier.pickle', 'rb')
  knnclass_classifier = pickle.load(f)
  f.close()

  f = open('models/snippet_classifiers/cb_classifier.pickle', 'rb')
  cb_classifier = pickle.load(f)
  f.close()

  f = open('models/snippet_classifiers/logreg_classifier.pickle', 'rb')
  logreg_classifier = pickle.load(f)
  f.close()

  f = open('models/snippet_classifiers/lgb_classifier.pickle', 'rb')
  lgb_classifier = pickle.load(f)
  f.close()

  f = open('models/snippet_classifiers/xgb_classifier.pickle', 'rb')
  xgb_classifier = pickle.load(f)
  f.close()

  f = open('models/snippet_classifiers/lsvm_classifier.pickle', 'rb')
  lsvm_classifier = pickle.load(f)
  f.close()

  f = open('models/admission_classifier/cart.pickle', 'rb')
  cart = pickle.load(f)
  f.close()

  cluster = MongoClient("mongodb+srv://ballkaya:mongodiebisivarmis1*@helicoprion.ov2lt.mongodb.net/helicoprion?retryWrites=true&w=majority")
  db = cluster["helicoprion"]
  collection = db["queryresults"]

  results = collection.find({"projectId": "5bdcd0cffb6fc074abb633ed", 
                              "isPredicted":"0"
                          })

  results = list(results)

  dataframe = df_creator(results)

  df_for_prediction = pd.DataFrame()

  #Model Sütunları
  df_for_prediction["knnclass_prediction"] = knnclass_classifier.predict(dataframe["snippet"])
  df_for_prediction["logreg_prediction"] = logreg_classifier.predict(dataframe["snippet"])
  df_for_prediction["lgb_prediction"] = lgb_classifier.predict(dataframe["snippet"])
  df_for_prediction["cb_prediction"] = cb_classifier.predict(dataframe["snippet"])
  df_for_prediction["nb_prediction"] = nb_classifier.predict(dataframe["snippet"])
  df_for_prediction["lsvm_prediction"] = lsvm_classifier.predict(dataframe["snippet"])
  df_for_prediction["xgb_prediction"] = xgb_classifier.predict(dataframe["snippet"])

  #Spam Sütunu
  df_for_prediction["isSpam"] = dataframe["isSpam"]

  #Kategori Sütunları
  category_list = ["categories_goverment", "categories_education", "categories_news", "categories_social", "categories_sport"]

  for category in category_list:
    df_for_prediction[category] = dataframe[category]

  df_for_prediction.rename(columns={"categories_goverment":"categories_government"}, inplace=True)

  #QueryID Sütunları
  query_id_list = ["queryId_5bf1c4fafb6fc0561ffb75da", "queryId_5bf1cefafb6fc0561ffb79d9", "queryId_5bf1cf9ffb6fc0561ffb7a1a", "queryId_5bf1cfe1fb6fc0561ffb7a37", "queryId_5c23ce4cfb6fc00eee86a59f", "queryId_5c23ce82fb6fc00eee86a5a4"]

  for query_id in query_id_list:
    df_for_prediction[query_id] = dataframe[query_id]

  #WebsiteRank Sütunları
  def website_converter(row):
    if row['websiteLink'] in dataframe["websiteLink"].value_counts()[:51]:
      value = 1

    elif row['websiteLink'] in dataframe["websiteLink"].value_counts()[51:101]:
      value = 2
      
    else:
      value = 0

    return value

  df_for_prediction["website_rank"] = dataframe.apply(website_converter, axis=1)
  df_for_prediction = pd.get_dummies(df_for_prediction, columns=["website_rank"], prefix= ["website_rank"])
  df_for_prediction.drop(columns=["website_rank_0"], axis=1, inplace=True)

  #Karar Sütunu
  predictions = list(cart.predict(df_for_prediction))
  confidence = cart.predict_proba(df_for_prediction).tolist()

  confidence_list = []

  for value in confidence:
    index_of_value = confidence.index(value)

    if predictions[index_of_value] == 1:
      confidence_list.append(str(confidence[index_of_value]).replace("[", "").replace("]", "").split(" ")[1])
    
    else:
      confidence_list.append(str(confidence[index_of_value]).replace("[", "").replace("]", "").split(" ")[0])

  return predictions, confidence_list

def update(predictions, confidence_list):
  cluster = MongoClient("mongodb+srv://ballkaya:mongodiebisivarmis1*@helicoprion.ov2lt.mongodb.net/helicoprion?retryWrites=true&w=majority")
  db = cluster["helicoprion"]
  collection = db["queryresults"]

  results = collection.find({"projectId": "5bdcd0cffb6fc074abb633ed",
                              "isPredicted":"0"
                          })

  results = list(results)

  for result in results:

    index_of_result = results.index(result)

    try:
      collection.update_many({"_id":result["_id"]}, {"$set": {"prediction":str(predictions[index_of_result])}})
      collection.update_many({"_id":result["_id"]}, {"$set": {"confidence":str(confidence_list[index_of_result])}})
      collection.update_many({"_id":result["_id"]}, {"$set": {"isPredicted":"1"}})
    
    except:
      print(index_of_result)

def scheduled_action():
  predictions, confidence_list = analyse()
  update(predictions, confidence_list)

while True:
  schedule.every().day.at("05:00").do(scheduled_action)

  schedule.every().day.at("09:00").do(scheduled_action)
  
  time.sleep(60)