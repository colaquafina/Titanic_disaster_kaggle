import pandas as pd


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
allData = pd.concat([train, test], ignore_index=True)

allData["Title"] = allData["Name"].apply(
    lambda x: x.split(",")[1].split(".")[0].strip()
)

TitleClassification = {
    "Officer": ["Capt", "Col", "Major", "Dr", "Rev"],
    "Royalty": ["Don", "Sir", "the Countess", "Dona", "Lady"],
    "Mrs": ["Mme", "Ms", "Mrs"],
    "Miss": ["Mlle", "Miss"],
    "Mr": ["Mr"],
    "Master": ["Master", "Jonkheer"],
}
TitleMap = {}
for title in TitleClassification.keys():
    TitleMap.update(dict.fromkeys(TitleClassification[title], title))

allData["Title"] = allData["Title"].map(TitleMap)  #group the title according to the dictionary TitleClassfication

# Ticket
TicketCnt = allData.groupby(["Ticket"]).size()  #calculate the number of the same tickets
allData["SameTicketNum"] = allData["Ticket"].apply(lambda x: TicketCnt[x])  #map the ticket to the number of the same tickets


# - Sex & Pclass & Embarked --> Ont-Hot
# - Age & Fare --> Standardize
# - FamilySize & Name & Ticket --> ints --> One-Hot

# Sex, change male or female of sex to the 0 or 1 in Sex_male and Sex_female
allData = allData.join(pd.get_dummies(allData["Sex"], prefix="Sex"))   

# Pclass, change Pclass to one-hot
allData = allData.join(pd.get_dummies(allData["Pclass"], prefix="Pclass")) 
# Embarked
allData[allData["Embarked"].isnull()]  
allData.groupby(by=["Pclass", "Embarked"]).Fare.mean()  
allData["Embarked"] = allData["Embarked"].fillna("C")
allData = allData.join(pd.get_dummies(allData["Embarked"], prefix="Embarked"))

# Age
allData["Child"] = allData["Age"].apply(lambda x: 1 if x <= 10 else 0)  # label of children
allData["Age"] = (allData["Age"] - allData["Age"].mean()) / allData["Age"].std()  # Normalization
allData["Age"].fillna(value=0, inplace=True)  # fill the null value

# Fare
allData["Fare"] = allData["Fare"].fillna(25)  # fill the null value
allData[allData["Survived"].notnull()]["Fare"] = allData[allData["Survived"].notnull()][
    "Fare"
].apply(lambda x: 300.0 if x > 500 else x)
allData["Fare"] = allData["Fare"].apply(
    lambda x: (x - allData["Fare"].mean()) / allData["Fare"].std()
)

# Name
TitleLabelMap = {
    "Mr": 1.0,
    "Mrs": 5.0,
    "Miss": 4.5,
    "Master": 2.5,
    "Royalty": 3.5,
    "Officer": 2.0,
}


def TitleLabel(s):
    return TitleLabelMap[s]

# allData['TitleLabel'] = allData['Title'].apply(TitleLabel)
allData = allData.join(pd.get_dummies(allData["Title"], prefix="Title"))  #one-hot

# Ticket
def TicketLabel(s):
    if s == 3 or s == 4:
        return 3
    elif s == 2 or s == 8:
        return 2
    elif s == 1 or s == 5 or s == 6 or s == 7:
        return 1
    elif s < 1 or s > 8:
        return 0


allData["TicketLabel"] = allData["SameTicketNum"].apply(TicketLabel)
allData = allData.join(pd.get_dummies(allData["TicketLabel"], prefix="TicNum"))

allData.drop(
    [
        "Cabin",
        "PassengerId",
        "Ticket",
        "Name",
        "Title",
        "Sex",
        "SibSp",
        "Parch",
        "Embarked",
        "Pclass",
        "Title",
        "SameTicketNum",
        "TicketLabel",
    ],
    axis=1,
    inplace=True,
)




train_data = allData[allData["Survived"].notnull()]
test_data = allData[allData["Survived"].isnull()]
test_data = test_data.reset_index(drop=True)

xTrain = train_data.drop(["Survived"], axis=1)
yTrain = train_data["Survived"]
xTest = test_data.drop(["Survived"], axis=1)


import tensorflow as tf
xTrain=tf.convert_to_tensor(xTrain.values,dtype=tf.float64)
xTest=tf.convert_to_tensor(xTest.values,dtype=tf.float64)
dataset = tf.data.Dataset.from_tensor_slices((xTrain,yTrain.values))

def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
  return model


train_dataset = dataset.shuffle(len(xTrain)).batch(1)
model = get_compiled_model()
model.fit(train_dataset, epochs=15)
import numpy as np
yTest=model.predict(xTest)
yTest=np.array(yTest)
yTest=np.squeeze(yTest)
output = pd.DataFrame(
    {"PassengerId": test["PassengerId"], "Survived": yTest.astype("int64")}
)
output.to_csv("my_submission.csv", index=False)
