import pandas 
import numpy
import sklearn


# lecture du fichier csv
file = pandas.read_csv('train.csv')

# afficher le contenu du fichier csv
#print(file)
nb_survived = file.query(expr="Survived==1")
nb_mort = file.query(expr="Survived==0")
print(len(nb_survived),"survivants",len(nb_mort),"morts")

# vérifier qu'il n'y a pas de valeurs nulles pour l'age
survived_null = len(file.isna().query(expr="Survived==True"))
print("valeurs survivant null",survived_null)

# calcul de l'afe moyen des passagers
mean_age = round(file['Age'].mean())
print("age moyen :", mean_age)

# pour les valeurs nulles, on remplace par la valeur moyenne
#modifier de façon importante les données
# inplace permet de modifier de façon permanente la collonne
print(file['Age'].head(10))
file.Age.fillna(mean_age, inplace=True)
print(file['Age'].head(10))

# afficher le nb de personne dans chacune des classes
nb_classe_1 = file.query(expr="Pclass == 1")
nb_classe_1_mort = file.query(expr="Pclass == 1 and Survived == 0")
nb_classe_2 = file.query(expr="Pclass == 2")
nb_classe_2_mort = file.query(expr="Pclass == 2 and Survived == 0")
nb_classe_3 = file.query(expr="Pclass == 3")
nb_classe_3_mort = file.query(expr="Pclass == 3 and Survived == 0")
print(len(nb_classe_1), "passagers en classe 1 et",len(nb_classe_1_mort), "morts")
print(len(nb_classe_2), "passagers en classe 2 et",len(nb_classe_2_mort), "morts")
print(len(nb_classe_3), "passagers en classe 3 et",len(nb_classe_3_mort), "morts")

# afficher le % de personnes qui ont suvécu pour chacune des classes
print("pourcentage de survivant en classe 1 :",100-round(len(nb_classe_1_mort)/len(nb_classe_1)*100),"%")
print("pourcentage de survivant en classe 2 :",100-round(len(nb_classe_2_mort)/len(nb_classe_2)*100),"%")
print("pourcentage de survivant en classe 3 :",100-round(len(nb_classe_3_mort)/len(nb_classe_3)*100),"%")

# remplacer "male" et "male" par 1 pour "male" et 0 pour "female"
file.replace('male', 1, inplace=True)
file.replace('female', 0, inplace=True)

# supprimer les colonnes qui ne seront pas utlisées
# contenant des valeurs textuelles ou 
# ne semblant n'avoir pas d'intéret pour l'analyse
# axis 1=colun, 0=line
file.drop(['Name','Ticket','PassengerId','SibSp','Parch','Fare','Cabin','Embarked'], axis)
print(file.head())

# appliquer les transformations appliqué a nos donnes tain au données test
test = pandas.read_csv()

# lancement des algo
kn_model = pandas.KNeighborsClassifier(n_neighbors=1)
kn_model.fit(file, test)
preds_kn = kn_model.predict(1)