import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pokemon = pd.read_csv(r'C:\Users\User\Desktop\tugas no3\Ujian_MachineLearning_JCDS04-master\Dataset_3\pokemon.csv')
combats = pd.read_csv(r'C:\Users\User\Desktop\tugas no3\Ujian_MachineLearning_JCDS04-master\Dataset_3\combats.csv')
tests = pd.read_csv(r'C:\Users\User\Desktop\tugas no3\Ujian_MachineLearning_JCDS04-master\Dataset_3\tests.csv')
pokemon = pokemon.drop(['Type 1','Type 2','Generation','Legendary'],axis = 1)
pokemon['total'] = pokemon['HP'] + pokemon['Attack'] + pokemon['Defense'] + pokemon['Sp. Atk'] + pokemon['Sp. Def'] + pokemon['Speed']
# Create dict
dictName = dict(zip(pokemon['#'], pokemon['Name']))
dictHp = dict(zip(pokemon['#'], pokemon['HP']))
dictAttack = dict(zip(pokemon['#'], pokemon['Attack']))
dictDeff = dict(zip(pokemon['#'], pokemon['Defense']))
dictSpAttack = dict(zip(pokemon['#'], pokemon['Sp. Atk']))
dictSpDeff = dict(zip(pokemon['#'], pokemon['Sp. Def']))
dictSpeed = dict(zip(pokemon['#'], pokemon['Speed']))
dictSum = dict(zip(pokemon['#'], pokemon['total']))
# Create dataframe combat
dfCombat = combats.copy()
# first Pokemon
dfCombat['First_pokemon_name'] = dfCombat['First_pokemon'].replace(dictName)
dfCombat['First_pokemon_hp'] = dfCombat['First_pokemon'].replace(dictHp)
dfCombat['First_pokemon_attack'] = dfCombat['First_pokemon'].replace(dictAttack)
dfCombat['First_pokemon_defense'] = dfCombat['First_pokemon'].replace(dictDeff)
dfCombat['First_pokemon_spattack'] = dfCombat['First_pokemon'].replace(dictSpAttack)
dfCombat['First_pokemon_spdefense'] = dfCombat['First_pokemon'].replace(dictSpDeff)
dfCombat['First_pokemon_speed'] = dfCombat['First_pokemon'].replace(dictSpeed)
dfCombat['First_pokemon_total'] = dfCombat['First_pokemon'].replace(dictSum)
# Second Pokemon
dfCombat['Second_pokemon_name'] = dfCombat['Second_pokemon'].replace(dictName)
dfCombat['Second_pokemon_hp'] = dfCombat['Second_pokemon'].replace(dictHp)
dfCombat['Second_pokemon_attack'] = dfCombat['Second_pokemon'].replace(dictAttack)
dfCombat['Second_pokemon_defense'] = dfCombat['Second_pokemon'].replace(dictDeff)
dfCombat['Second_pokemon_spattack'] = dfCombat['Second_pokemon'].replace(dictSpAttack)
dfCombat['Second_pokemon_spdefense'] = dfCombat['Second_pokemon'].replace(dictSpDeff)
dfCombat['Second_pokemon_speed'] = dfCombat['Second_pokemon'].replace(dictSpeed)
dfCombat['Second_pokemon_total'] = dfCombat['Second_pokemon'].replace(dictSum)
dfCombat['First_win'] = dfCombat.apply(lambda col: 1 
    if col['Winner'] == col['First_pokemon'] 
    else 0,
     axis=1
)
# Create feature X and target Y
X = dfCombat.drop(['First_pokemon', 'First_pokemon_name', 'Second_pokemon', 'Second_pokemon_name', 'Winner', 'First_win'], axis=1)
Y = dfCombat['First_win']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, 
    Y, 
    test_size=0.7, 
    random_state=10
)
# Decision Trees
from sklearn import tree
modelDT = tree.DecisionTreeClassifier()
modelDT.fit(X_train, Y_train)
import joblib
joblib.dump(modelDT, 'pokeModelDT')