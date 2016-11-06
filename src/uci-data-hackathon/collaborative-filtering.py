from math import sqrt
import numpy as np
from collections import defaultdict

def pearson_correlation(person1,person2):

    #print("ran")

    # To get both rated items
    both_rated = {}
    for item in dataset[person1]:
        if item in dataset[person2]:
            both_rated[item] = 1

    number_of_ratings = len(both_rated)     
    
    # Checking for number of ratings in common
    if number_of_ratings == 0:
        print "IM EXITING"
        return 0

    # Add up all the preferences of each user
    person1_preferences_sum = sum([dataset[person1][item] for item in both_rated])
    person2_preferences_sum = sum([dataset[person2][item] for item in both_rated])

    # Sum up the squares of preferences of each user
    person1_square_preferences_sum = sum([pow(dataset[person1][item],2) for item in both_rated])
    person2_square_preferences_sum = sum([pow(dataset[person2][item],2) for item in both_rated])

    # Sum up the product value of both preferences for each item
    product_sum_of_both_users = sum([dataset[person1][item] * dataset[person2][item] for item in both_rated])
    return product_sum_of_both_users
    # Calculate the pearson score
    numerator_value = product_sum_of_both_users - (person1_preferences_sum*person2_preferences_sum/number_of_ratings)
    #denominator_value = sqrt((person1_square_preferences_sum - pow(person1_preferences_sum,2)/number_of_ratings) * (person2_square_preferences_sum -pow(person2_preferences_sum,2)/number_of_ratings))
    #print numerator_value
    denominator_value = 1
    if denominator_value == 0:
        #print "DEMON"
        return 0
    else:
        r = numerator_value/denominator_value
    
    return r 



def similarity_score(person1,person2):

    # Returns ratio Euclidean distance score of person1 and person2 
 
    both_viewed = {} # To get both rated items by person1 and person2
 
    for item in dataset[person1]:
       if item in dataset[person2]:
          both_viewed[item] = 1
 
   # Conditions to check they both have an common rating items
    if len(both_viewed) == 0:
        return 0
 
   # Finding Euclidean distance
    sum_of_eclidean_distance = [] 
 
    for item in dataset[person1]:
       if item in dataset[person2]:
            sum_of_eclidean_distance.append(2)
           #sum_of_eclidean_distance.append(pow(dataset[person1][item] - dataset[person2][item],2))
    
    sum_of_eclidean_distance = sum(sum_of_eclidean_distance)
 
    print 1/(1+sqrt(sum_of_eclidean_distance))
    return 1/(1+sqrt(sum_of_eclidean_distance))

def user_reommendations(person):

    # Gets recommendations for a person by using a weighted average of every other user's rankings
    totals = {}
    simSums = {}
    rankings_list =[]
    for other in dataset:
        # don't compare me to myself
        if other == person:
            continue
        sim = pearson_correlation(person,other)
        #print ">>>>>>>",sim

        # ignore scores of zero or lower
        if sim <=0: 
            continue
        for item in dataset[other]:

            # only score movies i haven't seen yet
            if item not in dataset[person] or dataset[person][item] == 0:

            # Similrity * score
                totals.setdefault(item,0)
                totals[item] += dataset[other][item]* sim
                # sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+= sim

        # Create the normalized list

    rankings = [(total/simSums[item],item) for item,total in totals.items()]
    rankings.sort()
    rankings.reverse()
    # returns the recommended items
    recommendataions_list = [recommend_item for score,recommend_item in rankings]
    return recommendataions_list

def most_similar_users(person,number_of_users):
    # returns the number_of_users (similar persons) for a given specific person.
    scores = [(pearson_correlation(person,other_person),other_person)    for other_person in dataset if other_person != person ]
 
    # Sort the similar persons so that highest scores person will appear at the first
    scores.sort()
    scores.reverse()
    #print scores
    return scores[0:number_of_users]
    #return scores

def crappy_learner(person1, person2):
    results = {}
    print "running..."
    print dataset[person2]
    for i in dataset[person1]:
        print i

if __name__ == '__main__':

    file = open("gender.csv")

    dataset = defaultdict(lambda : defaultdict(int))

    inverted_index = defaultdict(list)

    print ("reading file")
    for i in file.readlines():

        id, package = i.split(",")
        package = package.strip('\n').replace('"', '')
        dataset[id][package] = 1
        inverted_index[package].append(id)



    #print pearson_correlation('34106','')
    from pprint import pprint
    print most_similar_users('2450732',4)
