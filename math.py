import csv
import math
import random
import pandas as pd

print("=" * 70)
print("TITANIC SURVIVAL PREDICTION")
print("Mini Project 1 - Data Cleaning Project")
print("=" * 70)

print("\nABOUT")
print("""
The Titanic dataset contains information about passengers who travelled
on the RMS Titanic. This project focuses on cleaning the passenger data,
understanding important features and predicting whether a passenger
survived the accident.

The target column is Survived.
0 = Not Survived
1 = Survived
""")


csv_file_path = './titanic.csv'

try:
    df = pd.read_csv(csv_file_path)
    data = df.to_dict('records')
except FileNotFoundError:
    print(f"\nError: The file '{csv_file_path}' was not found. Please check the path and try again.")
    exit()
except Exception as e:
    print(f"\nAn error occurred while loading the CSV: {e}")
    exit()

print("\nDATASET")
print("-" * 70)
print("Rows:", len(data))
print("Columns:", len(data[0]))

print("\nCOLUMNS")
for column in data[0].keys():
    print("-", column)

print("\nCOMPACT VIEW")
print("-" * 120)

print(
    f"{'ID':<6}"
    f"{'Survived':<10}"
    f"{'Pclass':<8}"
    f"{'Name':<30}"
    f"{'Sex':<10}"
    f"{'Age':<8}"
    f"{'Ticket':<20}"
)

print("-" * 120)

for row in data[:10]:
    print(
        f"{row['PassengerId']:<6}"
        f"{row['Survived']:<10}"
        f"{row['Pclass']:<8}"
        f"{row['Name'][:28]:<30}"
        f"{row['Sex']:<10}"
        f"{row['Age']:<8}"
        f"{row['Ticket'][:18]:<20}"
    )

print("\nCOLUMN INFORMATION")
print("-" * 70)

for column in data[0].keys():

    values = [str(row[column]) for row in data] 

    missing = 0

    for value in values:
        if value.strip() == "":
            missing += 1

    unique_values = len(set(values))

    print(
        f"{column:<12}"
        f"Missing = {missing:<5}"
        f"Unique = {unique_values}"
    )

print("\nMISSING VALUES BEFORE CLEANING")
print("-" * 70)

for column in data[0].keys():

    missing = 0

    for row in data:
        if str(row[column]).strip() == "": 
            missing += 1

    if missing > 0:
        print(column, ":", missing)

indian_names = [
    "Aarav Sharma",
    "Ananya Verma",
    "Arjun Mehta",
    "Diya Kapoor",
    "Rohan Gupta",
    "Priya Singh",
    "Aditya Patel",
    "Sneha Joshi",
    "Rahul Malhotra",
    "Isha Agarwal",
    "Vikram Rao",
    "Neha Nair",
    "Karan Shah",
    "Pooja Mishra",
    "Manish Kumar",
    "Riya Saxena",
    "Siddharth Jain",
    "Kavya Reddy",
    "Varun Bansal",
    "Anjali Desai"
]

for i, row in enumerate(data):
    row["Indian_Name"] = indian_names[i % len(indian_names)]

    if str(row["Survived"]) == "1": 
        row["Survival_Status"] = "Survived"
    else:
        row["Survival_Status"] = "Not Survived"

age_values = []

for row in data:
    if str(row["Age"]).strip() != "": 
        try:
            age_values.append(float(row["Age"]))
        except ValueError:
            pass

average_age = sum(age_values) / len(age_values)

for row in data:

    if str(row["Age"]).strip() == "": 
        row["Age"] = str(round(average_age, 2))

    if str(row["Embarked"]).strip() == "": 
        row["Embarked"] = "S"

    if str(row["Fare"]).strip() == "": 
        row["Fare"] = "0"

    if str(row["Cabin"]).strip() == "": 
        row["Cabin"] = "Unknown"

print("\nDATA CLEANING COMPLETED")
print("-" * 70)
print("Missing Age      -> replaced with average age")
print("Missing Embarked -> replaced with S")
print("Missing Fare     -> replaced with 0")
print("Missing Cabin    -> replaced with Unknown")

print("\nCLEANED COMPACT VIEW")
print("-" * 120)

print(
    f"{'ID':<6}"
    f"{'Indian Name':<20}"
    f"{'Ticket':<20}"
    f"{'Sex':<10}"
    f"{'Age':<8}"
    f"{'Survival':<15}"
)

print("-" * 120)

for row in data[:10]:

    print(
        f"{row['PassengerId']:<6}"
        f"{row['Indian_Name']:<20}"
        f"{str(row['Ticket'])[:18]:<20}"
        f"{row['Sex']:<10}"
        f"{str(row['Age']):<8}"
        f"{row['Survival_Status']:<15}"
    )

survived = 0
not_survived = 0

for row in data:

    if str(row["Survived"]) == "1": 
    else:
        not_survived += 1

survival_rate = survived / len(data) * 100

print("\nSURVIVAL ANALYSIS")
print("-" * 70)
print("Total Passengers :", len(data))
print("Survived         :", survived)
print("Not Survived     :", not_survived)
print("Survival Rate    :", round(survival_rate, 2), "%")

print("\nGRAPH 1 - SURVIVAL")
print("-" * 50)

print("Survived     |" + "#" * (survived // 10))
print("Not Survived |" + "#" * (not_survived // 10))

male_total = 0
male_survived = 0

female_total = 0
female_survived = 0

for row in data:

    if str(row["Sex"]) == "male": 
        male_total += 1

        if str(row["Survived"]) == "1": 
            male_survived += 1

    elif str(row["Sex"]) == "female": 
        female_total += 1

        if str(row["Survived"]) == "1": 
            female_survived += 1

print("\nGRAPH 2 - SURVIVAL BY GENDER")
print("-" * 50)

print("Male")
print("  Survived     |" + "#" * (male_survived // 10))
print("  Not Survived |" + "#" * ((male_total - male_survived) // 10))

print("Female")
print("  Survived     |" + "#" * (female_survived // 10))
print("  Not Survived |" + "#" * ((female_total - female_survived) // 10))

print("\nGENDER SURVIVAL RATE")
print("Male   :", round(male_survived / male_total * 100, 2), "%")
print("Female :", round(female_survived / female_total * 100, 2), "%")

print("\nGRAPH 3 - PASSENGER CLASS")
print("-" * 50)

for passenger_class in ["1", "2", "3"]:

    total = 0
    class_survived = 0

    for row in data:

        if str(row["Pclass"]) == passenger_class:

            total += 1

            if str(row["Survived"]) == "1": 
                class_survived += 1

    print(
        "Class",
        passenger_class,
        "|",
        "#" * (total // 10),
        "Total:",
        total,
        "Survived:",
        class_survived
    )

print("\nAGE ANALYSIS")
print("-" * 70)

child_survived = 0
child_total = 0

adult_survived = 0
adult_total = 0

senior_survived = 0
senior_total = 0

for row in data:

    age = float(row["Age"])

    if age < 18:

        child_total += 1

        if str(row["Survived"]) == "1": 
            child_survived += 1

    elif age < 60:

        adult_total += 1

        if str(row["Survived"]) == "1": 
            adult_survived += 1

    else:

        senior_total += 1

        if str(row["Survived"]) == "1": 
            senior_survived += 1

print("Children :", child_total, "Survived:", child_survived)
print("Adults   :", adult_total, "Survived:", adult_survived)
print("Seniors  :", senior_total, "Survived:", senior_survived)

print("\nMACHINE LEARNING")
print("-" * 70)
print("Model: Logistic Regression")
print("Features: Sex, Pclass, Age")

random.seed(42)

weights = [
    random.uniform(-0.1, 0.1),
    random.uniform(-0.1, 0.1),
    random.uniform(-0.1, 0.1),
    random.uniform(-0.1, 0.1)
]

def sigmoid(value):

    if value < -500:
        return 0

    if value > 500:
        return 1

    return 1 / (1 + math.exp(-value))

def get_features(row):

    sex = 1 if str(row["Sex"]) == "female" else 0 
    pclass = float(row["Pclass"])
    age = float(row["Age"]) / 100

    return [1, sex, pclass, age]

learning_rate = 0.05
epochs = 2500

for epoch in range(epochs):

    gradients = [0, 0, 0, 0]

    for row in data:

        x = get_features(row)
        actual = int(str(row["Survived"])) 

        value = 0

        for i in range(4):
            value += weights[i] * x[i]

        prediction = sigmoid(value)
        error = prediction - actual

        for i in range(4):
            gradients[i] += error * x[i]

    for i in range(4):
        weights[i] -= learning_rate * gradients[i] / len(data)

correct = 0

for row in data:

    x = get_features(row)

    value = 0

    for i in range(4):
        value += weights[i] * x[i]

    probability = sigmoid(value)

    prediction = 1 if probability >= 0.5 else 0

    if prediction == int(str(row["Survived"])):
        correct += 1

accuracy = correct / len(data) * 100

print("Model Accuracy:", round(accuracy, 2), "%")

print("\nSAMPLE PREDICTIONS")
print("-" * 100)

print(
    f"{'ID':<6}"
    f"{'Indian Name':<20}"
    f"{'Ticket':<18}"
    f"{'Actual':<16}"
    f"{'Prediction':<16}"
)

print("-" * 100)

for row in data[:10]:

    x = get_features(row)

    value = 0

    for i in range(4):
        value += weights[i] * x[i]

    probability = sigmoid(value)

    prediction = 1 if probability >= 0.5 else 0

    if prediction == 1:
        predicted_status = "Survived"
    else:
        predicted_status = "Not Survived"

    print(
        f"{row['PassengerId']:<6}"
        f"{row['Indian_Name']:<20}"
        f"{str(row['Ticket'])[:16]:<18}"
        f"{row['Survival_Status']:<16}"
        f"{predicted_status:<16}"
    )

print("\nPROJECT SUMMARY")
print("-" * 70)

print("""
The Titanic passenger data was successfully loaded and cleaned.
Missing values were handled using simple data-cleaning techniques.

The project examined passenger survival based on gender, age and
passenger class. Additional display fields such as Indian Name,
Survival Status and Ticket were included to make the dataset easier
to understand.

A Logistic Regression model was implemented using Python to predict
whether a passenger survived or not.

The project demonstrates data cleaning, exploratory analysis,
feature preparation and machine learning prediction.
""")

print("=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)
