import matplotlib.pyplot as plt
import numpy as np
import joblib as jl
import pandas as pd

def plot_predictions(actual_ratings,predicted_ratings,lower_bound,upper_bound):
    
    # Create a range for the x-axis
    x = np.arange(len(actual_ratings))

    # Plot actual ratings
    plt.plot(x, actual_ratings, 
             label='Actual Ratings',
             marker='o', 
             linestyle='-',
               color='blue',
               alpha = 0.6)

    # Plot predicted ratings
    plt.plot(x, predicted_ratings, 
             label='Predicted Ratings',
               marker='x',
                 linestyle='--',
                   color='orange',
                   alpha = 0.8)

    # Add confidence intervals
    plt.fill_between(x, lower_bound, upper_bound,
                      color='orange', 
                      alpha=0.2, 
                      label='95% Confidence Interval')

    # Add labels and legend
    plt.xlabel('Data Points')
    plt.ylabel('Ratings')
    plt.title('Actual vs Predicted Ratings with Confidence Intervals')
    plt.legend()

    # Display the plot
    plt.show()
    




model_path = "models/movie_ratings_prediction.joblib"
model = jl.load(model_path)

test_data_path = 'data/processed/filtered_training_data_actorCount_V2.csv'
test_data = pd.read_csv(test_data_path)

feature_columns = [col for col in test_data.columns if col not in ['id', 'vote_average']]
X_test = test_data[feature_columns]
y_test = test_data['vote_average']

predictions = model.predict(X_test)

X_test_array = X_test.values  

# Get predictions from all trees
all_tree_predictions = np.array([tree.predict(X_test_array) for tree in model.estimators_])

# Calculate the mean prediction
mean_prediction = np.mean(all_tree_predictions, axis=0)

# Calculate the standard deviation of predictions
std_dev = np.std(all_tree_predictions, axis=0)

# Define a 95% confidence interval
lower_bound = mean_prediction - 1.96 * std_dev
upper_bound = mean_prediction + 1.96 * std_dev


print(predictions)

num_points = 100

plot_predictions(y_test[:num_points],predictions[:num_points],lower_bound[:num_points],upper_bound[:num_points])
