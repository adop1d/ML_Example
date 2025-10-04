#import libraries
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from IPython.display import display, clear_output

#generate synthetic data
np.random.seed(0)
X = np.linspace(-5,5,100)

#Create function for interactive visualization
def plot_logistic_regression(slope=2.0, intercept=-1.0,threshold=0.5):
    new_linear_combination = slope * X + intercept
    new_probability = 1 / (1 + np.exp(-new_linear_combination))
    predicted_labels = (new_probability >= threshold).astype(int)

    #create subplots
    fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(18, 5))

    #plot the linear combination
    ax1.plot(X, new_linear_combination, label='Linear Combination')
    ax1.set_xlabel('Feature Value: X')
    ax1.set_ylabel('Linear Combination: Y')
    ax1.legend()

    #Plot the logistic function output as a subplot
    ax2.plot(X, new_probability, label='Logistic Function')
    ax2.set_xlabel('Feature Value: X')
    ax2.set_ylabel('Probability')
    ax2.legend()

    #Plot the predicted labels
    ax3.scatter(X, predicted_labels, label='Predicted Labels')
    ax3.set_xlabel('Feature Value: X')
    ax3.set_ylabel('Predicted Label')
    ax3.legend()

    plt.suptitle('Logistic Regression Visualization')

#create sliders for slope, intercept, and threshold
slope_slider = FloatSlider(value=2.0, min=-5.0, max=5.0, step=0.1, description='Slope:')
intercept_slider = FloatSlider(value=-1.0, min=-5.0, max=5.0, step=0.1, description='Intercept:')
threshold_slider = FloatSlider(value=0.5, min=0.0, max=1.0, step=0.01, description='Threshold:')

#use interact to create interactive plot
interact(plot_logistic_regression, slope=slope_slider, intercept=intercept_slider, threshold=threshold_slider)

plt.show()
