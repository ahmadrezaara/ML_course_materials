import matplotlib.pyplot as plt
import numpy as np

# Function to generate Pascal's Triangle
def generate_pascals_triangle(k):
    triangle = [[1] * (i + 1) for i in range(k)]
    for i in range(2, k):
        for j in range(1, i):
            triangle[i][j] = triangle[i - 1][j - 1] + triangle[i - 1][j]
    return triangle

# Taking order k as input from user
k = int(input("Enter the order of Pascal's Triangle: "))

# Generating Pascal's Triangle of order k
pascals_triangle = generate_pascals_triangle(k)

# Plotting the triangle using matplotlib
for i in range(k):
    numbers = pascals_triangle[i]
    x_coords = np.arange(len(numbers)) + (k - len(numbers)) / 2.0 # Centering each row of numbers
    y_coords = [k - i] * len(numbers) # Rows are at height k, k-1, ..., 0
    
    plt.text(x_coords, y_coords, numbers,
             fontsize=12,
             ha='center',
             va='center')

plt.axis('off') # Turn off axis labels and ticks
plt.show() # Display the plot window 
