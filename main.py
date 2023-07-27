import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.inspection import DecisionBoundaryDisplay

# ------------------------------------------------------------
# Set initial config for the whole program.
# ------------------------------------------------------------

random_state = np.random.RandomState(None)

# ------------------------------------------------------------
# Create sample data.
# ------------------------------------------------------------

random_dots = random_state.randn(10, 2)
cluster = 0.2 * random_state.randn(100, 2)
data = np.concatenate([random_dots, cluster])

# ------------------------------------------------------------
# Run anomaly detection on data.
# ------------------------------------------------------------

forest = IsolationForest(max_samples=75, random_state=0)
forest.fit(data)

# ------------------------------------------------------------
# Display results.
# ------------------------------------------------------------

disp = DecisionBoundaryDisplay.from_estimator(
    forest,
    data,
    response_method="predict"
)

disp.ax_.scatter(data[:, 0], data[:, 1], s=5)

plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()
