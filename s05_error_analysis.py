from matplotlib import pyplot as plt
from sklearn.calibration import cross_val_predict
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from s01_config import save_fig
from s02_load_mnist import create_mnist_test_train

X_train, X_test, y_train, y_test = create_mnist_test_train()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
sgd_clf = SGDClassifier(random_state=42)


y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4)) # Create the figure layout (2 subplots next to eachother)
## Make left plot
plt.rc('font', size=9) 
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=axs[0])
axs[0].set_title("Confusion matrix")
## Make right plot
plt.rc('font', size=10)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=axs[1],
                                        normalize="true", values_format=".0%")
axs[1].set_title("CM normalized by row")
## Save figure
save_fig("confusion_matrix_plot_1")
plt.show()


sample_weight = (y_train_pred != y_train) # Create error only weighting mask
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4)) # Creates figure with two subplots
plt.rc('font', size=10) # Set font size
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=axs[0],
                                        sample_weight=sample_weight,
                                        normalize="true", values_format=".0%") # Row normalised error confusion matrix
axs[0].set_title("Errors normalized by row") # Labels left plot
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=axs[1],
                                        sample_weight=sample_weight,
                                        normalize="pred", values_format=".0%") # Column normalised error confusion matrix
axs[1].set_title("Errors normalized by column") # Labels right plot
save_fig("confusion_matrix_plot_2") # Saves image
plt.show()
plt.rc('font', size=14) # Reset font size



size = 5
pad = 0.2
plt.figure(figsize=(size, size))

cl_a, cl_b = '3', '5' # Indentifying classes
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)] # Contains images correctly classified as 3
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)] # Contains images incorrectly classified as 5 
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)] # Contains images incorrectly classified as 3
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)] # Contains images correctly classified as 5

# Arrange the four groups in a 2x2 grid (Rows true, columns predicted)
for images, (label_col, label_row) in [(X_ba, (0, 0)), 
                                       (X_bb, (1, 0)),
                                       (X_aa, (0, 1)), 
                                       (X_ab, (1, 1))]:
    # Plot individual images inside each cell
    for idx, image_data in enumerate(images[:size*size]):
        # Place images in a small grid inside each cell
        x = idx % size + label_col * (size + pad) 
        y = idx // size + label_row * (size + pad)
        # Draw digit
        plt.imshow(image_data.reshape(28, 28), cmap="binary",
                   extent=(x, x + 1, y, y + 1))

# Label axis with digit names 
plt.xticks([size / 2, size + pad + size / 2], [str(cl_a), str(cl_b)])
plt.yticks([size / 2, size + pad + size / 2], [str(cl_b), str(cl_a)])

# Draw dividing lines
plt.plot([size + pad / 2, size + pad / 2], [0, 2 * size + pad], "k:")
plt.plot([0, 2 * size + pad], [size + pad / 2, size + pad / 2], "k:")

# Final formatting
plt.axis([0, 2 * size + pad, 0, 2 * size + pad])
plt.xlabel("Predicted label")
plt.ylabel("True label")

save_fig("error_analysis_digits_plot")
plt.show()