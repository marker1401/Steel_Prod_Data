import matplotlib.pyplot as plt



def plot_loss(history):
   def plot_loss(history):
    plt.figure(figsize=(20,5))
    plt.plot(history.history['loss'], 'g', label='Training Loss')
   # plt.plot(history.history['val_loss'], 'b', label='Validation Loss')
    plt.xlim(0,100)
    plt.ylim(bottom=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)



def plot_data(X_data, y_data, y, title=None):
     
    plt.figure(figsize=(15,5))
    plt.scatter(X_data[:,20], y_data, label='ground truth', alpha=0.4)
    plt.scatter(X_data[:,20], y, label='Model Predictions',alpha=0.4)
    plt.xlabel('Input21')
    plt.ylabel('Output')
    plt.title(title)
    plt.grid(True)
    plt.legend()


