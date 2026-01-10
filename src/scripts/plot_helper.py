import matplotlib.pyplot as plt



def plot_loss(history):
    plt.figure(figsize=(20,5))
    plt.plot(history.history['loss'], 'g', label='Training Loss')
   # plt.plot(history.history['val_loss'], 'b', label='Validation Loss')
    plt.xlim(0,200)
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


def evaluate_model(model, X_test, y_test):
   
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import pandas as pd
    
    y_pred = model.predict(X_test)
       
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Create and display table
    metrics_df = pd.DataFrame({
        'Metric': ['R² Score', 'RMSE', 'MAE'],
        'Value': [f'{r2:.6f}', f'{rmse:.6f}', f'{mae:.6f}']
    })
    
    print("="*40)
    print("Model Evaluation Metrics")
    print("="*40)
    print(metrics_df.to_string())
    print("="*40)
    

