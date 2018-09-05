import pandas as pd 
import matplotlib.pyplot as plt 

file = 'val_score_1.csv'
val_score = pd.read_csv(file)
kappa = val_score.loc[:,['kappa']]
plt.plot(kappa)
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.title('Kappa score after blending in 50 iterations')
plt.savefig('kappa.png')
plt.show()


precision = val_score.loc[:,['precision']]
plt.plot(precision)
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.title('Precision score after blending in 50 iterations')
plt.savefig('precision.png')
plt.show()

recall = val_score.loc[:,['recall']]
plt.plot(recall)
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.title('Recall (sensitivity) score after blending in 50 iterations')
plt.savefig('recall.png')
plt.show()