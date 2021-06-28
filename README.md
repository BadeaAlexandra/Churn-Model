# Churn-Model
ANN: Based upon data of employees of a bank we calculate whether a employee stands a chance to stay in the company or not.

----------------------------------


Epoch 1/100
536/536 [==============================] - 0s 876us/step - loss: 0.5261 - accuracy: 0.7835 - val_loss: 0.4827 - val_accuracy: 0.7933



Epoch 100/100
536/536 [==============================] - 1s 1ms/step - loss: 0.3325 - accuracy: 0.8614 - val_loss: 0.3339 - val_accuracy: 0.8637


----------------------------------


print(cm)
[[1548 47]
 [ 227  178]]
 
 
 print(accuracy_score(y_test, y_pred))
0.863
