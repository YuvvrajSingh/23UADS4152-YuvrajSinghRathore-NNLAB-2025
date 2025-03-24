### **Output:**  
```
Predictions for XOR: [0 1 1 0]
```

---

### **My Comments:**  

- The perceptron is trying to **solve the XOR problem**, which a **single-layer perceptron can't do**. So, the code cleverly **uses two hidden neurons** to handle it. 
- The **step function** is used for activation, meaning the model **only outputs 0 or 1** (no probabilities like in sigmoid).  
- The model **learns through updates** to its weights and bias using a simple **perceptron learning rule**.  
- It **trains three perceptrons**: two for the hidden layer and one for the output.  
