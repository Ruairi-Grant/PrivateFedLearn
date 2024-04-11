- fix evaluate error in central_mnist.py 
  File "/home/rock/PrivateFedLearn/dp_test/dp_sgd_mnist/Central_mnist.py", line 135, in evaluate_model
    _, eval_acc = eval_model.evaluate(dataset, verbose=1)
 ValueError: Layer "sequential" expects 1 input(s), but it received 2 input tensors. Inputs received: [<tf.Tensor 'IteratorGetNext:0' shape=(32, 28, 28, 1) dtype=float32>, <tf.Tensor 'IteratorGetNext:1' shape=(32, 10) dtype=float32>]

- setup .bat scripts for windows computer 
- output validation and accuracy plots
- output confusion matricies
- 

# Notes on current hyper-paramaters
- central dp is not training well, try reducing the noise, maye sligfhtly smaller learning rate, introducing momentum
