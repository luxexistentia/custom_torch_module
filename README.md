This is a custom module for PyTorch training
-
(especially only for computer vision projects)

Each file has relavant functions and classes.

1. deploy_utils.py : for when you want to export as .onnx file easily and also want to load and run .onnx file easily without writing too complicated codes
2. engine.py : for when you develop/test your model. once you make Model_Trainer instance and register model, weights .etc, Then you can train the model only by calling .train() method. And load weights explicitly again with .load_weights() You do not need to send a lot of variables everytime you just want to train and load. Just make an instance once then done.
3. eval_utils.py : some functions or classes to help you visualize ur model's prediction will be added. Not added yet(2024 Aug 24th)
4. setup_utils.py : group of functions that help you to setup your data, dataloader.etc


I've been making it for myself.
But also you can use it and contribute as you want under MIT License.

Let me know if there is any bug, error or idea!

Idea List
-
1. To add wandb compatibility
2. To add Visualize function for evaluation for test images
