train_model:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py args_train.Carla.txt
test_compute_matrix:
python evaluate.py args_test_Carla_local.txt
visualize_test_result:
python infer.py
