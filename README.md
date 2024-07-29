Dataset and Network combinations:

--dataset cifar --model vgg16
--dataset fashion_mnist --model cnn

To run our rate allocation, please install matlab.engine

To run our client selection methods in PB settings:
python main.py --dataset cifar --local_ep 3 --model vgg16 --epochs 20 --bias 0.25 --lr 0.03  --cov_analysis --user_select cov

To run our client selection methods in PS settings:
python main.py --dataset cifar --local_ep 3 --model vgg16 --epochs 20 --bias 0.0 --lr 0.01 --cov_analysis --user_select cov --shards_per_client 1


To compare with SOTA methods (follow above to change heterogeneity):
python main.py --dataset cifar --local_ep 3 --model vgg16 --epochs 20 --bias 0.25 --lr 0.03  --gpr

python main.py --dataset cifar --local_ep 3  --model vgg16 --epochs 20 --bias 0.25 --lr 0.03  --afl

python main.py --dataset cifar --local_ep 3  --model vgg16 --epochs 20 --bias 0.25 --lr 0.03  --powd


To run rate allocation:
python main.py --dataset cifar --local_ep 3 --model vgg16 --epochs 20 --bias 0.25 --lr 0.03  --cov_analysis --quant_method binning  --simulate_quant

python main.py --dataset cifar --local_ep 3 --model vgg16 --epochs 20 --bias 0.25 --lr 0.03  --cov_analysis --quant_method no_bin --simulate_quant

python main.py --dataset cifar --local_ep 3 --model vgg16 --epochs 20 --bias 0.25 --lr 0.03  --cov_analysis --quant_method centralized --simulate_quant


To run client selection + rate allocation:
python main_both.py --dataset cifar --local_ep 3 --model vgg16 --epochs 20 --bias 0.25 --lr 0.03 --cov_analysis --quant_method binning --user_select cov --both
