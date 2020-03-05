#python main.py -a alexnet /home/xiaohang/ImageNet/ 
#nohup python main.py -a mobilenet /home/xiaohang/ImageNet/  > log.txt &

# run the torchvision version.
#python  main.py --pretrained --evaluate -a mobilenet_v2 ~/imagenet/
#python main.py --pretrained --evaluate -a shufflenet_v2_x1_0 ~/imagenet/


# run mobilenetv1 with pretrained model.
#python main.py --resume ./mobilenet_sgd_68.848.pth.tar -a mobilenet  --evaluate ~/imagenet/
python main_cpu.py --resume ./mobilenet_sgd_68.848.pth.tar -a mobilenet  --evaluate ~/imagenet/
