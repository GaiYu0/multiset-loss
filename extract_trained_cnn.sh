cnn_path="trained-cnn"
if [ ! -d ${cnn_path} ]; then
  mkdir ${cnn_path}
fi
options="--disable-tensorboard --disable-visdom --n=${1}"
ipython evaluate.py -- --cnn-path="${cnn_path}/ce_loss${1}" --criterion=ce_loss --gpu=0 ${options} &
ipython evaluate.py -- --cnn-path="${cnn_path}/regression_loss${1}" --criterion=regression_loss --entropy-scale=0 --gpu=1 ${options} &
ipython evaluate.py -- --cnn-path="${cnn_path}/rl_loss${1}" --criterion=rl_loss --gpu=2 ${options} &
ipython evaluate.py -- --cnn-path="${cnn_path}/semi_cross_entropy${1}" --criterion=semi_cross_entropy --gpu=3 ${options} &
wait
