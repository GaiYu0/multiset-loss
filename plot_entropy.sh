gpu=0
for scale in 1e0 1e-1 1e-2 1e-3
do
  ipython evaluate.py -- --criterion=regression_loss --entropy-scale=${scale} --n=${1} --gpu=${gpu} --plot-entropy --plot-l1 --tensorboard-log=${scale} &
  let "gpu++"
done
wait

gpu=0
for scale in 1e-4 1e-5 1e-10 1e-20
do
  ipython evaluate.py -- --criterion=regression_loss --entropy-scale=${scale} --n=${1} --gpu=${gpu} --plot-entropy --plot-l1 --tensorboard-log=${scale} &
  let "gpu++"
done
wait
