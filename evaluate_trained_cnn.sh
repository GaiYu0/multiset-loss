if [ ! -d "temporary" ]; then
  mkdir temporary
fi

for ((i = 1; i != 10; i++))
do
  for loss in ce_loss regression_loss rl_loss semi_cross_entropy
  do
    python inspect_trained_cnn_on_mnist.py --path="trained-cnn/${loss}${i}" > "temporary/${loss}${i}" &
  done
  wait
done
