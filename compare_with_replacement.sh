for ((n_digits = 1; n_digits != 11; n_digits++))
do
  python evaluate.py --criterion=alternative_semi_cross_entropy --datapath=data-with-replacement --gpu=0 --interval=100 --n=${n_digits} --tensorboard-log=semi-cross-entropy --tensorboard-postfix=re${n_digits} &
  python evaluate.py --criterion=jsd_loss --datapath=data-with-replacement --gpu=1 --interval=100 --n=${n_digits} --tensorboard-log=jsd-loss --tensorboard-postfix=re${n_digits} &
  python evaluate.py --criterion=rl_loss --datapath=data-with-replacement --gpu=2 --interval=100 --n=${n_digits} --tensorboard-log=rl-loss --tensorboard-postfix=re${n_digits} &
  python evaluate.py --criterion=ce_loss --datapath=data-with-replacement --gpu=3 --interval=100 --n=${n_digits} --tensorboard-log=ce-loss --tensorboard-postfix=re${n_digits} &
  wait
done
