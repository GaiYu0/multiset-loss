if [[ -z $1 ]]
then
  for ((i = 1; i != 11; i++))
  do
    python generate_data.py --n=${i} --path=data/training-${i}.data --size=50000 --source=training &
    python generate_data.py --n=${i} --path=data/validation-${i}.data --size=10000 --source=validation &
    python generate_data.py --n=${i} --path=data/test-${i}.data --size=10000 --source=test &
  done
else
  python generate_data.py --n=${1} --path=data/training-${1}.data --size=50000 --source=training &
  python generate_data.py --n=${1} --path=data/validation-${1}.data --size=10000 --source=validation &
  python generate_data.py --n=${1} --path=data/test-${1}.data --size=10000 --source=test &
fi
wait
