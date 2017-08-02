if [[ -z $1 ]]
then
  for ((i = 1; i != 11; i++))
  do
    python generate_data.py --n=${i} --path=data-with-replacement/training-${i}.data --replace --size=50000 --source=training &
    python generate_data.py --n=${i} --path=data-with-replacement/validation-${i}.data --replace --size=10000 --source=validation &
    python generate_data.py --n=${i} --path=data-with-replacement/test-${i}.data --replace --size=10000 --source=test &
  done
else
  python generate_data.py --n=${1} --path=data-with-replacement/training-${1}.data --replace --size=50000 --source=training &
  python generate_data.py --n=${1} --path=data-with-replacement/validation-${1}.data --replace --size=10000 --source=validation &
  python generate_data.py --n=${1} --path=data-with-replacement/test-${1}.data --replace --size=10000 --source=test &
fi
wait
