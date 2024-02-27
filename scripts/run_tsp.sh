model_name="$1"
for dataset in klokan
do
  echo "Running evaluation for dataset: $dataset with model: $model_name"
  python -m czeval.run --dataset-config-path="config/dataset/$dataset.yaml" --model-config-path="config/model/$model_name.yaml" --task="config/task/qa.yaml"
done
