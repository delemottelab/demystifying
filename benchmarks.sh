for it in {1..3}; do
echo "Iteration $iter"
for ns in 1e-2; do
for tm in linear non-linear; do
for ft in inv-dist; do
cmd="python run_benchmarks.py --extractor_type=AE --output_dir=output/mega/results/benchmarking/ --feature_type=$ft --noise_level=$ns --test_model=$tm --accuracy_method mse relevant_fraction --overwrite=False;"
echo $cmd;
$cmd
done
done
done;
done;

echo "Running all"
cmd="python run_benchmarks.py --extractor_type=all --output_dir=output/mega/results/benchmarking/ --feature_type inv-dist cartesian_rot compact-dist --noise_level 1e-2 5e-2 --test_model linear non-linear --accuracy_method mse relevant_fraction --overwrite=False;"
echo $cmd
$cmd

echo "Done"