#!/bin/bash --login
#SBATCH --time=2:00:00
#SBATCH --mem=0
#SBATCH --cpus-per-task=40
#SBATCH --constraint=cpu_intel_gold_6248
#SBATCH --partition=batch
#SBATCH --job-name=alphafold-casp13
#SBATCH --output=alphafold_casp13/results/%x/slurm-%j.out
#SBATCH --error=alphafold_casp13/results/%x/slurm-%j.err

# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

# We assume the script is being run from the deepmind_research/ directory.

# Copy the archived data from /ibex/scratch to local storage on the compute node
cp -r alphafold_casp13/data/ /tmp/
unzip -oq /tmp/data/alphafold-casp13-data.zip -d /tmp/data/
unzip -oq /tmp/data/alphafold-casp13-weights.zip -d /tmp/data/

DISTOGRAM_MODEL="/tmp/data/873731"  # Path to the directory with the distogram model.
BACKGROUND_MODEL="/tmp/data/916425"  # Path to the directory with the background model.
TORSION_MODEL="/tmp/data/941521"  # Path to the directory with the torsion model.

TARGET="T1019s2"  # The name of the target.
TARGET_PATH=/tmp/data/${TARGET}  # Path to the directory with the target input data.

# Set up the virtual environment
conda activate alphafold_casp13/env

# Create the output directory on /ibex/scratch (for now)
OUTPUT_DIR=alphafold_casp13/results/$SLURM_JOB_NAME/$SLURM_JOB_ID/contacts_${TARGET}_$(date +%Y_%m_%d_%H_%M_%S)
mkdir -p "${OUTPUT_DIR}"
echo "Saving output to ${OUTPUT_DIR}/"

# Set TF XLA flags to make use of XLA compiler
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit

# Run contact prediction over 4 replicas.
for replica in 0 1 2 3; do
  echo "Launching all models for replica ${replica}"

  # Run the distogram model.
  python3 -m alphafold_casp13.contacts \
    --logtostderr \
    --cpu=true \
    --config_path="${DISTOGRAM_MODEL}/${replica}/config.json" \
    --checkpoint_path="${DISTOGRAM_MODEL}/${replica}/tf_graph_data/tf_graph_data.ckpt" \
    --output_path="${OUTPUT_DIR}/distogram/${replica}" \
    --eval_sstable="${TARGET_PATH}/${TARGET}.tfrec" \
    --stats_file="${DISTOGRAM_MODEL}/stats_train_s35.json" &

  # Run the background model.
  python3 -m alphafold_casp13.contacts \
    --logtostderr \
    --cpu=true \
    --config_path="${BACKGROUND_MODEL}/${replica}/config.json" \
    --checkpoint_path="${BACKGROUND_MODEL}/${replica}/tf_graph_data/tf_graph_data.ckpt" \
    --output_path="${OUTPUT_DIR}/background_distogram/${replica}" \
    --eval_sstable="${TARGET_PATH}/${TARGET}.tfrec" \
    --stats_file="${BACKGROUND_MODEL}/stats_train_s35.json" &
done

# Run the torsion model, but only 1 replica.
python3 -m alphafold_casp13.contacts \
  --logtostderr \
  --cpu=true \
  --config_path="${TORSION_MODEL}/0/config.json" \
  --checkpoint_path="${TORSION_MODEL}/0/tf_graph_data/tf_graph_data.ckpt" \
  --output_path="${OUTPUT_DIR}/torsion/0" \
  --eval_sstable="${TARGET_PATH}/${TARGET}.tfrec" \
  --stats_file="${TORSION_MODEL}/stats_train_s35.json" &

echo "All models running, waiting for them to complete"
wait

echo "Ensembling all replica outputs"

# Run the ensembling jobs for distograms, background distograms.
for output_dir in "${OUTPUT_DIR}/distogram" "${OUTPUT_DIR}/background_distogram"; do
  pickle_dirs="${output_dir}/0/pickle_files/,${output_dir}/1/pickle_files/,${output_dir}/2/pickle_files/,${output_dir}/3/pickle_files/"

  # Ensemble distograms.
  python3 -m alphafold_casp13.ensemble_contact_maps \
    --logtostderr \
    --pickle_dirs="${pickle_dirs}" \
    --output_dir="${output_dir}/ensemble/"
done

# Only ensemble single replica distogram for torsions.
python3 -m alphafold_casp13.ensemble_contact_maps \
  --logtostderr \
  --pickle_dirs="${OUTPUT_DIR}/torsion/0/pickle_files/" \
  --output_dir="${OUTPUT_DIR}/torsion/ensemble/"

echo "Pasting contact maps"

python3 -m alphafold_casp13.paste_contact_maps \
  --logtostderr \
  --pickle_input_dir="${OUTPUT_DIR}/distogram/ensemble/" \
  --output_dir="${OUTPUT_DIR}/pasted/" \
  --tfrecord_path="${TARGET_PATH}/${TARGET}.tfrec"

echo "Done"
