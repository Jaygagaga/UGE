#!/bin/bash

# Script to run eval_urban_ranking_multiview.py on datasets in different benchmark folders
# This script runs evaluation in two modes for each folder:
#   - image_only: Evaluates using image-only queries
#   - with_graph: Evaluates using graph/fusion queries
#
# For each benchmark folder, it runs 2 blocks (image_only + with_graph):
#   1-2. geolocation folder (image_only, with_graph)
#   3-4. perception folder (image_only, with_graph)
#   5-6. retrieval folder (image_only, with_graph)
#   7-8. spatial_reasoning/by_type_paris (image_only, with_graph)
#   9-10. spatial_reasoning/by_type_beijing (image_only, with_graph)
#   11-12. spatial_reasoning/by_type_newyork (image_only, with_graph)
#   13-14. spatial_reasoning/by_type_singapore (image_only, with_graph)

# Change to project root directory to ensure relative paths work
# Get the directory where this script is located, then go to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}" || exit 1
echo "Changed to project root: ${PROJECT_ROOT}"

# Set PYTHONPATH to ensure Python can find the swift modules
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Set base paths (relative to project root)
BENCHMARK_DIR="${PROJECT_ROOT}/benchmark"
#OUTPUT_DIR="eval_output/stage2_qwen2vl2b"
#OUTPUT_DIR="eval_output/stage2_qwen25vl7b_sr_re"
OUTPUT_DIR="eval_output/stage2_qwen25vl7b"
#OUTPUT_DIR="eval_output/stage2_qwen25vl3b_sr1"
#OUTPUT_DIR="eval_output/stage2_gme2b"
#OUTPUT_DIR="eval_output/stage2_vlm2b_sr"
#OUTPUT_DIR="eval_output/stage2_llava"
#OUTPUT_DIR="eval_output/stage2_internvl31b"
#OUTPUT_DIR="eval_output/stage2_qwen2vl7b_sr1"
#OUTPUT_DIR="eval_output/stage2_qwen25vl7b_no_edge_feature_sr"
#OUTPUT_DIR="eval_output/stage2_qwen25vl7b_only_node_encoder_sr"
#OUTPUT_DIR="eval_output/stage2_qwen25vl7b_no_coords_encoding_sr"
#OUTPUT_DIR="eval_output/stage2_qwen2vl7b_edge_feature1"
#OUTPUT_DIR="eval_output/stage2_phi_edge_feature_sr"
# Model and adapter configuration
#MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
#MODEL_NAME='OpenGVLab/InternVL3-1B'
#MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
#MODEL_NAME="iic/gme-Qwen2-VL-2B-Instruct"
#MODEL_NAME="iic/gme-Qwen2-VL-7B-Instruct"
#MODEL_NAME="microsoft/Phi-3-vision-128k-instruct"
#MODEL_NAME='llava-hf/llava-v1.6-mistral-7b-hf'
#ADAPTERS_PATH="output/stage2_vlm2vec7b_edge_feature/v1-20251225-172124/20251225-172124_Qwen2_VL_7B_Instruct_graph_spatial_edgefeat_gat4_infonce/checkpoint-4025/" # UPDATE THIS PATH
ADAPTERS_PATH="output/stage2_qwen25vl7b_edge_feature/v0-20251219-205258/20251219-205258_Qwen2.5_VL_7B_Instruct_graph_spatial_edgefeat_gat4_infonce/checkpoint-4025/"
#ADAPTERS_PATH="output/stage2_qwen25vl7b_only_node_encoder/v0-20251226-181810/20251226-181811_Qwen2.5_VL_7B_Instruct_graph_gat4_infonce/checkpoint-4025/" # UPDATE THIS PATH
#ADAPTERS_PATH="output/stage2_qwen25vl7b_no_edge_feature/v1-20251216-105341/20251216-105342_Qwen2.5_VL_7B_Instruct_graph_spatial_infonce/checkpoint-4025/"
#ADAPTERS_PATH="output/stage2_qwen2vl7b_edge_feature/v0-20251221-130835/20251221-130835_Qwen2_VL_7B_Instruct_graph_spatial_edgefeat_gat4_infonce/checkpoint-4025/"
#ADAPTERS_PATH="output/stage2_phi_edge_feature/v0-20251231-105245/20251231-105245_Phi_3_vision_128k_instruct_graph_spatial_edgefeat_gat4_infonce/checkpoint-4025/"
#ADAPTERS_PATH="output/stage2_qwen2vl2b/v0-20260104-172358/20260104-172359_Qwen2_VL_2B_Instruct_graph_spatial_edgefeat_gat4_infonce/checkpoint-1610/"
#ADAPTERS_PATH="output/stage2_qwen253b/v1-20260103-233647/20260103-233647_Qwen2.5_VL_3B_Instruct_graph_spatial_edgefeat_gat4_infonce/checkpoint-1610/"
#ADAPTERS_PATH="output/stage2_vlm2vec2b/v0-20260110-103204/20260110-103205_Qwen2_VL_2B_Instruct_graph_spatial_edgefeat_gat4_infonce/checkpoint-2015/"
#ADAPTERS_PATH="output/stage2_gme2b/v2-20260106-221102/20260106-221103_gme_Qwen2_VL_2B_Instruct_graph_spatial_edgefeat_gat4_infonce/checkpoint-800/"
#ADAPTERS_PATH="output/stage2_qwen25vl7b_stage2_only/v0-20260116-235821/20260116-235822_Qwen2.5_VL_7B_Instruct_graph_spatial_edgefeat_gat4_infonce/checkpoint-4025/"
#ADAPTERS_PATH="output/stage2_internvl31b/v2-20260120-170729/20260120-170730_InternVL3_1B_graph_spatial_edgefeat_gat4_infonce/checkpoint-5365/"
#ADAPTERS_PATH='llava-hf/llava-v1.6-mistral-7b-hf'
#ADAPTERS_PATH="output/stage2_qwen25vl7b_only_node_encoder/v0-20251226-181810/20251226-181811_Qwen2.5_VL_7B_Instruct_graph_gat4_infonce/checkpoint-4025/"
#ADAPTERS_PATH="output/stage2_qwen25vl7b_no_coords_encoding/v1-20260107-174905/20260107-174905_Qwen2.5_VL_7B_Instruct_graph_edgefeat_gat4_infonce/checkpoint-4025/"
#ADAPTERS_PATH="output/stage2_llava/"
#ADAPTERS_PATH="output/stage2_qwen25vl7b_lr_scale1/v0-20260114-234248/20260114-234249_Qwen2.5_VL_7B_Instruct_graph_spatial_edgefeat_gat4_infonce/checkpoint-4025/"
#ADAPTERS_PATH="output/stage2_qwen25vl7b_lr_scale05/v0-20260113-235139/20260113-235140_Qwen2.5_VL_7B_Instruct_graph_spatial_edgefeat_gat4_infonce/checkpoint-4025/"
#ADAPTERS_PATH="output/stage2_qwen25vl7b_edge_embed_8/v0-20260125-131213/20260125-131213_Qwen2.5_VL_7B_Instruct_graph_spatial_edgefeat_gat4_infonce/checkpoint-5370/"
#ADAPTERS_PATH="output/stage2_qwen25vl7b_edge_embed_32/v0-20260126-161913/20260126-161914_Qwen2.5_VL_7B_Instruct_graph_spatial_edgefeat_gat4_infonce/checkpoint-5370/"
#ADAPTERS_PATH="output/stage2_qwen25vl7b_edge_embed_16/v0-20260124-123558/20260124-123558_Qwen2.5_VL_7B_Instruct_graph_spatial_edgefeat_gat4_infonce/checkpoint-5370/"
BATCH_SIZE=1
DEVICE_MAP="cuda:0"
GRAPH_ROOT=""  # Set to empty if graphs are already in JSONL, or provide path if needed

# Inference parameters
MAX_PIXELS=600000
GRAPH_MAX_NODES=900
TORCH_DTYPE="bfloat16"
ATTN_IMPL="eager"   #"sdpa" #eager
K_VALUES="1,3,5,10"
EVAL_LIMIT=700  # Limit evaluation to 600 samples per dataset

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Running eval_urban_ranking_multiview.py on all benchmark folders"
echo "Model: ${MODEL_NAME}"
echo "Adapters: ${ADAPTERS_PATH}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

BLOCK_NUM=0

# ============================================================================
# GEOLOCATION FOLDER
# ============================================================================
#
## Block 1: geolocation folder - image_only
###--filename_pattern "image_only"
#BLOCK_NUM=$((BLOCK_NUM + 1))
#echo "----------------------------------------"
#echo "Block ${BLOCK_NUM}/14: Processing geolocation folder (image_only)"
#echo "----------------------------------------"
python examples/eval/embedding/eval_urban_ranking_multiview.py \
  --model "${MODEL_NAME}" \
  --adapters "${ADAPTERS_PATH}" \
  --benchmark_dir "${BENCHMARK_DIR}/geolocation" \
  --filename_pattern "image_only" \
  --query_mode "image" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size ${BATCH_SIZE} \
  --device_map "${DEVICE_MAP}" \
  --max_pixels ${MAX_PIXELS} \
  --graph_max_nodes ${GRAPH_MAX_NODES} \
  --torch_dtype "${TORCH_DTYPE}" \
  --attn_impl "${ATTN_IMPL}" \
  --k_values "${K_VALUES}" \
  --eval_limit ${EVAL_LIMIT} \
#
##if [ $? -ne 0 ]; then
##    echo "⚠️  WARNING: Block ${BLOCK_NUM} (geolocation/image_only) failed! Continuing with next block..."
##    # Don't exit - continue with other blocks
##fi
##
##echo ""
##echo "✓ Block ${BLOCK_NUM} completed"
##echo ""
###
#
#
## ============================================================================
## PERCEPTION FOLDER
## ============================================================================
##
### Block 3: perception folder - image_only
##BLOCK_NUM=$((BLOCK_NUM + 1))
##echo "----------------------------------------"
##echo "Block ${BLOCK_NUM}/14: Processing perception folder (image_only)"
##echo "----------------------------------------"
#python examples/eval/embedding/eval_urban_ranking_multiview.py \
#  --model "${MODEL_NAME}" \
#  --adapters "${ADAPTERS_PATH}" \
#  --benchmark_dir "${BENCHMARK_DIR}/perception_old" \
#  --filename_pattern "image_only" \
#  --query_mode "image" \
#  --output_dir "${OUTPUT_DIR}" \
#  --batch_size ${BATCH_SIZE} \
#  --device_map "${DEVICE_MAP}" \
#  --max_pixels ${MAX_PIXELS} \
#  --graph_max_nodes ${GRAPH_MAX_NODES} \
#  --torch_dtype "${TORCH_DTYPE}" \
#  --attn_impl "${ATTN_IMPL}" \
#  --k_values "${K_VALUES}" \
#  --eval_limit ${EVAL_LIMIT}
##
##if [ $? -ne 0 ]; then
##    echo "⚠️  WARNING: Block ${BLOCK_NUM} (perception/image_only) failed! Continuing with next block..."
##    # Continue to next block instead of exiting
##fi
##
##echo ""
##echo "✓ Block ${BLOCK_NUM} completed"
##echo ""
##--filename_pattern "with_graph_filtered_389" \
### Block 4: perception folder - with_graph
#BLOCK_NUM=$((BLOCK_NUM + 1))
#echo "----------------------------------------"
#echo "Block ${BLOCK_NUM}/14: Processing perception folder (with_graph)"
#echo "----------------------------------------"
python examples/eval/embedding/eval_urban_ranking_multiview.py \
      --model "${MODEL_NAME}" \
      --adapters "${ADAPTERS_PATH}" \
      --benchmark_dir "${BENCHMARK_DIR}/perception" \
      --filename_pattern "with_graph_filtered_389" \
      --query_mode "fusion" \
      --output_dir "${OUTPUT_DIR}" \
      --batch_size ${BATCH_SIZE} \
      --device_map "${DEVICE_MAP}" \
      --max_pixels ${MAX_PIXELS} \
      --graph_max_nodes ${GRAPH_MAX_NODES} \
      --torch_dtype "${TORCH_DTYPE}" \
      --attn_impl "${ATTN_IMPL}" \
      --k_values "${K_VALUES}" \
      --eval_limit ${EVAL_LIMIT}
#fi
#
#if [ $? -ne 0 ]; then
#    echo "⚠️  WARNING: Block ${BLOCK_NUM} (perception/with_graph) failed! Continuing with next block..."
#    # Continue to next block instead of exiting
#fi
#
#echo ""
#echo "✓ Block ${BLOCK_NUM} completed"
#echo ""
#
## ============================================================================
## RETRIEVAL FOLDER (uses eval_image_retrieval.py instead of eval_urban_ranking_multiview.py)
## ============================================================================
#
### Block 5: retrieval folder - image_only (text queries)
##BLOCK_NUM=$((BLOCK_NUM + 1))
##echo "----------------------------------------"
##echo "Block ${BLOCK_NUM}/14: Processing retrieval folder (image_only, text queries)"
##echo "----------------------------------------"
#python examples/eval/embedding/eval_image_retrieval.py \
#  --model "${MODEL_NAME}" \
#  --adapters "${ADAPTERS_PATH}" \
#  --benchmark_dir "${BENCHMARK_DIR}/retrieval" \
#  --filename_pattern "image_only" \
#  --query_mode "text" \
#  --output_dir "${OUTPUT_DIR}" \
#  --batch_size ${BATCH_SIZE} \
#  --device_map "${DEVICE_MAP}" \
#  --max_pixels ${MAX_PIXELS} \
#  --graph_max_nodes ${GRAPH_MAX_NODES} \
#  --torch_dtype "${TORCH_DTYPE}" \
#  --attn_impl "${ATTN_IMPL}" \
#  --k_values "${K_VALUES}" \
#  --eval_limit ${EVAL_LIMIT}
##
##if [ $? -ne 0 ]; then
##    echo "⚠️  WARNING: Block ${BLOCK_NUM} (retrieval/image_only) failed! Continuing with next block..."
##    # Continue to next block instead of exiting
##fi
##
##echo ""
##echo "✓ Block ${BLOCK_NUM} completed"
##echo ""
###
#### Block 6: retrieval folder - with_graph (fusion queries)
#BLOCK_NUM=$((BLOCK_NUM + 1))
#echo "----------------------------------------"
#echo "Block ${BLOCK_NUM}/14: Processing retrieval folder (with_graph, fusion queries)"
#echo "----------------------------------------"
python examples/eval/embedding/eval_image_retrieval.py \
  --model "${MODEL_NAME}" \
  --adapters "${ADAPTERS_PATH}" \
  --benchmark_dir "${BENCHMARK_DIR}/retrieval" \
  --filename_pattern "with_graph" \
  --query_mode "fusion" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size 1 \
  --device_map "${DEVICE_MAP}" \
  --max_pixels ${MAX_PIXELS} \
  --graph_max_nodes ${GRAPH_MAX_NODES} \
  --torch_dtype "${TORCH_DTYPE}" \
  --attn_impl "${ATTN_IMPL}" \
  --k_values "${K_VALUES}" \
  --eval_limit ${EVAL_LIMIT} \
  --fusion_max_text_length 3000

#if [ $? -ne 0 ]; then
#    echo "⚠️  WARNING: Block ${BLOCK_NUM} (retrieval/with_graph) failed! Continuing with next block..."
#    # Continue to next block instead of exiting
#fi
#
#echo ""
#echo "✓ Block ${BLOCK_NUM} completed"
#echo ""
#
## ============================================================================
## SPATIAL_REASONING/BY_TYPE_PARIS
## ============================================================================
#
## Block 7: spatial_reasoning/by_type_paris - image_only
#BLOCK_NUM=$((BLOCK_NUM + 1))
#echo "----------------------------------------"
#echo "Block ${BLOCK_NUM}/14: Processing spatial_reasoning/by_type_paris (image_only)"
#echo "----------------------------------------"
#python examples/eval/embedding/eval_urban_ranking_multiview.py \
#  --model "${MODEL_NAME}" \
#  --adapters "${ADAPTERS_PATH}" \
#  --benchmark_dir "${BENCHMARK_DIR}/spatial_reasoning/by_type_paris" \
#  --filename_pattern "image_only" \
#  --query_mode "image" \
#  --output_dir "${OUTPUT_DIR}" \
#  --batch_size ${BATCH_SIZE} \
#  --device_map "${DEVICE_MAP}" \
#  --max_pixels ${MAX_PIXELS} \
#  --graph_max_nodes ${GRAPH_MAX_NODES} \
#  --torch_dtype "${TORCH_DTYPE}" \
#  --attn_impl "${ATTN_IMPL}" \
#  --k_values "${K_VALUES}" \
#  --eval_limit ${EVAL_LIMIT}
#
#if [ $? -ne 0 ]; then
#    echo "⚠️  WARNING: Block ${BLOCK_NUM} (spatial_reasoning/by_type_paris/image_only) failed! Continuing with next block..."
#    # Continue to next block instead of exiting
#fi
#
#echo ""
#echo "✓ Block ${BLOCK_NUM} completed"
#echo ""
##
#### Block 8: spatial_reasoning/by_type_paris - with_graph
##BLOCK_NUM=$((BLOCK_NUM + 1))
##echo "----------------------------------------"
##echo "Block ${BLOCK_NUM}/14: Processing spatial_reasoning/by_type_paris (with_graph)"
##echo "----------------------------------------"
python examples/eval/embedding/eval_urban_ranking_multiview.py \
      --model "${MODEL_NAME}" \
      --adapters "${ADAPTERS_PATH}" \
      --benchmark_dir "${BENCHMARK_DIR}/spatial_reasoning/by_type_paris" \
      --filename_pattern "with_graph" \
      --query_mode "fusion" \
      --output_dir "${OUTPUT_DIR}" \
      --batch_size ${BATCH_SIZE} \
      --device_map "${DEVICE_MAP}" \
      --max_pixels ${MAX_PIXELS} \
      --graph_max_nodes ${GRAPH_MAX_NODES} \
      --torch_dtype "${TORCH_DTYPE}" \
      --attn_impl "${ATTN_IMPL}" \
      --k_values "${K_VALUES}" \
      --eval_limit ${EVAL_LIMIT} \
####
####if [ $? -ne 0 ]; then
####    echo "⚠️  WARNING: Block ${BLOCK_NUM} (spatial_reasoning/by_type_paris/with_graph) failed! Continuing with next block..."
####    # Continue to next block instead of exiting
####fi
####
#####echo ""
#####echo "✓ Block ${BLOCK_NUM} completed"
#####echo ""
#####
###### ============================================================================
###### SPATIAL_REASONING/BY_TYPE_BEIJING
###### ============================================================================
#####
###### Block 9: spatial_reasoning/by_type_beijing - image_only
#####BLOCK_NUM=$((BLOCK_NUM + 1))
#####echo "----------------------------------------"
#####echo "Block ${BLOCK_NUM}/14: Processing spatial_reasoning/by_type_beijing (image_only)"
#####echo "----------------------------------------"
#####python examples/eval/embedding/eval_urban_ranking_multiview.py \
#####  --model "${MODEL_NAME}" \
#####  --adapters "${ADAPTERS_PATH}" \
#####  --benchmark_dir "${BENCHMARK_DIR}/spatial_reasoning/by_type_beijing" \
#####  --filename_pattern "image_only" \
#####  --query_mode "image" \
#####  --output_dir "${OUTPUT_DIR}" \
#####  --batch_size ${BATCH_SIZE} \
#####  --device_map "${DEVICE_MAP}" \
#####  --max_pixels ${MAX_PIXELS} \
#####  --graph_max_nodes ${GRAPH_MAX_NODES} \
#####  --torch_dtype "${TORCH_DTYPE}" \
#####  --attn_impl "${ATTN_IMPL}" \
#####  --k_values "${K_VALUES}" \
#####  --eval_limit ${EVAL_LIMIT}
#####
#####if [ $? -ne 0 ]; then
#####    echo "⚠️  WARNING: Block ${BLOCK_NUM} (spatial_reasoning/by_type_beijing/image_only) failed! Continuing with next block..."
#####    # Continue to next block instead of exiting
#####fi
#####
#####echo ""
#####echo "✓ Block ${BLOCK_NUM} completed"
#####echo ""
#####
######### Block 10: spatial_reasoning/by_type_beijing - with_graph
####BLOCK_NUM=$((BLOCK_NUM + 1))
####echo "----------------------------------------"
####echo "Block ${BLOCK_NUM}/14: Processing spatial_reasoning/by_type_beijing (with_graph)"
####echo "----------------------------------------"
####if [ -n "${GRAPH_ROOT}" ]; then
python examples/eval/embedding/eval_urban_ranking_multiview.py \
      --model "${MODEL_NAME}" \
      --adapters "${ADAPTERS_PATH}" \
      --benchmark_dir "${BENCHMARK_DIR}/spatial_reasoning/by_type_beijing" \
      --filename_pattern "with_graph" \
      --query_mode "fusion" \
      --output_dir "${OUTPUT_DIR}" \
      --batch_size ${BATCH_SIZE} \
      --device_map "${DEVICE_MAP}" \
      --max_pixels ${MAX_PIXELS} \
      --graph_max_nodes ${GRAPH_MAX_NODES} \
      --torch_dtype "${TORCH_DTYPE}" \
      --attn_impl "${ATTN_IMPL}" \
      --k_values "${K_VALUES}" \
      --eval_limit ${EVAL_LIMIT}
####fi
####
####
####
#####
###### ============================================================================
###### SPATIAL_REASONING/BY_TYPE_NEWYORK
###### ============================================================================
#####
###### Block 11: spatial_reasoning/by_type_newyork - image_only
#####BLOCK_NUM=$((BLOCK_NUM + 1))
#####echo "----------------------------------------"
#####echo "Block ${BLOCK_NUM}/14: Processing spatial_reasoning/by_type_newyork (image_only)"
#####echo "----------------------------------------"
#####python examples/eval/embedding/eval_urban_ranking_multiview.py \
#####  --model "${MODEL_NAME}" \
#####  --adapters "${ADAPTERS_PATH}" \
#####  --benchmark_dir "${BENCHMARK_DIR}/spatial_reasoning/by_type_newyork" \
#####  --filename_pattern "image_only" \
#####  --query_mode "image" \
#####  --output_dir "${OUTPUT_DIR}" \
#####  --batch_size ${BATCH_SIZE} \
#####  --device_map "${DEVICE_MAP}" \
#####  --max_pixels ${MAX_PIXELS} \
#####  --graph_max_nodes ${GRAPH_MAX_NODES} \
#####  --torch_dtype "${TORCH_DTYPE}" \
#####  --attn_impl "${ATTN_IMPL}" \
#####  --k_values "${K_VALUES}" \
#####  --eval_limit ${EVAL_LIMIT}
#####
#####if [ $? -ne 0 ]; then
#####    echo "⚠️  WARNING: Block ${BLOCK_NUM} (spatial_reasoning/by_type_newyork/image_only) failed! Continuing with next block..."
#####    # Continue to next block instead of exiting
#####fi
#####
#####echo ""
#####echo "✓ Block ${BLOCK_NUM} completed"
#####echo ""
######
###### Block 12: spatial_reasoning/by_type_newyork - with_graph
####BLOCK_NUM=$((BLOCK_NUM + 1))
####echo "----------------------------------------"
####echo "Block ${BLOCK_NUM}/14: Processing spatial_reasoning/by_type_newyork (with_graph)"
####echo "----------------------------------------"
python examples/eval/embedding/eval_urban_ranking_multiview.py \
      --model "${MODEL_NAME}" \
      --adapters "${ADAPTERS_PATH}" \
      --benchmark_dir "${BENCHMARK_DIR}/spatial_reasoning/by_type_newyork" \
      --filename_pattern "with_graph" \
      --query_mode "fusion" \
      --output_dir "${OUTPUT_DIR}" \
      --batch_size ${BATCH_SIZE} \
      --device_map "${DEVICE_MAP}" \
      --max_pixels ${MAX_PIXELS} \
      --graph_max_nodes ${GRAPH_MAX_NODES} \
      --torch_dtype "${TORCH_DTYPE}" \
      --attn_impl "${ATTN_IMPL}" \
      --k_values "${K_VALUES}" \
      --eval_limit ${EVAL_LIMIT}
####fi
####
####
####
#####
###### ============================================================================
###### SPATIAL_REASONING/BY_TYPE_SINGAPORE
###### ============================================================================
#####
###### Block 13: spatial_reasoning/by_type_singapore - image_only
#####BLOCK_NUM=$((BLOCK_NUM + 1))
#####echo "----------------------------------------"
#####echo "Block ${BLOCK_NUM}/14: Processing spatial_reasoning/by_type_singapore (image_only)"
#####echo "----------------------------------------"
#####python examples/eval/embedding/eval_urban_ranking_multiview.py \
#####  --model "${MODEL_NAME}" \
#####  --adapters "${ADAPTERS_PATH}" \
#####  --benchmark_dir "${BENCHMARK_DIR}/spatial_reasoning/by_type_singapore" \
#####  --filename_pattern "image_only" \
#####  --query_mode "image" \
#####  --output_dir "${OUTPUT_DIR}" \
#####  --batch_size ${BATCH_SIZE} \
#####  --device_map "${DEVICE_MAP}" \
#####  --max_pixels ${MAX_PIXELS} \
#####  --graph_max_nodes ${GRAPH_MAX_NODES} \
#####  --torch_dtype "${TORCH_DTYPE}" \
#####  --attn_impl "${ATTN_IMPL}" \
#####  --k_values "${K_VALUES}" \
#####  --eval_limit ${EVAL_LIMIT}
#####
#####if [ $? -ne 0 ]; then
#####    echo "⚠️  WARNING: Block ${BLOCK_NUM} (spatial_reasoning/by_type_singapore/image_only) failed! Continuing with next block..."
#####    # Continue to next block instead of exiting
#####fi
#####
#####echo ""
#####echo "✓ Block ${BLOCK_NUM} completed"
#####echo ""
#####
###### Block 14: spatial_reasoning/by_type_singapore - with_graph
####BLOCK_NUM=$((BLOCK_NUM + 1))
####echo "----------------------------------------"
####echo "Block ${BLOCK_NUM}/14: Processing spatial_reasoning/by_type_singapore (with_graph)"
####echo "----------------------------------------"
python examples/eval/embedding/eval_urban_ranking_multiview.py \
      --model "${MODEL_NAME}" \
      --adapters "${ADAPTERS_PATH}" \
      --benchmark_dir "${BENCHMARK_DIR}/spatial_reasoning/by_type_singapore" \
      --filename_pattern "with_graph" \
      --query_mode "fusion" \
      --output_dir "${OUTPUT_DIR}" \
      --batch_size ${BATCH_SIZE} \
      --device_map "${DEVICE_MAP}" \
      --max_pixels ${MAX_PIXELS} \
      --graph_max_nodes ${GRAPH_MAX_NODES} \
      --torch_dtype "${TORCH_DTYPE}" \
      --attn_impl "${ATTN_IMPL}" \
      --k_values "${K_VALUES}" \
      --eval_limit ${EVAL_LIMIT}
#fi
#
#
#
##
#echo "=========================================="
#echo "All blocks completed!"
#echo "Results saved to: ${OUTPUT_DIR}"
#echo "=========================================="
#echo ""
#echo "Note: Some blocks may have been skipped if datasets were not found."
#echo "      Check the logs above for details."

