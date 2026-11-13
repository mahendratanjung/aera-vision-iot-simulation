#!/usr/bin/env bash
# Download the test data and the published model weights used by the evaluation, and check them.
#   bash scripts/get_data.sh        (from model/; about 3.4 GB, CrowdHuman is 2.5 GB of it)
# Ultralytics and RF-DETR weights download themselves on first use (official release assets).
set -euo pipefail
cd "$(dirname "$0")/.."

check() { echo "$2  $1" | shasum -a 256 -c -; }
fetch() { [ -f "$2" ] || { mkdir -p "$(dirname "$2")"; curl -fSL -o "$2" "$1"; }; check "$2" "$3"; }

# CrowdHuman validation set, from the authors' Hugging Face copy (CC BY-NC 4.0, non-commercial research).
CH=https://huggingface.co/datasets/sshao0516/CrowdHuman/resolve/main
fetch "$CH/annotation_val.odgt" data/raw/crowdhuman/annotation_val.odgt be422c79a190ff7e30fe5cbd74cbf45a2dadda6c5af58c6ec11a038ba2993c04
if [ ! -d data/raw/crowdhuman/Images ]; then
  fetch "$CH/CrowdHuman_val.zip" data/raw/crowdhuman/CrowdHuman_val.zip c0ab99bb80ac162cd3efdf94a1a6100c4f059a61d69596412c6b44ebc20d1363
  (cd data/raw/crowdhuman && unzip -q CrowdHuman_val.zip && rm CrowdHuman_val.zip)
fi

# ZIP density counters (Ma et al., 2025; MIT): published checkpoints.
Z=https://github.com/Yiming-M/ZIP/releases/download
for spec in "weights_nwpu/ebc_n.zip nwpu/ebc_n" "weights_nwpu/ebc_t.zip nwpu/ebc_t"; do
  set -- $spec
  if [ ! -f "weights/zip/$2/best_mae.pth" ]; then
    mkdir -p "weights/zip/$2" && curl -fSL -o /tmp/aera_zip.zip "$Z/$1" && unzip -oq /tmp/aera_zip.zip -d "weights/zip/$2" && rm /tmp/aera_zip.zip
  fi
done
if [ ! -f weights/zip/shb/ebc_t_best/best_mae.pth ]; then
  mkdir -p weights/zip/shb && curl -fSL -o /tmp/aera_zip.zip "$Z/weights_shb/ebc_t.zip" && unzip -oq /tmp/aera_zip.zip -d weights/zip/shb && rm /tmp/aera_zip.zip
fi
check weights/zip/nwpu/ebc_n/best_mae.pth 30474977b7979394469802b0b5e57476391252c86a1e5fd3248aa255d28d817e
check weights/zip/nwpu/ebc_t/best_mae.pth 680227fab0485b9d2abdc77776544e31ae03a540fe45d82a7d018f2e5c742e7f
check weights/zip/shb/ebc_t_best/best_mae.pth e3df1534528758f42948e38904b035a105b5ae01664f9976afb481c4608d3504

# YOLOv5m trained on CrowdHuman (person + head), community ONNX export.
fetch https://github.com/yakhyo/yolov5-crowdhuman-onnx/releases/download/v0.0.1/crowdhuman.onnx \
  weights/yolov5_crowdhuman/crowdhuman.onnx 4f2b3a96231891de06df6838b4437fe68e9ae2df1ad1e1714cf700be9ea1561e

echo "Data and weights ready."
