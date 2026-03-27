#!/bin/bash
# 脚本生成器

NUM_BATCH=100
OUTPUT_DIR="./scrpits_ss_dataset"

mkdir -p $OUTPUT_DIR

for i in $(seq -f "%03g" 0 $((NUM_BATCH-1))); do
    cat <<EOF > $OUTPUT_DIR/run_batch_$i.sh
#!/bin/bash
module load cmake/3.16.9
export GCC_HOME=/home/wllpro/llwang/yfdai/gcc11/gcc11
export PATH=$GCC_HOME/bin:$PATH
export LD_LIBRARY_PATH=$GCC_HOME/lib64:$GCC_HOME/lib:$LD_LIBRARY_PATH
source /home/wllpro/llwang/yfdai/env/dgl_mat/bin/activate  # 替换为你的环境名
python process_one_batch_ss.py --batch_index $i
EOF
    chmod +x $OUTPUT_DIR/run_batch_$i.sh
done

echo "✅ 生成了 $NUM_BATCH 个脚本：$OUTPUT_DIR/run_batch_*.sh"
