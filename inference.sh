for i in $(seq -f "%04g" 0 24)
do 
    $(python inference.py \
        --variant shufflenetv2 \
        --checkpoint "./checkpoint/2022.05.19/stage2/epoch-19.pth" \
        --device cuda \
        --input-source "../result/gt/videomatte_512x288/videomatte_motion/${i}/com" \
        --downsample-ratio 1.0 \
        --output-type png_sequence \
        --output-composition "../result/05.20/videomatte_512x288/videomatte_motion/${i}/com" \
        --output-alpha "../result/05.20/videomatte_512x288/videomatte_motion/${i}/pha" \
        --output-foreground "../result/05.20/videomatte_512x288/videomatte_motion/${i}/fgr" \
        --seq-chunk 5)
    $(python inference.py \
        --variant shufflenetv2 \
        --checkpoint "./checkpoint/2022.05.19/stage2/epoch-19.pth" \
        --device cuda \
        --input-source "../result/gt/videomatte_512x288/videomatte_static/${i}/com" \
        --downsample-ratio 1.0 \
        --output-type png_sequence \
        --output-composition "../result/05.20/videomatte_512x288/videomatte_static/${i}/com" \
        --output-alpha "../result/05.20/videomatte_512x288/videomatte_static/${i}/pha" \
        --output-foreground "../result/05.20/videomatte_512x288/videomatte_static/${i}/fgr" \
        --seq-chunk 5)
done
