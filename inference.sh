for i in $(seq -f "%04g" 0 24)
do 
    $(python inference.py \
        --variant mobilenetv3 \
        --checkpoint "./checkpoint/2022.06.21/stage3/epoch-29.pth" \
        --device cuda \
        --input-source "../result/HR/GT/videomatte_1920x1080/videomatte_motion/${i}/com" \
        --downsample-ratio 0.25 \
        --output-type png_sequence \
        --output-composition "../result/HR/06.26/videomatte_1920x1080/videomatte_motion/${i}/com" \
        --output-alpha "../result/HR/06.26/videomatte_1920x1080/videomatte_motion/${i}/pha" \
        --output-foreground "../result/HR/06.26/videomatte_1920x1080/videomatte_motion/${i}/fgr" \
        --seq-chunk 5)
    $(python inference.py \
        --variant mobilenetv3 \
        --checkpoint "./checkpoint/2022.06.21/stage3/epoch-29.pth" \
        --device cuda \
        --input-source "../result/HR/GT/videomatte_1920x1080/videomatte_static/${i}/com" \
        --downsample-ratio 0.25 \
        --output-type png_sequence \
        --output-composition "../result/HR/06.26/videomatte_1920x1080/videomatte_static/${i}/com" \
        --output-alpha "../result/HR/06.26/videomatte_1920x1080/videomatte_static/${i}/pha" \
        --output-foreground "../result/HR/06.26/videomatte_1920x1080/videomatte_static/${i}/fgr" \
        --seq-chunk 5)
done
