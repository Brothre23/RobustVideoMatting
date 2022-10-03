for i in $(seq -f "%04g" 0 24)
do 
    $(python inference.py \
        --variant mobilenetv3 \
        --checkpoint "./checkpoint/PointRend-confidence-New/stage3/epoch-27.pth" \
        --device cuda \
        --input-source "/run/media/caig/2T-HDD/Thesis/Synthetic/video/HR/GT/videomatte_1920x1080/videomatte_motion/${i}/com" \
        --downsample-ratio 0.25 \
        --output-type png_sequence \
        --output-composition "/run/media/caig/2T-HDD/Thesis/Synthetic/video/HR/PointRend-confidence-New/videomatte_1920x1080/videomatte_motion/${i}/com" \
        --output-alpha "/run/media/caig/2T-HDD/Thesis/Synthetic/video/HR/PointRend-confidence-New/videomatte_1920x1080/videomatte_motion/${i}/pha" \
        --output-foreground "/run/media/caig/2T-HDD/Thesis/Synthetic/video/HR/PointRend-confidence-New/videomatte_1920x1080/videomatte_motion/${i}/fgr" \
        --seq-chunk 5)
    $(python inference.py \
        --variant mobilenetv3 \
        --checkpoint "./checkpoint/PointRend-confidence-New/stage3/epoch-27.pth" \
        --device cuda \
        --input-source "/run/media/caig/2T-HDD/Thesis/Synthetic/video/HR/GT/videomatte_1920x1080/videomatte_static/${i}/com" \
        --downsample-ratio 0.25 \
        --output-type png_sequence \
        --output-composition "/run/media/caig/2T-HDD/Thesis/Synthetic/video/HR/PointRend-confidence-New/videomatte_1920x1080/videomatte_static/${i}/com" \
        --output-alpha "/run/media/caig/2T-HDD/Thesis/Synthetic/video/HR/PointRend-confidence-New/videomatte_1920x1080/videomatte_static/${i}/pha" \
        --output-foreground "/run/media/caig/2T-HDD/Thesis/Synthetic/video/HR/PointRend-confidence-New/videomatte_1920x1080/videomatte_static/${i}/fgr" \
        --seq-chunk 5)
done
