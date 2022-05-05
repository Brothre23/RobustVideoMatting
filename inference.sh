for i in $(seq -f "%04g" 0 24)
do 
    $(python inference.py \
        --variant mobilenetv3 \
        --checkpoint "./checkpoint/2022.05.01/stage2/epoch-19.pth" \
        --device cuda \
        --input-source "../../下載/result/05.03-seg-resnet101/videomatte_gt/videomatte_512x288/videomatte_motion/${i}/com" \
        --alpha-source "../../下載/result/05.03-seg-resnet101/videomatte_gt/videomatte_512x288/videomatte_motion/${i}/pha" \
        --downsample-ratio 1.0 \
        --output-type png_sequence \
        --output-composition "../../下載/result/05.03-seg-resnet101/videomatte_pr/videomatte_512x288/videomatte_motion/${i}/com" \
        --output-alpha "../../下載/result/05.03-seg-resnet101/videomatte_pr/videomatte_512x288/videomatte_motion/${i}/pha" \
        --output-foreground "../../下載/result/05.03-seg-resnet101/videomatte_pr/videomatte_512x288/videomatte_motion/${i}/fgr" \
        --seq-chunk 5)
    $(python inference.py \
        --variant mobilenetv3 \
        --checkpoint "./checkpoint/2022.05.01/stage2/epoch-19.pth" \
        --device cuda \
        --input-source "../../下載/result/05.03-seg-resnet101/videomatte_gt/videomatte_512x288/videomatte_static/${i}/com" \
        --alpha-source "../../下載/result/05.03-seg-resnet101/videomatte_gt/videomatte_512x288/videomatte_static/${i}/pha" \
        --downsample-ratio 1.0 \
        --output-type png_sequence \
        --output-composition "../../下載/result/05.03-seg-resnet101/videomatte_pr/videomatte_512x288/videomatte_static/${i}/com" \
        --output-alpha "../../下載/result/05.03-seg-resnet101/videomatte_pr/videomatte_512x288/videomatte_static/${i}/pha" \
        --output-foreground "../../下載/result/05.03-seg-resnet101/videomatte_pr/videomatte_512x288/videomatte_static/${i}/fgr" \
        --seq-chunk 5)
done
