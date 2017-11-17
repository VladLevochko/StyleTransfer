#!/usr/bin/env bash
python runner_squeezenet.py\
    --content_image styles/tubingen.jpg\
    --content_image_size 256\
    --style_image styles/la_muse.jpg\
    --style_image_size 256\
    --output_image generated/tub_muse_256.jpg
#python runner_squeezenet.py\
#    --content_image styles/tubingen.jpg\
#    --content_image_size 256\
#    --style_image styles/rain_princess.jpg\
#    --style_image_size 256\
#    --output_image generated/tub_princess_256.jpg
#python runner_squeezenet.py\
#    --content_image styles/tubingen.jpg\
#    --content_image_size 256\
#    --style_image styles/stary_night.jpg\
#    --style_image_size 256\
#    --output_image generated/tub_snight_256.jpg
#python runner_squeezenet.py\
#    --content_image styles/tubingen.jpg\
#    --content_image_size 256\
#    --style_image styles/the_scream.jpg\
#    --style_image_size 256\
#    --output_image generated/tub_scream_256.jpg
#python runner_squeezenet.py\
#    --content_image styles/tubingen.jpg\
#    --content_image_size 256\
#    --style_image styles/wave_1.jpg\
#    --style_image_size 256\
#    --output_image generated/tub_wave_256.jpg

