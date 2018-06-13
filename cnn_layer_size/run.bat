echo "conv"
python show_conv_dim.py -F conv_layer.txt -H 32 -W 32 -D 0
echo "deconv"
python show_conv_dim.py -F deconv_layer.txt -H 4 -W 4 -D 1
pause