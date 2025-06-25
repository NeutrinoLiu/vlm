pys=(38 39 310)
cus=(113 114 115 116 117)
pyts=(1120 1121 1130 1131)

for py in "${pys[@]}"; do
  for cu in "${cus[@]}"; do
    for pyt in "${pyts[@]}"; do
      url="https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py${py}_cu${cu}_pyt${pyt}/download.html"
      status_code=$(curl -L -s -o /dev/null -w "%{http_code}" $url)
      if [ "$status_code" == "200" ]; then
        echo "Wheel available for Python $py, CUDA $cu, PyTorch $pyt: $url"
      fi
    done
  done
done
