@setlocal enabledelayedexpansion

set root=%1
echo image root path: "!root!"

python im2rec.py train !root!\train\ --num-thread 2
python im2rec.py test !root!\test\ --num-thread 2

@endlocal
