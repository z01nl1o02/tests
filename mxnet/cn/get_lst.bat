@setlocal enabledelayedexpansion

set root=%1
echo image root path: "!root!"
::label id is sorted by class name 
::check im2rec.py
::           dirs.sort()
::            files.sort()
python im2rec.py train !root!\train\ --recursive true --list true
python im2rec.py test !root!\test\ --recursive true --list true

::python im2rec.py train !root!\train\ --num-thread 2
::python im2rec.py test !root!\test\ --num-thread 2

@endlocal
