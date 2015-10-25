echo off
setlocal enabledelayedexpansion
set rdir=.\
mkdir B\
for /f "delims=" %%n in (B.txt) do (
echo %%n
move !rdir!"%%n.jpg" B\
)



mkdir A\
for /f "delims=" %%n in (A.txt) do (
echo %%n
move !rdir!"%%n.jpg" A\
)
endlocal
echo on
